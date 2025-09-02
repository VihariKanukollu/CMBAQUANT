from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
except Exception:
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        # FIXME: Assuming the batch is always full
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Optional MCP regularizer
        mcp_cost = outputs.get("mcp_cost") if isinstance(outputs, dict) else None
        if mcp_cost is not None:
            metrics["mcp_cost"] = mcp_cost.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        if mcp_cost is not None:
            total_loss = total_loss + mcp_cost
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


class ACTDistillLossHead(nn.Module):
    def __init__(self, model: nn.Module, teacher_model_id: str, loss_type: str = "softmax_cross_entropy", distill_temp: float = 1.0, kl_weight: float = 1.0, ce_weight: float = 1.0, trust_remote_code: bool = True):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.distill_temp = float(distill_temp)
        self.kl_weight = float(kl_weight)
        self.ce_weight = float(ce_weight)

        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers is required for ACTDistillLossHead")

        # Load teacher
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=trust_remote_code)
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        self.teacher.eval()

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    @torch.no_grad()
    def _teacher_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Teacher expects input_ids shaped [B, S]
        outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.to(torch.float32)

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Compute student CE loss (masked)
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

        student_logits = outputs["logits"].to(torch.float32)
        ce_loss = (self.loss_fn(student_logits, labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()

        # Teacher KL loss on supervised tokens only
        # Use the same input_ids the student saw
        input_ids = new_carry.current_data["inputs"].to(torch.long)
        pad_id = self.teacher_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        attention_mask = (input_ids != pad_id).to(torch.long)

        with torch.no_grad():
            teacher_logits = self._teacher_logits(input_ids=input_ids, attention_mask=attention_mask)

        # Align shapes and mask
        T = max(self.distill_temp, 1e-4)
        t_log_probs = torch.log_softmax(teacher_logits / T, dim=-1)
        s_log_probs = torch.log_softmax(student_logits / T, dim=-1)
        # KL(student || teacher) = sum p_s * (log p_s - log p_t)
        s_probs = s_log_probs.exp()
        token_kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)  # [B, S]
        token_kl = torch.where(mask, token_kl, torch.zeros_like(token_kl))
        kl_loss = (T * T) * (token_kl / loss_divisor.squeeze(-1)).sum()

        # Q-learning heads retained
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], (mask.sum(-1) > 0).to(outputs["q_halt_logits"].dtype), reduction="sum")
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

        # Metrics
        with torch.no_grad():
            is_correct = mask & (torch.argmax(student_logits, dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "lm_loss": ce_loss.detach(),
                "kl_loss": kl_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }
            if "target_q_continue" in outputs:
                metrics["q_continue_loss"] = torch.as_tensor(q_continue_loss).detach()

        # Optional MCP regularizer
        mcp_cost = outputs.get("mcp_cost") if isinstance(outputs, dict) else None
        if mcp_cost is not None:
            metrics["mcp_cost"] = mcp_cost.detach()

        # Total loss
        total_loss = self.ce_weight * ce_loss + self.kl_weight * kl_loss + 0.5 * (q_halt_loss + (q_continue_loss if isinstance(q_continue_loss, torch.Tensor) else 0))
        if mcp_cost is not None:
            total_loss = total_loss + mcp_cost

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
