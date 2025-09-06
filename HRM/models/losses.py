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

        # Optional Ponder KL regularizer (computed by model when enabled)
        ponder_kl = outputs.get("ponder_kl") if isinstance(outputs, dict) else None
        ponder_kl_weight = float(getattr(getattr(self.model, "config", object), "ponder_kl_div_loss_weight", 0.0))
        if ponder_kl is not None and ponder_kl_weight > 0.0:
            metrics["ponder_kl"] = ponder_kl.detach()

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Optional MCP regularizer
        mcp_cost = outputs.get("mcp_cost") if isinstance(outputs, dict) else None
        if mcp_cost is not None:
            metrics["mcp_cost"] = mcp_cost.detach()
        # Add per-feature costs from MCP gates (if exposed via inner extras)
        ntm_extras = getattr(getattr(self.model, 'inner', object), '_ntm_extras', None)
        if isinstance(ntm_extras, dict):
            for k, v in ntm_extras.items():
                if k.startswith('mcp_cost_') and torch.is_tensor(v):
                    metrics[k] = v.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        if ponder_kl is not None and ponder_kl_weight > 0.0:
            # Optionally scale ponder KL by MCP ponder gate mean (if exposed via extras)
            mcp_extras = getattr(getattr(self.model, 'inner', object), '_ntm_extras', None)
            ponder_gate = None
            if isinstance(mcp_extras, dict) and ('gate_ponder' in mcp_extras):
                ponder_gate = mcp_extras['gate_ponder']
            scale = ponder_gate if (torch.is_tensor(ponder_gate)) else 1.0
            total_loss = total_loss + (ponder_kl_weight * (ponder_kl * scale))
        # Optional NTM/MCP extras: add ntm_reg and surface metrics (gate means, costs)
        ntm_extras = getattr(getattr(self.model, 'inner', object), '_ntm_extras', None)
        if isinstance(ntm_extras, dict):
            if 'ntm_reg' in ntm_extras:
                total_loss = total_loss + ntm_extras['ntm_reg']
            for k, v in ntm_extras.items():
                if torch.is_tensor(v):
                    metrics[k] = v.detach()
            # If MCP provided a total cost (mcp_cost_total), include it
            mcp_total = ntm_extras.get('mcp_cost_total') if isinstance(ntm_extras, dict) else None
            if torch.is_tensor(mcp_total):
                total_loss = total_loss + mcp_total
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
        # Keep divisor shape [B, 1] to broadcast across sequence length safely
        kl_loss = (T * T) * (token_kl / loss_divisor).sum()

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

        # Optional Ponder KL regularizer (computed by model when enabled)
        ponder_kl = outputs.get("ponder_kl") if isinstance(outputs, dict) else None
        ponder_kl_weight = float(getattr(getattr(self.model, "config", object), "ponder_kl_div_loss_weight", 0.0))
        if ponder_kl is not None and ponder_kl_weight > 0.0:
            metrics["ponder_kl"] = ponder_kl.detach()

        # Optional MCP regularizer
        mcp_cost = outputs.get("mcp_cost") if isinstance(outputs, dict) else None
        if mcp_cost is not None:
            metrics["mcp_cost"] = mcp_cost.detach()

        # NTM regularizer and metrics (if inner attached extras)
        ntm_extras = getattr(getattr(self.model, 'inner', object), '_ntm_extras', None)
        if isinstance(ntm_extras, dict):
            if 'ntm_reg' in ntm_extras:
                metrics['ntm_reg'] = ntm_extras['ntm_reg'].detach() if torch.is_tensor(ntm_extras['ntm_reg']) else torch.as_tensor(ntm_extras['ntm_reg'])
            for k, v in ntm_extras.items():
                if k != 'ntm_reg' and torch.is_tensor(v):
                    metrics[k] = v

        # Total loss
        total_loss = self.ce_weight * ce_loss + self.kl_weight * kl_loss + 0.5 * (q_halt_loss + (q_continue_loss if isinstance(q_continue_loss, torch.Tensor) else 0))
        if isinstance(ntm_extras, dict) and ('ntm_reg' in ntm_extras):
            total_loss = total_loss + (ntm_extras['ntm_reg'] if torch.is_tensor(ntm_extras['ntm_reg']) else torch.as_tensor(ntm_extras['ntm_reg']))
        if ponder_kl is not None and ponder_kl_weight > 0.0:
            total_loss = total_loss + ponder_kl_weight * ponder_kl
        if mcp_cost is not None:
            total_loss = total_loss + mcp_cost

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


class ACTDistillM2LossHead(nn.Module):
    """Distillm-2 v2 style KD on chosen/rejected pairs.

    Expects the batch to contain concatenated pairs in order [pos0, neg0, pos1, neg1, ...].
    """
    def __init__(self, model: nn.Module, teacher_model_id: str, *, base_alpha_1: float = 0.1, base_alpha_2: float = 0.1, trust_remote_code: bool = True):
        super().__init__()
        self.model = model
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers is required for ACTDistillM2LossHead")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        ).eval()
        self.base_alpha_1 = float(base_alpha_1)
        self.base_alpha_2 = float(base_alpha_2)

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    @staticmethod
    def _gather_label_logps(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        vocab_logps = logits.log_softmax(-1)
        per_tok = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return (per_tok * mask).sum(-1) / mask.sum(-1).clamp_min(1)

    def forward(self, return_keys: Sequence[str], **model_kwargs):
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"].to(torch.long)
        student_logits = outputs["logits"].to(torch.float32)

        # Mask/shift
        lbl = labels[:, 1:].clone()
        stu = student_logits[:, :-1, :]
        mask = (lbl != IGNORE_LABEL_ID).to(stu.dtype)

        # Teacher logits
        with torch.no_grad():
            tea_logits = self.teacher(input_ids=new_carry.current_data["inputs"].to(torch.long), attention_mask=(new_carry.current_data["inputs"] != 0)).logits.to(torch.float32)
            tea = tea_logits[:, :-1, :]

        # Split pos/neg
        pos_idx = torch.arange(0, stu.shape[0], 2, device=stu.device)
        neg_idx = torch.arange(1, stu.shape[0], 2, device=stu.device)

        def per_position_kls(stu_logits, tea_logits, lbls, msk, *, alpha1, alpha2):
            vocab_logps = stu_logits.log_softmax(-1)
            tea_logps = tea_logits.log_softmax(-1)
            # Mixes
            log_alpha1 = torch.log(alpha1)
            log_1m_a1 = torch.log(1 - alpha1)
            mix1 = torch.logsumexp(torch.stack([log_alpha1 + tea_logps, log_1m_a1 + vocab_logps], dim=0), dim=0)
            tea_pos_kl = (tea_logps.exp() * (tea_logps - mix1)).sum(-1)

            log_alpha2 = torch.log(alpha2)
            log_1m_a2 = torch.log(1 - alpha2)
            mix2 = torch.logsumexp(torch.stack([log_1m_a2 + tea_logps, log_alpha2 + vocab_logps.detach()], dim=0), dim=0)
            ref_pos_kl = (vocab_logps.exp() * (vocab_logps - mix2)).sum(-1)
            # mean over supervised tokens
            tea_pos_kl = (tea_pos_kl * msk).sum(-1) / msk.sum(-1).clamp_min(1)
            ref_pos_kl = (ref_pos_kl * msk).sum(-1) / msk.sum(-1).clamp_min(1)
            return tea_pos_kl, ref_pos_kl

        # Sentence-level anchors
        pos_logp_s = self._gather_label_logps(stu[pos_idx], lbl[pos_idx], mask[pos_idx])
        pos_logp_t = self._gather_label_logps(tea[pos_idx], lbl[pos_idx], mask[pos_idx])
        neg_logp_s = self._gather_label_logps(stu[neg_idx], lbl[neg_idx], mask[neg_idx])
        neg_logp_t = self._gather_label_logps(tea[neg_idx], lbl[neg_idx], mask[neg_idx])

        logps_logqs = (pos_logp_t - pos_logp_s).exp()
        logqs_logps = (neg_logp_s - neg_logp_t).exp()

        alpha1 = torch.clamp(1 - (1 - self.base_alpha_1) * (1.0 / (logps_logqs + 1e-5)), min=1e-2, max=self.base_alpha_1).view(-1, 1, 1)
        alpha2 = torch.clamp(1 - (1 - self.base_alpha_2) * (1.0 / (logqs_logps + 1e-5)), min=1e-2, max=self.base_alpha_2).view(-1, 1, 1)

        pos_tkl, _ = per_position_kls(stu[pos_idx], tea[pos_idx], lbl[pos_idx], mask[pos_idx], alpha1=alpha1, alpha2=alpha2)
        _, neg_rkl = per_position_kls(stu[neg_idx], tea[neg_idx], lbl[neg_idx], mask[neg_idx], alpha1=alpha1, alpha2=alpha2)

        beta = 1.0
        loss = ((2 - beta) * pos_tkl.mean() + beta * neg_rkl.mean())

        # Keep Q losses for ACT heads as regularizers
        q_halt_loss = torch.tensor(0.0, device=loss.device)
        q_continue_loss = torch.tensor(0.0, device=loss.device)
        if "q_halt_logits" in outputs:
            # treat any supervised sequence as positive
            seq_pos = (mask.sum(-1) > 0).to(outputs["q_halt_logits"].dtype)
            q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_pos, reduction="mean")
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="mean")

        total = loss + 0.5 * (q_halt_loss + q_continue_loss)
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, total, {"count": torch.tensor(labels.shape[0], device=labels.device), "distillm2_loss": loss.detach()}, detached_outputs, new_carry.halted.all()
