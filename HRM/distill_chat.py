import argparse
import os
from typing import Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.functions import load_model_class
from chat_dataset import ChatDataset, ChatDatasetConfig
from models.losses import ACTDistillLossHead


def build_model_config(args, tokenizer) -> Dict[str, Any]:
    return dict(
        batch_size=args.global_batch_size,
        seq_len=args.max_seq_len,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=tokenizer.vocab_size,

        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,

        H_layers=args.H_layers,
        L_layers=args.L_layers,

        hidden_size=args.hidden_size,
        expansion=args.expansion,
        num_heads=args.num_heads,
        pos_encodings="rope",

        rms_norm_eps=1e-5,
        rope_theta=10000.0,

        halt_max_steps=1,
        halt_exploration_prob=0.0,

        forward_dtype=("bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"),

        # Ensure causal attention in chat model
        causal=True,

        # Disable optional features by default for first pass
        puzzle_emb_type="table",
        puzzle_emb_cache_eval=True,
        puzzle_emb_cache_size=0,

        quantum_gate_enabled=False,
        act_sched_enabled=False,
        per_head_bias_enabled=False,
        token_routing_enabled=False,
        pqc_shared=False,
        film_enabled=False,
        rope_phase_bias_enabled=False,
        mcp_enabled=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--teacher_model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--arch_name", type=str, default="models.hrm.hrm_quant_v1_chat@HierarchicalReasoningModel_ACTV1")
    parser.add_argument("--global_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--distill_temp", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default="checkpoints/distill_chat")

    # Architecture knobs (keep modest defaults)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--expansion", type=float, default=4.0)
    parser.add_argument("--H_layers", type=int, default=6)
    parser.add_argument("--L_layers", type=int, default=6)
    parser.add_argument("--H_cycles", type=int, default=1)
    parser.add_argument("--L_cycles", type=int, default=1)

    args = parser.parse_args()

    # Teacher tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        # Ensure a pad token exists
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset and loader
    ds = ChatDataset(ChatDatasetConfig(
        jsonl_path=args.jsonl_path,
        tokenizer=tokenizer,
        global_batch_size=args.global_batch_size,
        max_seq_len=args.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    ))
    loader = DataLoader(ds, batch_size=None, num_workers=0)

    # Model
    model_cls = load_model_class(args.arch_name)
    model_cfg = build_model_config(args, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.device(device):
        base_model: nn.Module = model_cls(model_cfg)
        model = ACTDistillLossHead(
            base_model,
            teacher_model_id=args.teacher_model_id,
            distill_temp=args.distill_temp,
            kl_weight=args.kl_weight,
            ce_weight=args.ce_weight,
        )
        model = model.to(device)

    # Optimizer
    # Only optimize CUDA params (teacher stays on its own device map)
    cuda_params = [p for p in model.parameters() if p.device.type == "cuda"]
    if not cuda_params:
        cuda_params = list(model.parameters())
    optim = torch.optim.AdamW(cuda_params, lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        for _set_name, batch, _gbs in loader:
            # To device
            batch = {k: v.to(device) for k, v in batch.items()}
            # Reset carry per batch for simplicity
            carry = model.initial_carry(batch)
            carry, loss, metrics, _, _ = model(return_keys=[], carry=carry, batch=batch)

            (loss / max(args.global_batch_size, 1)).backward()
            optim.step()
            optim.zero_grad()

            step += 1
            if step % 10 == 0:
                try:
                    acc = float(metrics.get("accuracy", torch.tensor(0.0)))
                    lm = float(metrics.get("lm_loss", torch.tensor(0.0)))
                    kl = float(metrics.get("kl_loss", torch.tensor(0.0)))
                except Exception:
                    acc = lm = kl = 0.0
                print(f"step {step} | loss {float(loss):.4f} | lm {lm:.4f} | kl {kl:.4f} | acc {acc:.4f}")

            if step % 1000 == 0:
                ckpt_path = os.path.join(args.out_dir, f"step_{step}.pt")
                torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()


