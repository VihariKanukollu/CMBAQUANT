#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Dict, Any

import torch

# Ensure local package imports work when run from repo root
sys.path.append("HRM")
from utils.functions import load_model_class  # type: ignore
from transformers import AutoTokenizer  # type: ignore
import yaml


def _build_model_cfg_from_yaml(pretrain_cfg: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    arch_cfg = dict(pretrain_cfg.get("arch", {}))
    # Remove non-model keys
    arch_cfg.pop("name", None)
    arch_cfg.pop("loss", None)
    # Base cfg
    model_cfg = {
        **arch_cfg,
        "batch_size": 1,
        "vocab_size": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        "seq_len": int(pretrain_cfg.get("chat_max_seq_len", 2048)),
        "num_puzzle_identifiers": 1,
        "causal": True,
    }
    return model_cfg


def _load_from_raw_state(ckpt_path: str, teacher_model_id: str, device: str):
    # Load tokenizer first to compute vocab size
    tok = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Resolve YAML config saved during training
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    yaml_path = os.path.join(ckpt_dir, "all_config.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Cannot find training config YAML next to checkpoint: {yaml_path}")
    with open(yaml_path, "r") as f:
        pretrain_cfg = yaml.safe_load(f)

    # Model class from saved arch
    arch_name = pretrain_cfg.get("arch", {}).get("name", "models.hrm.hrm_quant_v1_chat@HierarchicalReasoningModel_ACTV1")
    Model = load_model_class(arch_name)
    model_cfg = _build_model_cfg_from_yaml(pretrain_cfg, tok)
    model = Model(model_cfg).to(device).eval()

    # Load state dict; strip 'model.' prefix if present (loss-head wrapper)
    raw_sd = torch.load(ckpt_path, map_location=device)
    if isinstance(raw_sd, dict) and all(isinstance(k, str) for k in raw_sd.keys()):
        has_prefixed = any(k.startswith("model.") for k in raw_sd.keys())
        if has_prefixed:
            base_sd = {}
            for k, v in raw_sd.items():
                if k.startswith("model."):
                    base_sd[k[len("model."):]] = v
            missing, unexpected = model.load_state_dict(base_sd, strict=False)
        else:
            missing, unexpected = model.load_state_dict(raw_sd, strict=False)
        if len(unexpected):
            # ignore non-model keys
            pass
    else:
        raise RuntimeError("Unexpected checkpoint format: expected a state_dict mapping")

    cfg = {
        "seq_len": model_cfg["seq_len"],
    }
    return model, tok, cfg, device


def load_student(ckpt_path: str, teacher_model_id: str, device: str = "cuda"):
    device = device if (device in {"cuda", "cpu"}) else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Distilled format: {"model_cfg": ..., "student_state_dict": ...}
    if isinstance(ckpt, dict) and ("model_cfg" in ckpt and "student_state_dict" in ckpt):
        cfg = ckpt["model_cfg"]
        arch_name = cfg.get("name", "models.hrm.hrm_quant_v1_chat@HierarchicalReasoningModel_ACTV1") if isinstance(cfg, dict) else "models.hrm.hrm_quant_v1_chat@HierarchicalReasoningModel_ACTV1"
        Model = load_model_class(arch_name)
        model = Model(cfg).to(device).eval()
        model.load_state_dict(ckpt["student_state_dict"], strict=False)

        tok = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return model, tok, cfg, device

    # Training checkpoint (raw state_dict)
    return _load_from_raw_state(ckpt_path, teacher_model_id, device)


@torch.no_grad()
def generate_reply(model, tok, cfg: Dict, device: str, messages: List[Dict[str, str]], *,
                   max_new_tokens: int = 256, temperature: float = 0.8, top_p: float = 0.9) -> str:
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")[0].to(device)
    seq_len = int(cfg["seq_len"])  # sliding window length used by the student
    eos_id = tok.eos_token_id

    # Rolling decode (no KV cache). Window the last seq_len tokens at each step.
    context = ids.tolist()
    for _ in range(max_new_tokens):
        window = context[-seq_len:]

        # Build fixed-length input the student expects (right-padding, as in training)
        inp = torch.full((1, seq_len), tok.pad_token_id, dtype=torch.int32, device=device)
        inp[0, :len(window)] = torch.tensor(window, dtype=torch.int32, device=device)

        batch = {
            "inputs": inp,
            "puzzle_identifiers": torch.zeros((1,), dtype=torch.int32, device=device),
        }
        carry = model.initial_carry(batch)
        carry, out = model(carry, batch)

        logits = out["logits"][0]
        pos = min(len(window) - 1, seq_len - 1)
        probs = torch.softmax(logits[pos] / max(temperature, 1e-5), dim=-1)

        if 0.0 < top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            csum = torch.cumsum(sp, 0)
            k = int((csum <= top_p).sum().item())
            k = max(k, 1)
            keep = si[:k]
            masked = torch.zeros_like(probs).index_fill(0, keep, 1.0)
            probs = masked / masked.sum()

        nxt = torch.multinomial(probs, 1).item()
        context.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break

    return tok.decode(context[len(ids):], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description="Chat with the distilled HRM model")
    ap.add_argument("--ckpt", required=True, help="Path to distilled checkpoint (step_*.pt)")
    ap.add_argument("--teacher", default="mistralai/Mistral-7B-Instruct-v0.2", help="Teacher/tokenizer id")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--prompt", default=None, help="Optional single-turn prompt (non-interactive)")
    args = ap.parse_args()

    model, tok, cfg, device = load_student(args.ckpt, args.teacher, device=args.device)

    if args.prompt is not None:
        msgs = [{"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt}]
        out = generate_reply(model, tok, cfg, device, msgs,
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature,
                             top_p=args.top_p)
        print(out)
        return

    # Interactive loop
    print("HRM chat ready. Type 'exit' to quit.\n")
    history: List[Dict[str, str]] = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    try:
        while True:
            user = input("User: ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            history.append({"role": "user", "content": user})
            reply = generate_reply(model, tok, cfg, device, history,
                                   max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature,
                                   top_p=args.top_p)
            history.append({"role": "assistant", "content": reply})
            print(f"Assistant: {reply}\n")
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()


