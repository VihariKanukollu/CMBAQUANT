#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import torch

# Local imports
from chat_demo import load_student, generate_reply  # type: ignore
from tqdm import tqdm


def _read_prompts_from_teacher_json(path: str) -> List[str]:
    """Read prompts from a teacher outputs JSON file written by distillm-2/generate_vllm.py.

    The expected format is a JSON list of objects with at least keys: 'prompt', 'generated_text'.
    """
    with open(path, "r") as f:
        data = json.load(f)
    prompts: List[str] = []
    for rec in data:
        p = rec.get("prompt")
        if isinstance(p, str) and len(p.strip()):
            prompts.append(p)
    return prompts


def _collect_teacher_prompts(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_dir():
        files = sorted([str(x) for x in p.glob("*.json")])
    else:
        files = [str(p)]
    prompts: List[str] = []
    for fp in files:
        try:
            prompts.extend(_read_prompts_from_teacher_json(fp))
        except Exception:
            continue
    return prompts


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Generate student outputs (HRM) for distillation pairs")
    ap.add_argument("--ckpt", required=True, help="Path to HRM checkpoint (state_dict)")
    ap.add_argument("--teacher_tokenizer_id", required=True, help="Tokenizer id used for chat template")
    ap.add_argument("--teacher_outputs", required=True, help="Teacher outputs file or directory of files (*.json)")
    ap.add_argument("--output_dir", required=True, help="Where to save student outputs JSON")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_prompts", type=int, default=0, help="If >0, limit number of prompts for quick runs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok, cfg, device = load_student(args.ckpt, args.teacher_tokenizer_id, device=args.device)
    prompts = _collect_teacher_prompts(args.teacher_outputs)
    if args.max_prompts and args.max_prompts > 0:
        prompts = prompts[: int(args.max_prompts)]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "student_outputs.json")

    results: List[Dict[str, str]] = []
    for i, prompt in enumerate(tqdm(prompts, desc="Generating (student)", unit="prompt")):
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": prompt},
        ]
        text = generate_reply(
            model,
            tok,
            cfg,
            device,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        results.append({"prompt": prompt, "generated_text": text})
        # Periodic checkpointing (JSONL) for long runs
        if (i + 1) % 200 == 0:
            with open(os.path.join(args.output_dir, "student_outputs.partial.jsonl"), "a") as fw:
                for r in results[-200:]:
                    fw.write(json.dumps(r) + "\n")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("saved:", out_path, "records:", len(results))


if __name__ == "__main__":
    main()


