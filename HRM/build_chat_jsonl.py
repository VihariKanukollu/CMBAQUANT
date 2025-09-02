import argparse
import json
import os
from typing import List, Dict

from datasets import load_dataset
from itertools import islice


def save_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def convert_sharegpt(max_samples: int) -> List[Dict]:
    # Try multiple known ShareGPT-style sources; skip gracefully if unavailable
    candidates = [
        ("lmsys/sharegpt", "conversations", "from", "value"),
        ("anon8231489123/ShareGPT_Vicuna_unfiltered", "conversations", "from", "value"),
    ]
    for ds_name, conv_field, role_key, text_key in candidates:
        try:
            ds = load_dataset(ds_name, split="train")
        except Exception:
            ds = None
        if ds is None:
            continue
        rows: List[Dict] = []
        for ex in ds:
            conv = ex.get(conv_field) or ex.get("messages") or ex.get("conversations")
            if not conv:
                continue
            messages = []
            for turn in conv:
                role = turn.get(role_key) or turn.get("role")
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant"):
                    role = "assistant"
                elif role not in ("system", "user", "assistant"):
                    role = "user"
                content = turn.get(text_key) or turn.get("content") or turn.get("value") or ""
                if not content:
                    continue
                messages.append({"role": role, "content": content})
            if len(messages) >= 2:
                rows.append({"messages": messages})
            if len(rows) >= max_samples:
                break
        if len(rows):
            return rows
    print("[build_chat_jsonl] Warning: ShareGPT sources unavailable; skipping this shard.")
    return []


def convert_openorca(max_samples: int) -> List[Dict]:
    rows: List[Dict] = []
    # Prefer streaming to avoid storing multi-GB parquet files
    try:
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        iterator = islice(ds, max_samples)
    except Exception:
        # Tiny fallback: non-streaming small slice
        ds = load_dataset("Open-Orca/OpenOrca", split="train[:1%]")
        iterator = iter(ds)

    for ex in iterator:
        sys_prompt = ex.get("system_prompt") or "You are a helpful assistant."
        question = ex.get("question") or ""
        response = ex.get("response") or ex.get("response_gpt4") or ""
        if not question or not response:
            continue
        rows.append({
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
        })
        if len(rows) >= max_samples:
            break
    return rows


def convert_ultrachat(max_samples: int) -> List[Dict]:
    rows: List[Dict] = []
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        iterator = islice(ds, max_samples)
    except Exception:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:1%]")
        iterator = iter(ds)

    for ex in iterator:
        conv = ex.get("messages")
        if not conv:
            turns = ex.get("conversations")
            if isinstance(turns, list):
                conv = turns
        if not conv:
            continue
        messages = []
        for m in conv:
            role = m.get("role")
            content = m.get("content") or m.get("value") or ""
            if role not in ("system", "user", "assistant"):
                role = "user"
            if not content:
                continue
            messages.append({"role": role, "content": content})
        if len(messages) >= 2:
            rows.append({"messages": messages})
        if len(rows) >= max_samples:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/chat/medium.jsonl")
    ap.add_argument("--sharegpt", type=int, default=5000)
    ap.add_argument("--openorca", type=int, default=5000)
    ap.add_argument("--ultrachat", type=int, default=5000)
    args = ap.parse_args()

    all_rows: List[Dict] = []
    if args.sharegpt > 0:
        all_rows.extend(convert_sharegpt(args.sharegpt))
    if args.openorca > 0:
        all_rows.extend(convert_openorca(args.openorca))
    if args.ultrachat > 0:
        all_rows.extend(convert_ultrachat(args.ultrachat))

    save_jsonl(args.out, all_rows)
    print(f"Saved {len(all_rows)} conversations to {args.out}")


if __name__ == "__main__":
    main()


