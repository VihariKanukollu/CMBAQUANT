import argparse
import json
from typing import List, Tuple, Optional, Dict, Any

import torch
from lean_dojo import Dojo, LeanGitRepo, Theorem

from pretrain import PretrainConfig, init_train_state
import yaml
import os


class CMBAProver:
    """Minimal prover wrapper around CMBA model for LeanDojo.

    NOTE:
    - This is a starter harness. You likely want to improve `encode_state` and
      `propose_tactics` to use a curated tactic vocabulary or multi-token decoding.
    - Works best with a chat-trained CMBA (text tokenizer). For puzzle-trained
      models, you must adapt encoding to match training format.
    """

    def __init__(self, base_model: torch.nn.Module, tokenizer_id: Optional[str], max_model_steps: int = 128):
        self.model = base_model.eval().cuda()
        self.max_model_steps = int(max_model_steps)
        self.tokenizer = None
        if tokenizer_id:
            try:
                from transformers import AutoTokenizer  # type: ignore
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token or "<|pad|>"
            except Exception as e:
                print(f"[Warn] Failed to load tokenizer '{tokenizer_id}': {e}")

    def _encode_text(self, text: str, max_len: int) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for text encoding. Provide --tokenizer_id for chat-trained models.")
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = enc["input_ids"].to(torch.long).cuda()
        batch = {
            "inputs": inputs,
            # For CMBA, puzzle identifiers exist; set to zeros when unused
            "puzzle_identifiers": torch.zeros((inputs.shape[0],), dtype=torch.long, device=inputs.device),
        }
        return batch

    def encode_state(self, state, max_len: int) -> Dict[str, torch.Tensor]:
        # Basic pretty-printed state; improve with goal/context formatting as needed
        txt = f"GOAL:\n{str(state)}\n"
        return self._encode_text(txt, max_len=max_len)

    @torch.inference_mode()
    def _forward_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        carry = self.model.initial_carry(batch)
        for _ in range(self.max_model_steps):
            carry, outputs = self.model(carry=carry, batch=batch)
            if carry.halted.all():
                break
        return outputs["logits"]  # [B, S, V]

    @torch.inference_mode()
    def propose_tactics(self, state, top_k: int, max_len: int) -> List[Tuple[str, float]]:
        batch = self.encode_state(state, max_len=max_len)
        logits = self._forward_logits(batch)
        last = logits[0, -1]
        probs = torch.softmax(last, dim=-1)
        vals, idxs = torch.topk(probs, k=top_k)
        # Single-token tactic candidates (starter). Replace with curated mapping or multi-token decoding.
        if self.tokenizer is None:
            return []
        res: List[Tuple[str, float]] = []
        for v, i in zip(vals.tolist(), idxs.tolist()):
            tok = self.tokenizer.decode([i], skip_special_tokens=True).strip()
            if tok:
                res.append((tok, float(-v)))  # negative score for best-first queue
        return res

    def prove(self, theorem: Theorem, attempts: int = 32, timeout: int = 600, top_k: int = 10, max_len: int = 2048) -> Optional[List[str]]:
        with Dojo(theorem, timeout=timeout) as dojo:
            state = dojo.initialize()
            if state.is_proved():
                return []
            queue: List[Tuple[float, Any, List[str]]] = [(0.0, state, [])]
            seen = set()
            tried = 0
            while queue and tried < attempts:
                score, s, trail = queue.pop(0)
                for tactic, tscore in self.propose_tactics(s, top_k=top_k, max_len=max_len):
                    try:
                        ns = dojo.step(s, tactic)
                        key = str(ns)
                        if ns.is_proved():
                            return trail + [tactic]
                        if key in seen:
                            continue
                        seen.add(key)
                        queue.append((score + tscore, ns, trail + [tactic]))
                    except Exception:
                        pass
                queue.sort(key=lambda x: x[0])
                tried += 1
        return None


def load_model_from_checkpoint(ckpt_path: str):
    """Load CMBA model from checkpoint, adapting vocab/ids to match weights.

    Handles cases where saved config lacks training vocab size or puzzle id count
    by inferring them from the checkpoint tensor shapes.
    """
    # Load config
    cfg_file = os.path.join(os.path.dirname(ckpt_path), "all_config.yaml")
    with open(cfg_file, "r") as f:
        cfg = PretrainConfig(**yaml.safe_load(f))

    # Inspect checkpoint first to infer dims
    raw_sd = torch.load(ckpt_path, map_location="cpu")
    # Unwrap compiled prefix if needed
    sd_keys = list(raw_sd.keys())
    has_prefix = any(k.startswith("_orig_mod.") for k in sd_keys)
    sd = {k.removeprefix("_orig_mod."): v for k, v in raw_sd.items()} if has_prefix else raw_sd

    # Infer vocab size and puzzle ids from tensors
    vocab_size = None
    if "model.inner.embed_tokens.embedding_weight" in sd:
        vocab_size = sd["model.inner.embed_tokens.embedding_weight"].shape[0]
    elif "model.inner.lm_head.weight" in sd:
        vocab_size = sd["model.inner.lm_head.weight"].shape[0]
    if vocab_size is None:
        vocab_size = int(cfg.arch.__pydantic_extra__.get("vocab_size", 32000))

    num_puzzle_ids = 1
    if "model.inner.puzzle_emb_table.weights" in sd:
        num_puzzle_ids = sd["model.inner.puzzle_emb_table.weights"].shape[0]

    seq_len = int(cfg.arch.__pydantic_extra__.get("seq_len", 2048))

    # Build model with inferred metadata
    meta = type("M", (), {
        "total_groups": 1,
        "mean_puzzle_examples": 1,
        "seq_len": seq_len,
        "vocab_size": int(vocab_size),
        "num_puzzle_identifiers": int(num_puzzle_ids),
    })()

    train_state = init_train_state(cfg, train_metadata=meta, world_size=1)

    # Load weights (assign=True requires matching names; shapes now agree)
    train_state.model.load_state_dict(sd, assign=True)

    base_model = train_state.model.model  # unwrap loss head
    return base_model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint file (from checkpoints/.../step_*)")
    ap.add_argument("--repo_url", required=True, help="Lean Git repo URL (e.g., miniF2F)")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--theorems_json", required=True, help="JSON file with a list of theorem names (key 'test' or a flat list)")
    ap.add_argument("--tokenizer_id", required=False, default=None, help="HF tokenizer id for chat-trained CMBA")
    ap.add_argument("--attempts", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--max_input_len", type=int, default=2048)
    args = ap.parse_args()

    base_model, cfg = load_model_from_checkpoint(args.checkpoint)
    prover = CMBAProver(base_model=base_model, tokenizer_id=args.tokenizer_id, max_model_steps=int(cfg.arch.__pydantic_extra__.get("halt_max_steps", 128)))

    # Load theorem names
    with open(args.theorems_json, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        theorems: List[str] = payload.get("test") or payload.get("theorems") or []
    else:
        theorems = list(payload)
    assert len(theorems), "No theorems found in JSON. Provide a flat list or {'test': [...]}"

    repo = LeanGitRepo(args.repo_url, args.branch)

    proved = 0
    for name in theorems:
        # Newer LeanDojo exposes Theorem(repo, name) directly
        thm = Theorem(repo, name)
        proof = prover.prove(thm, attempts=args.attempts, timeout=args.timeout, top_k=args.top_k, max_len=args.max_input_len)
        print(("✓" if proof else "✗"), name)
        proved += int(proof is not None)

    print(f"Pass@1: {100.0 * proved / len(theorems):.1f}%  ({proved}/{len(theorems)})")


if __name__ == "__main__":
    main()


