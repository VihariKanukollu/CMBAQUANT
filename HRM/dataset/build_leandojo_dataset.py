from typing import Dict, List, Tuple
import os
import json
import math

import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata  # type: ignore


cli = ArgParser()


class LeanDojoConvertConfig(BaseModel):
    # Path to a Hugging Face dataset saved via datasets.save_to_disk(...)
    input_dir: str = "data/leandojo"
    # Output directory in CMBA/HRM format expected by PuzzleDataset
    output_dir: str = "data/leandojo_act"

    # Sequence length for tactic-prediction windows
    seq_len: int = 64

    # Minimum number of tactics required to create at least one example
    min_tactics: int = 2

    # If true, merge validation into test split
    merge_val_into_test: bool = True


def _gather_train_vocab(ds_train) -> List[str]:
    vocab: List[str] = []
    seen: set = set()
    for ex in ds_train:
        tactics = ex.get("traced_tactics", []) or []
        for t in tactics:
            name = str(t.get("tactic", "")).strip()
            if not name:
                continue
            if name not in seen:
                seen.add(name)
                vocab.append(name)
    return vocab


def _encode_tactics(tactics: List[Dict[str, str]], tactic_to_id: Dict[str, int], unk_id: int) -> List[int]:
    tokens: List[int] = []
    for t in tactics:
        name = str(t.get("tactic", "")).strip()
        if not name:
            continue
        tokens.append(tactic_to_id.get(name, unk_id))
    return tokens


def _save_split(
    examples: List[Tuple[np.ndarray, np.ndarray, int]],
    seq_len: int,
    output_dir: str,
    set_name: str,
    num_puzzle_identifiers: int,
):
    """Save a list of windows to CMBA/HRM dataset layout.

    examples: list of (inputs[S], labels[S], puzzle_identifier)
    Groups correspond to puzzles (theorems). 'puzzle_identifiers' has length = num_puzzles.
    """
    # Group by theorem id
    by_pid: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for inp, lab, pid in examples:
        by_pid.setdefault(pid, []).append((inp, lab))

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    example_id = 0
    puzzle_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for pid in sorted(by_pid.keys()):
        # Add all windows for this theorem
        for inp, lab in by_pid[pid]:
            results["inputs"].append(inp)
            results["labels"].append(lab)
            example_id += 1

        # Close puzzle and record mapping
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(pid)
        puzzle_id += 1
        results["group_indices"].append(puzzle_id)

    if example_id == 0:
        raise RuntimeError("No examples produced for split '" + set_name + "'. Try lowering min_tactics.")

    def _stack(seq):
        return np.stack(seq, axis=0).astype(np.int32)

    np_results = {
        "inputs": _stack(results["inputs"]),
        "labels": _stack(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata: PAD=0, ignore_label_id=0, blank_identifier_id=0
    total_groups = int(np_results["group_indices"].size - 1)
    mean_puzzle_examples = float(np_results["inputs"].shape[0]) / float(max(total_groups, 1))
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=-1,  # Will be filled later by caller via a shared value per split set
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=num_puzzle_identifiers,
        total_groups=total_groups,
        mean_puzzle_examples=mean_puzzle_examples,
        sets=["all"],
    )

    save_dir = os.path.join(output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    # Dump metadata (temporary without vocab_size)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save arrays
    for k, v in np_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)


@cli.command(singleton=True)
def convert_leandojo(config: LeanDojoConvertConfig):
    """Convert a saved LeanDojo Hugging Face dataset to CMBA/HRM format.

    Task used: next-tactic prediction over windows of fixed length.
    - inputs[t]  = tactic_id at position t (PAD=0, UNK=1, known tactics start from 2)
    - labels[t]  = next tactic_id at position t (0 means ignore)
    """
    from datasets import load_from_disk

    ds = load_from_disk(config.input_dir)

    # Build tactic vocabulary from TRAIN ONLY to match model's vocab at train time
    train_vocab = _gather_train_vocab(ds["train"])  # type: ignore[index]
    # IDs: 0=PAD, 1=UNK, 2.. = known tactics
    tactic_to_id: Dict[str, int] = {name: (i + 2) for i, name in enumerate(sorted(train_vocab))}
    unk_id = 1
    vocab_size = 2 + len(tactic_to_id)  # PAD + UNK + known tactics

    # Build theorem-id mapping (puzzle identifiers) from TRAIN ONLY
    train_theorems: List[str] = [ex["full_name"] for ex in ds["train"]]  # type: ignore[index]
    theorem_to_pid: Dict[str, int] = {name: (idx + 1) for idx, name in enumerate(train_theorems)}  # 0 reserved for blank/unknown
    num_puzzle_identifiers = 1 + len(theorem_to_pid)

    # Save vocab for reference
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "tactic_vocab.json"), "w") as f:
        json.dump({"PAD": 0, "UNK": 1, **{k: v for k, v in tactic_to_id.items()}}, f, indent=2)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"] + train_theorems, f)

    def build_examples(hf_split, set_name: str) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        seq_len = int(config.seq_len)
        examples: List[Tuple[np.ndarray, np.ndarray, int]] = []
        for ex in hf_split:
            tactics_raw = ex.get("traced_tactics", []) or []
            tokens = _encode_tactics(tactics_raw, tactic_to_id, unk_id)
            if len(tokens) < config.min_tactics:
                continue
            # Determine puzzle identifier (0 for unknown theorems in non-train splits)
            pid = theorem_to_pid.get(str(ex.get("full_name", "")), 0)
            # Create non-overlapping windows
            for start in range(0, len(tokens), seq_len):
                window = tokens[start : start + seq_len]
                if len(window) == 0:
                    continue
                inp = np.zeros((seq_len,), dtype=np.int32)
                lab = np.zeros((seq_len,), dtype=np.int32)
                # inputs
                inp[: len(window)] = np.array(window, dtype=np.int32)
                # labels are next tokens (within the global sequence). For the last element of a window,
                # or if next token is out of window, mark as ignore (0)
                for i in range(len(window)):
                    global_idx = start + i + 1
                    if global_idx < len(tokens):
                        # Use next token id; if next token is unknown in non-train splits, still keep its id
                        lab[i] = int(tokens[global_idx]) if (global_idx - start) < seq_len else 0
                    else:
                        lab[i] = 0
                examples.append((inp, lab, pid))
        # Sort by pid for stable grouping
        examples.sort(key=lambda x: x[2])
        return examples

    # Build and save splits
    train_examples = build_examples(ds["train"], "train")  # type: ignore[index]
    _save_split(train_examples, config.seq_len, config.output_dir, "train", num_puzzle_identifiers)

    # Validation/Test → test split (merged if requested)
    if config.merge_val_into_test:
        from datasets import concatenate_datasets
        merged = concatenate_datasets([ds["validation"], ds["test"]])  # type: ignore[index]
        test_examples = build_examples(merged, "test")
    else:
        test_examples = build_examples(ds["test"], "test")  # type: ignore[index]
    _save_split(test_examples, config.seq_len, config.output_dir, "test", num_puzzle_identifiers)

    # After saving, update vocab_size in both splits' metadata to the shared train vocab size
    for split_name in ("train", "test"):
        meta_path = os.path.join(config.output_dir, split_name, "dataset.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["vocab_size"] = int(vocab_size)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    print(f"Converted LeanDojo to CMBA format at: {config.output_dir}\n"
          f"- seq_len={config.seq_len}, vocab_size={vocab_size}, num_puzzle_identifiers={num_puzzle_identifiers}\n"
          f"- train examples={len(train_examples)}, test examples={len(test_examples)}")


if __name__ == "__main__":
    cli()


