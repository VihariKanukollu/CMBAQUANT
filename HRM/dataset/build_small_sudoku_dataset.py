from typing import List, Tuple
import os
import csv
import json
import math
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata  # type: ignore


cli = ArgParser()


class SmallSudokuConfig(BaseModel):
    # Grid configuration
    n: int  # 4 or 6
    block_rows: int  # 2 for both 4x4 and 6x6
    block_cols: int  # 2 for 4x4, 3 for 6x6

    # Input and output
    input_file: str  # CSV or JSONL with (puzzle, solution)
    output_dir: str = "data/sudoku-small"

    # Sampling and splitting
    num_puzzles: int = 1000
    holdout: float = 0.1  # fraction for test split
    split_seed: int = 20250901
 

def _char_to_int(c: str) -> int:
    """Map a single character to an integer value.

    Supports digits '0'-'9'. For safety, also maps letters 'A'-'Z' to 10..35
    though this should not be needed for 4x4/6x6.
    """
    if c == ".":
        return 0
    if "0" <= c <= "9":
        return ord(c) - ord("0")
    uc = c.upper()
    if "A" <= uc <= "Z":
        return 10 + (ord(uc) - ord("A"))
    return 0


def _string_to_board(s: str, n: int) -> np.ndarray:
    s = s.strip().replace(" ", "")
    s = s.replace(".", "0")
    # Some CSVs might wrap values in quotes
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1]
    if len(s) != n * n:
        raise ValueError(f"Unexpected puzzle/solution length {len(s)} != {n*n}")
    vals = np.frombuffer(bytes([_char_to_int(ch) for ch in s]), dtype=np.uint8)
    return vals.reshape(n, n)


def _read_pairs_from_csv(path: str, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        maybe_header = next(reader)
        # Detect header heuristically
        header_like = any("puzzle" in x.lower() or "solution" in x.lower() for x in maybe_header)
        if not header_like:
            # treat as data and process first row
            row = maybe_header
            if len(row) == 1:
                row = row[0].split(",")
            if len(row) >= 2:
                p, a = row[0], row[1]
                pairs.append((_string_to_board(p, n), _string_to_board(a, n)))
        # Continue with the rest
        for row in reader:
            if len(row) == 0:
                continue
            if len(row) == 1:
                row = row[0].split(",")
            if len(row) < 2:
                continue
            p, a = row[0], row[1]
            try:
                pairs.append((_string_to_board(p, n), _string_to_board(a, n)))
            except Exception:
                # Skip bad rows silently to keep conversion robust
                continue
    return pairs


def _read_pairs_from_jsonl(path: str, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Read JSONL where each line has keys for puzzle and solution.
    Tries common key names: (puzzle, solution), (problem, solution), (grid, grid_solution).
    """
    candidates = [
        ("puzzle", "solution"),
        ("problem", "solution"),
        ("grid", "grid_solution"),
        ("puzzle_str", "solution_str"),
    ]
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            found = False
            for kp, ks in candidates:
                if kp in obj and ks in obj:
                    try:
                        pairs.append((_string_to_board(str(obj[kp]), n), _string_to_board(str(obj[ks]), n)))
                        found = True
                        break
                    except Exception:
                        pass
            if not found:
                continue
    return pairs


def _load_pairs(input_file: str, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if input_file.lower().endswith(".csv"):
        return _read_pairs_from_csv(input_file, n)
    if input_file.lower().endswith(".jsonl") or input_file.lower().endswith(".json"):
        return _read_pairs_from_jsonl(input_file, n)
    raise ValueError("input_file must be .csv or .jsonl/.json containing puzzle and solution pairs")


def _save_split(pairs: List[Tuple[np.ndarray, np.ndarray]], n: int, output_dir: str, set_name: str) -> None:
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for orig_inp, orig_out in tqdm(pairs):
        inp, out = orig_inp, orig_out
        results["inputs"].append(inp)
        results["labels"].append(out)
        example_id += 1
        puzzle_id += 1
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)
        # mark end of this puzzle's group
        results["group_indices"].append(puzzle_id)

    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        assert np.all((arr >= 0) & (arr <= n))
        return arr + 1  # shift by 1 so PAD=0

    np_results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=n * n,
        vocab_size=(n + 1) + 1,  # digits 0..n plus PAD
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(np_results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"],
    )

    save_dir = os.path.join(output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    for k, v in np_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def convert_small_sudoku(config: SmallSudokuConfig):
    """Convert an external 4x4/6x6 Sudoku dataset to the project's format.

    - Reads (puzzle, solution) pairs from CSV or JSONL.
    - Randomly subsamples to num_puzzles.
    - Splits into train/test with the provided holdout fraction and seed.
    - Saves in the same .npy/metadata layout as the 9x9 converter.
    """
    assert config.n in (4, 6), "Only 4x4 and 6x6 are supported by this converter"
    assert config.block_rows * config.block_cols == config.n, "Block shape must tile the grid (block_rows*block_cols == n)"

    pairs = _load_pairs(config.input_file, config.n)
    if len(pairs) == 0:
        raise RuntimeError("No valid puzzle/solution pairs parsed from input_file")

    rng = np.random.default_rng(config.split_seed)
    total = min(config.num_puzzles, len(pairs))
    indices = rng.choice(len(pairs), size=total, replace=False)
    rng.shuffle(indices)

    test_count = max(1, int(math.floor(total * config.holdout)))
    test_idx = set(indices[:test_count])
    train_idx = [i for i in indices[test_count:]]
    test_idx_list = [i for i in indices[:test_count]]

    train_pairs = [pairs[i] for i in train_idx]
    test_pairs = [pairs[i] for i in test_idx_list]

    os.makedirs(config.output_dir, exist_ok=True)
    _save_split(train_pairs, config.n, config.output_dir, "train")
    _save_split(test_pairs, config.n, config.output_dir, "test")


if __name__ == "__main__":
    cli()


