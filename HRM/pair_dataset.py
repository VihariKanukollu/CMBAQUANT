import json
from typing import Iterator, Tuple, Dict, Any

import torch
from torch.utils.data import IterableDataset

from models.losses import IGNORE_LABEL_ID


class PairDatasetConfig:
    def __init__(self, *, dataset_dir: str, tokenizer, global_batch_size: int, max_seq_len: int = 1024):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.global_batch_size = int(global_batch_size)
        self.max_seq_len = int(max_seq_len)


class PairDatasetMetadata:
    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = int(vocab_size)
        self.num_puzzle_identifiers = 1
        self.sets = ("train",)


class DistillPairsDataset(IterableDataset):
    """Stream pairs dataset saved with datasets.save_to_disk.

    Expects dict fields: 'prompt', 'chosen', 'rejected'.
    Each of 'chosen'/'rejected' is a messages list; 'prompt' is the user text.
    """

    def __init__(self, config: PairDatasetConfig):
        super().__init__()
        self.config = config

    def _encode_chat(self, messages) -> torch.Tensor:
        return self.config.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors='pt'
        )[0]

    def _encode_completion(self, text: str) -> torch.Tensor:
        out = self.config.tokenizer(text, add_special_tokens=False, return_tensors='pt')["input_ids"][0]
        eos = self.config.tokenizer.eos_token_id
        if eos is not None:
            out = torch.cat([out, torch.tensor([eos], dtype=out.dtype)])
        return out

    def _pad(self, x: torch.Tensor, L: int, pad_id: int) -> torch.Tensor:
        if x.numel() >= L:
            return x[:L]
        return torch.cat([x, torch.full((L - x.numel(),), pad_id, dtype=torch.long)])

    def _batch_iter(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor], int]]:
        import datasets
        ds = datasets.load_from_disk(self.config.dataset_dir)
        ds = ds["train"]

        pad_id = self.config.tokenizer.pad_token_id or 0
        L = self.config.max_seq_len

        batch_inputs = []
        batch_labels = []

        for rec in ds:
            try:
                chosen_msgs = rec.get("chosen")
                rejected_msgs = rec.get("rejected")
                if not isinstance(chosen_msgs, list) or not isinstance(rejected_msgs, list):
                    continue

                # prompt from chosen up to assistant
                # chosen/rejected both include [user, assistant]; build prompt from the first turn only
                # Use messages[:-1] as prompt and messages[-1] as completion if it's assistant
                def build_io(msgs):
                    if len(msgs) < 2 or msgs[-1].get('role') != 'assistant':
                        return None
                    prompt = msgs[:-1]
                    completion = msgs[-1]['content']
                    inp = self._encode_chat(prompt)
                    tgt = self._encode_completion(completion)
                    ids = torch.cat([inp, tgt], dim=0)
                    labels = ids.clone().to(torch.long)
                    labels[:-1] = ids[1:]
                    labels[-1] = IGNORE_LABEL_ID
                    if inp.numel() > 0:
                        labels[: max(int(inp.numel()) - 1, 0)] = IGNORE_LABEL_ID
                    return ids, labels

                pos = build_io(chosen_msgs)
                neg = build_io(rejected_msgs)
                if pos is None or neg is None:
                    continue

                for ids, labels in (pos, neg):
                    batch_inputs.append(self._pad(ids, L, pad_id))
                    batch_labels.append(self._pad(labels, L, IGNORE_LABEL_ID))

                if len(batch_inputs) == 2 * self.config.global_batch_size:
                    inputs = torch.stack(batch_inputs, dim=0)
                    labels = torch.stack(batch_labels, dim=0)
                    yield "train", {
                        "inputs": inputs.to(torch.int32),
                        "labels": labels.to(torch.int32),
                        "puzzle_identifiers": torch.zeros((inputs.shape[0],), dtype=torch.int32),
                    }, inputs.shape[0]
                    batch_inputs, batch_labels = [], []
            except Exception:
                continue

    def __iter__(self):
        yield from self._batch_iter()


