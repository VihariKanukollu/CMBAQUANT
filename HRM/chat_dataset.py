import json
from typing import Iterator, Tuple, Dict, Any, Optional

import torch
from torch.utils.data import IterableDataset

from models.losses import IGNORE_LABEL_ID


class ChatDatasetConfig:
    def __init__(self, *,
                 jsonl_path: str,
                 tokenizer,
                 global_batch_size: int,
                 max_seq_len: int = 2048,
                 supervise_roles: Tuple[str, ...] = ("assistant",),
                 pad_token_id: Optional[int] = None):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.global_batch_size = int(global_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.supervise_roles = tuple(supervise_roles)
        self.pad_token_id = tokenizer.pad_token_id if pad_token_id is None else int(pad_token_id)


class ChatDataset(IterableDataset):
    def __init__(self, config: ChatDatasetConfig):
        super().__init__()
        self.config = config

    def _encode_example(self, messages) -> Tuple[torch.Tensor, torch.Tensor]:
        # Build full chat prompt with template
        input_ids = self.config.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors='pt'
        )[0]

        # Shifted labels: predict next token; supervise only assistant tokens
        labels = input_ids.clone()
        # Mask non-supervised spans
        # Simple heuristic: re-render messages role-by-role and mask tokens outside assistant turns
        role_mask = torch.zeros_like(labels, dtype=torch.bool)
        cur = 0
        for m in messages:
            chunk = self.config.tokenizer.apply_chat_template([m], add_generation_prompt=False, return_tensors='pt')[0]
            length = chunk.numel()
            if m.get('role') in self.config.supervise_roles:
                role_mask[cur:cur+length] = True
            cur += length
        labels[~role_mask] = IGNORE_LABEL_ID

        # Truncate to max_seq_len
        input_ids = input_ids[: self.config.max_seq_len]
        labels = labels[: self.config.max_seq_len]

        return input_ids, labels

    def _batch_iter(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor], int]]:
        batch_inputs = []
        batch_labels = []

        with open(self.config.jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                messages = obj.get('messages') or obj.get('conversations') or obj.get('data')
                if messages is None:
                    continue
                input_ids, labels = self._encode_example(messages)

                batch_inputs.append(input_ids)
                batch_labels.append(labels)

                if len(batch_inputs) == self.config.global_batch_size:
                    batch_inputs = torch.nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True, padding_value=self.config.pad_token_id)
                    batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=IGNORE_LABEL_ID)

                    yield "train", {
                        "inputs": batch_inputs.to(torch.int32),
                        "labels": batch_labels.to(torch.int32),
                        "puzzle_identifiers": torch.zeros((batch_inputs.shape[0],), dtype=torch.int32),
                    }, batch_inputs.shape[0]

                    batch_inputs, batch_labels = [], []

        # Drop incomplete last batch for simplicity

    def __iter__(self):
        yield from self._batch_iter()


