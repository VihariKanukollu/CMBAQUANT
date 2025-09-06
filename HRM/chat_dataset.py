import json
"""Chat dataset for supervised finetuning with next-token prediction on assistant turns.

This replaces any prior chat dataset file. It yields fixed-length, padded batches that
match the training loop interface: (set_name, batch_dict, global_batch_size).
"""

import json
from typing import Iterator, Tuple, Dict, Any, Optional, List

import torch
from torch.utils.data import IterableDataset
import random

from models.losses import IGNORE_LABEL_ID


class ChatDatasetConfig:
    """Configuration for ChatDataset.

    Args:
        jsonl_path: Path to JSONL with conversations (OpenAI-style messages list).
        tokenizer: Hugging Face tokenizer with a chat template.
        global_batch_size: Number of examples per yielded batch.
        max_seq_len: Final length after padding/truncation.
        supervise_roles: Which roles' tokens should be supervised; default assistant-only.
        pad_token_id: Override pad id; defaults to tokenizer.pad_token_id.
        drop_incomplete_last_batch: Drop trailing partial batch for stability.
    """

    def __init__(self, *,
                 jsonl_path: str,
                 tokenizer,
                 global_batch_size: int,
                 max_seq_len: int = 2048,
                 supervise_roles: Tuple[str, ...] = ("assistant",),
                 pad_token_id: Optional[int] = None,
                 drop_incomplete_last_batch: bool = True,
                 # Augmentations (use defaults/zeros for eval)
                 sample_any_assistant_turn: bool = False,
                 samples_per_conversation: int = 1,
                 random_truncate_prob: float = 0.0,
                 token_dropout_prob: float = 0.0,
                 system_prompt_variant_prob: float = 0.0):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.global_batch_size = int(global_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.supervise_roles = tuple(supervise_roles)
        self.pad_token_id = tokenizer.pad_token_id if pad_token_id is None else int(pad_token_id)
        self.drop_incomplete_last_batch = bool(drop_incomplete_last_batch)
        # Augmentations
        self.sample_any_assistant_turn = bool(sample_any_assistant_turn)
        self.samples_per_conversation = max(1, int(samples_per_conversation))
        self.random_truncate_prob = float(random_truncate_prob)
        self.token_dropout_prob = float(token_dropout_prob)
        self.system_prompt_variant_prob = float(system_prompt_variant_prob)


class ChatDataset(IterableDataset):
    """Stream a chat JSONL file and build batched tensors for training/eval."""

    def __init__(self, config: ChatDatasetConfig):
        super().__init__()
        self.config = config

    def _normalize_messages(self, messages: Any):
        # Keep expected roles and string content; ensure alternation starting from user.
        cleaned = []
        system = None
        for m in messages:
            role = m.get('role')
            content = m.get('content') or m.get('value') or ""
            if not content:
                continue
            if role == 'system' and system is None:
                system = {'role': 'system', 'content': content}
            elif role in ('user', 'assistant'):
                cleaned.append({'role': role, 'content': content})

        alternated = []
        expect = 'user'
        for m in cleaned:
            if m['role'] == expect:
                alternated.append(m)
                expect = 'assistant' if expect == 'user' else 'user'
        if len(alternated) and alternated[-1]['role'] == 'user':
            alternated.pop()

        result = []
        if system is not None:
            result.append(system)
        result.extend(alternated)
        return result

    def _encode_example(self, messages) -> Tuple[torch.Tensor, torch.Tensor]:
        messages = self._normalize_messages(messages)
        if not messages:
            raise ValueError('empty conversation after normalization')

        # Choose assistant turn: last or any
        ass_indices: List[int] = [i for i, m in enumerate(messages) if m.get('role') == 'assistant']
        if not len(ass_indices):
            raise ValueError('no assistant turn')
        if self.config.sample_any_assistant_turn:
            ass_idx = random.choice(ass_indices)
        else:
            ass_idx = ass_indices[-1]

        # Optionally vary/insert system prompt
        if (self.config.system_prompt_variant_prob > 0.0) and (random.random() < self.config.system_prompt_variant_prob):
            variants = [
                "You are a helpful assistant.",
                "You are a concise and accurate assistant.",
                "You are an expert assistant.",
                "Be helpful, safe, and honest.",
            ]
            variant = random.choice(variants)
            if len(messages) and messages[0].get('role') == 'system':
                messages[0] = {'role': 'system', 'content': variant}
            else:
                messages = [{'role': 'system', 'content': variant}] + messages
                ass_indices = [i + 1 for i in ass_indices]
                ass_idx += 1

        # Randomly drop some leading (user, assistant) pairs before target
        # After normalization, dialogue alternates and starts with user (after optional system)
        assistants_before = len([i for i in ass_indices if i < ass_idx])
        drop_pairs = 0
        if (self.config.random_truncate_prob > 0.0) and (assistants_before > 1) and (random.random() < self.config.random_truncate_prob):
            drop_pairs = random.randint(0, assistants_before - 1)

        if len(messages) and messages[0].get('role') == 'system':
            start_idx = 1 + 2 * drop_pairs
        else:
            start_idx = 2 * drop_pairs
        start_idx = min(start_idx, max(0, ass_idx - 1))
        prompt_messages = messages[:ass_idx]
        if start_idx > 0:
            if len(messages) and messages[0].get('role') == 'system':
                prompt_messages = [messages[0]] + messages[start_idx:ass_idx]
            else:
                prompt_messages = messages[start_idx:ass_idx]
        assistant_text = messages[ass_idx]['content']

        prompt_ids = self.config.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors='pt'
        )[0]

        target = self.config.tokenizer(
            assistant_text,
            add_special_tokens=False,
            return_tensors='pt'
        )["input_ids"][0]
        eos_id = self.config.tokenizer.eos_token_id
        if eos_id is not None:
            target = torch.cat([target, torch.tensor([eos_id], dtype=target.dtype)])

        # Optional token dropout on prompt tokens only
        if self.config.token_dropout_prob > 0.0 and prompt_ids.numel() > 0:
            p = self.config.token_dropout_prob
            mask = (torch.rand_like(prompt_ids.to(torch.float32)) < p)
            if mask.numel() > 0:
                mask[0] = False  # keep first special token intact
            prompt_ids = torch.where(mask, torch.tensor(self.config.pad_token_id, dtype=prompt_ids.dtype), prompt_ids)

        input_ids = torch.cat([prompt_ids, target], dim=0)

        labels = input_ids.clone().to(torch.long)
        labels[:-1] = input_ids[1:]
        labels[-1] = IGNORE_LABEL_ID
        if prompt_ids.numel() > 0:
            labels[: max(int(prompt_ids.numel()) - 1, 0)] = IGNORE_LABEL_ID

        input_ids = input_ids[: self.config.max_seq_len]
        labels = labels[: self.config.max_seq_len]
        return input_ids, labels

    def _batch_iter(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor], int]]:
        batch_inputs: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []

        with open(self.config.jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                messages = obj.get('messages') or obj.get('conversations') or obj.get('data')
                if messages is None:
                    continue
                # Emit multiple augmented samples per conversation if requested
                samples = max(1, int(self.config.samples_per_conversation))
                for _ in range(samples):
                    try:
                        input_ids, labels = self._encode_example(messages)
                    except Exception:
                        continue

                    batch_inputs.append(input_ids)
                    batch_labels.append(labels)

                    if len(batch_inputs) == self.config.global_batch_size:
                        L = int(self.config.max_seq_len)
                        pad_id = int(self.config.pad_token_id)
                        inps, labs = [], []
                        for inp, lab in zip(batch_inputs, batch_labels):
                            if inp.numel() > L:
                                inp = inp[:L]
                            if lab.numel() > L:
                                lab = lab[:L]
                            if inp.numel() < L:
                                inp = torch.cat([inp, torch.full((L - inp.numel(),), pad_id, dtype=torch.long)])
                            if lab.numel() < L:
                                lab = torch.cat([lab, torch.full((L - lab.numel(),), IGNORE_LABEL_ID, dtype=torch.long)])
                            inps.append(inp)
                            labs.append(lab)
                        batch_inputs_fixed = torch.stack(inps, dim=0)
                        batch_labels_fixed = torch.stack(labs, dim=0)

                        yield "train", {
                            "inputs": batch_inputs_fixed.to(torch.int32),
                            "labels": batch_labels_fixed.to(torch.int32),
                            "puzzle_identifiers": torch.zeros((batch_inputs_fixed.shape[0],), dtype=torch.int32),
                        }, batch_inputs_fixed.shape[0]

                        batch_inputs, batch_labels = [], []

        # Optionally drop incomplete last batch; keeping it complicates distributed sync
        if (not self.config.drop_incomplete_last_batch) and len(batch_inputs) > 0:
            L = int(self.config.max_seq_len)
            pad_id = int(self.config.pad_token_id)
            inps, labs = [], []
            for inp, lab in zip(batch_inputs, batch_labels):
                if inp.numel() > L:
                    inp = inp[:L]
                if lab.numel() > L:
                    lab = lab[:L]
                if inp.numel() < L:
                    inp = torch.cat([inp, torch.full((L - inp.numel(),), pad_id, dtype=torch.long)])
                if lab.numel() < L:
                    lab = torch.cat([lab, torch.full((L - lab.numel(),), IGNORE_LABEL_ID, dtype=torch.long)])
                inps.append(inp)
                labs.append(lab)
            batch_inputs_fixed = torch.stack(inps, dim=0)
            batch_labels_fixed = torch.stack(labs, dim=0)

            yield "train", {
                "inputs": batch_inputs_fixed.to(torch.int32),
                "labels": batch_labels_fixed.to(torch.int32),
                "puzzle_identifiers": torch.zeros((batch_inputs_fixed.shape[0],), dtype=torch.int32),
            }, batch_inputs_fixed.shape[0]

    def __iter__(self):
        yield from self._batch_iter()

