from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []

    # Evaluation controls
    # Limit number of eval batches (across all sets). None = evaluate full test set
    eval_max_batches: Optional[int] = None
    # Cap inner ACT thinking steps during eval (<= arch.halt_max_steps). None = use full max
    eval_halt_max_steps: Optional[int] = None
    # Random sampling for eval subset
    eval_sample_random: bool = False
    eval_sample_seed: int = 20250901


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,

        dataset_path=config.data_path,

        rank=rank,
        num_replicas=world_size,
        
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,

        num_workers=1,
        prefetch_factor=8,

        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore

        batch_size=config.global_batch_size // world_size,

        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

        # Ensure base model tensors live on CUDA, then move PQC TorchLayer weights to CPU
        model = model.cuda()

        # Collect PQC TorchLayer params to keep on CPU and optimize separately
        pqc_params: List[torch.nn.Parameter] = []
        try:
            # TorchLayer class is imported by quantum model file; detect by class name to avoid hard import
            for module in model.modules():
                if module.__class__.__name__ == "TorchLayer":
                    for p in module.parameters(recurse=False):
                        p.data = p.data.cpu()
                        if p.grad is not None:
                            p.grad.data = p.grad.data.cpu()
                        pqc_params.append(p)
        except Exception:
            pqc_params = []

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    optimizers = []
    optimizer_lrs = []

    # Use special optimizer only for table-based sparse puzzle embeddings (3 buffers expected)
    use_sparse_emb_opt = False
    if hasattr(model.model, "puzzle_emb"):
        try:
            use_sparse_emb_opt = (len(list(model.model.puzzle_emb.buffers())) == 3)  # type: ignore[attr-defined]
        except Exception:
            use_sparse_emb_opt = False

    if use_sparse_emb_opt:
        optimizers.append(
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore[attr-defined]
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        )
        optimizer_lrs.append(config.puzzle_emb_lr)

    # Main optimizer for CUDA parameters only (exclude CPU PQC weights)
    cuda_params = [p for p in model.parameters() if p.device.type == "cuda"]
    if len(cuda_params):
        optimizers.append(
            AdamATan2(
                cuda_params,
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        )
        optimizer_lrs.append(config.lr)

    # Separate CPU optimizer for PQC TorchLayer weights (if any)
    if 'pqc_params' in locals() and len(pqc_params):
        optimizers.append(
            torch.optim.Adam(pqc_params, lr=0, weight_decay=0.0)
        )
        optimizer_lrs.append(config.lr)

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(config.checkpoint_path, f"step_{train_state.step}")
    torch.save(train_state.model.state_dict(), checkpoint_file)

    # Also upload checkpoint to Weights & Biases for remote availability
    if wandb.run is not None:
        try:
            artifact = wandb.Artifact(name=f"checkpoint-step-{train_state.step}", type="model")
            artifact.add_file(checkpoint_file)
            wandb.log_artifact(artifact)
        except Exception:
            # Do not fail training if artifact upload has any transient issue
            pass


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    with torch.inference_mode():
        # Optionally cap halting steps during evaluation to speed it up
        orig_halt_max = None
        try:
            if getattr(config, 'eval_halt_max_steps', None) is not None:
                orig_halt_max = train_state.model.config.halt_max_steps  # type: ignore[attr-defined]
                train_state.model.config.halt_max_steps = min(orig_halt_max, int(config.eval_halt_max_steps))  # type: ignore[attr-defined]
        except Exception:
            pass

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        
        all_preds = {}

        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]
        
        # Evaluation progress indicator (rank 0 only)
        eval_pbar = None
        if rank == 0:
            try:
                ds = getattr(eval_loader, 'dataset', None)
                if ds is not None and hasattr(ds, '_lazy_load_dataset'):
                    ds._lazy_load_dataset()
                    total_batches = 0
                    for set_name in ds.metadata.sets:  # type: ignore[attr-defined]
                        total_examples = len(ds._data[set_name]["inputs"])  # type: ignore[attr-defined]
                        total_batches += math.ceil(total_examples / config.global_batch_size)
                    if getattr(config, 'eval_max_batches', None) is not None:
                        total_batches = min(total_batches, int(config.eval_max_batches))
                    eval_pbar = tqdm.tqdm(total=total_batches, leave=False, desc="Eval")
            except Exception:
                # Fallback: show an indeterminate bar by omitting total
                try:
                    eval_pbar = tqdm.tqdm(leave=False, desc="Eval")
                except Exception:
                    eval_pbar = None

        carry = None
        seen_batches = 0

        # Optional random batch sampling: if enabled, we precompute a target set of batch indices
        # and skip batches not in the sample. This keeps streaming behavior and low memory.
        sample_batch_indices = None
        if getattr(config, 'eval_sample_random', False) and getattr(config, 'eval_max_batches', None) is not None:
            # Estimate total number of batches using dataset sizes
            try:
                ds = getattr(eval_loader, 'dataset', None)
                if ds is not None and hasattr(ds, '_lazy_load_dataset'):
                    ds._lazy_load_dataset()
                    total_batches_est = 0
                    for _sn in ds.metadata.sets:  # type: ignore[attr-defined]
                        total_examples = len(ds._data[_sn]["inputs"])  # type: ignore[attr-defined]
                        total_batches_est += math.ceil(total_examples / config.global_batch_size)
                    rng = torch.Generator()
                    rng.manual_seed(int(getattr(config, 'eval_sample_seed', 20250901)))
                    k = min(int(config.eval_max_batches), total_batches_est)
                    # Draw unique indices without replacement
                    sample_batch_indices = set(torch.randperm(total_batches_est, generator=rng)[:k].tolist())
            except Exception:
                sample_batch_indices = None

        batch_index = 0
        for set_name, batch, global_batch_size in eval_loader:
            if sample_batch_indices is not None and (batch_index not in sample_batch_indices):
                batch_index += 1
                continue
            if getattr(config, 'eval_max_batches', None) is not None and seen_batches >= int(config.eval_max_batches):
                break
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            while True:
                carry, _, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=config.eval_save_outputs)
                
                if all_finish:
                    break

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory
                        
            del carry, preds, batch, all_finish

            # Update eval progress bar
            if eval_pbar is not None:
                try:
                    eval_pbar.update(1)
                except Exception:
                    pass
            seen_batches += 1
            batch_index += 1

            # Aggregate
            set_id = set_ids[set_name]
            
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")
                
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))

        # Logging
        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)
            
            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {set_name: {metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
                                   for set_id, set_name in enumerate(set_ids)}
                
                # Postprocess
                for set_name, metrics in reduced_metrics.items():
                    count = metrics.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}

                if eval_pbar is not None:
                    try:
                        eval_pbar.close()
                    except Exception:
                        pass
                # Restore halting steps after evaluation
                if orig_halt_max is not None:
                    try:
                        train_state.model.config.halt_max_steps = orig_halt_max  # type: ignore[attr-defined]
                    except Exception:
                        pass
                return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_path).capitalize()} ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    # Train state
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)

        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

        ############ Evaluation
        train_state.model.eval()
        metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

        if RANK == 0 and metrics is not None:
            wandb.log(metrics, step=train_state.step)
            
        ############ Checkpointing
        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state)

    # finalize
    if RANK == 0 and progress_bar is not None:
        # Ensure progress bar reaches 100% even if last partial batch was dropped
        progress_bar.update(train_state.total_steps - progress_bar.n)
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
