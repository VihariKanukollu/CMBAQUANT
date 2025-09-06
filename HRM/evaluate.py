from typing import List
import yaml
import os

import torch
import torch.distributed as dist

import pydantic
from omegaconf import OmegaConf
from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader


class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    
    save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"]


def launch():
    # Parse CLI once for EvalConfig, and again for raw overrides
    cli_all = OmegaConf.to_container(OmegaConf.from_cli())  # type: ignore
    eval_cfg = EvalConfig(**{k: v for k, v in (cli_all or {}).items() if k in {"checkpoint", "save_outputs"}})  # type: ignore
    
    RANK = 0
    WORLD_SIZE = 1
    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Robust config loader: handle OmegaConf/Hydra YAMLs with custom tags
    cfg_dir = os.path.dirname(eval_cfg.checkpoint)
    cfg_path = os.path.join(cfg_dir, "all_config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found next to checkpoint: {cfg_path}")

    def _strip_omegaconf_artifacts(obj):
        if isinstance(obj, dict):
            # If this looks like an OmegaConf container, unwrap its content
            if '_content' in obj and all(k.startswith('_') or k == '_content' for k in obj.keys()):
                return _strip_omegaconf_artifacts(obj['_content'])
            clean = {}
            for k, v in obj.items():
                if isinstance(k, str) and k.startswith('_'):
                    continue
                clean[k] = _strip_omegaconf_artifacts(v)
            return clean
        if isinstance(obj, list):
            return [_strip_omegaconf_artifacts(v) for v in obj]
        return obj

    def _sanitize_omegaconf_yaml_text(txt: str) -> str:
        import re
        lines = txt.splitlines()
        out = []
        skip_stack = []
        def current_skip_indent():
            return skip_stack[-1] if skip_stack else None
        for line in lines:
            # Remove explicit python object tags inline
            line = line.replace('!!python/object:omegaconf.dictconfig.DictConfig', '')
            line = line.replace('!!python/object:omegaconf.listconfig.ListConfig', '')
            line = line.replace('tag:yaml.org,2002:python/object:omegaconf.dictconfig.DictConfig', '')
            line = line.replace('tag:yaml.org,2002:python/object:omegaconf.listconfig.ListConfig', '')
            # Strip anchors and replace aliases with null
            line = re.sub(r'&id\w+', '', line)
            line = re.sub(r':\s*\*id\w+', ': null', line)

            # Determine indentation
            indent = len(line) - len(line.lstrip(' '))

            # Close skip blocks on dedent
            while skip_stack and indent <= current_skip_indent():
                skip_stack.pop()

            # Start skip blocks for known OmegaConf internals
            if re.match(r'^\s*_(metadata|resolvers|parent)\s*:', line):
                skip_stack.append(indent)
                continue
            if re.match(r'^\s*resolver_cache\s*:', line):
                skip_stack.append(indent)
                continue

            if skip_stack:
                continue
            out.append(line)
        return '\n'.join(out)

    def _load_config_dict(yaml_path: str):
        try:
            with open(yaml_path, "r") as f:
                return yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            pass
        except Exception:
            # Try continue with OmegaConf
            pass
        try:
            oc = OmegaConf.load(yaml_path)
            return OmegaConf.to_container(oc, resolve=True)
        except Exception:
            # Fallback: custom YAML loader that ignores python/object tags from OmegaConf dumps,
            # then strip OmegaConf artifacts like _content/_metadata recursively.
            with open(yaml_path, "r") as f:
                txt = _sanitize_omegaconf_yaml_text(f.read())

            class _OmegaIgnoreLoader(yaml.SafeLoader):
                pass

            def _ignore_pyobj(loader, tag_suffix, node):
                if isinstance(node, yaml.MappingNode):
                    return loader.construct_mapping(node, deep=True)
                if isinstance(node, yaml.SequenceNode):
                    return loader.construct_sequence(node, deep=True)
                return loader.construct_scalar(node)

            yaml.add_multi_constructor('tag:yaml.org,2002:python/object:', _ignore_pyobj, Loader=_OmegaIgnoreLoader)
            yaml.add_multi_constructor('tag:yaml.org,2002:python/object/apply:', _ignore_pyobj, Loader=_OmegaIgnoreLoader)
            yaml.add_multi_constructor('tag:yaml.org,2002:python/object/new:', _ignore_pyobj, Loader=_OmegaIgnoreLoader)
            yaml.add_multi_constructor('tag:yaml.org,2002:python/name:', _ignore_pyobj, Loader=_OmegaIgnoreLoader)
            yaml.add_constructor('tag:yaml.org,2002:python/tuple', lambda loader, node: loader.construct_sequence(node, deep=True), Loader=_OmegaIgnoreLoader)

            data = yaml.load(txt, Loader=_OmegaIgnoreLoader)
            if not isinstance(data, (dict, list)):
                raise RuntimeError("Unsupported YAML root type")
            return _strip_omegaconf_artifacts(data)

    loaded_cfg = _load_config_dict(cfg_path)
    if not isinstance(loaded_cfg, dict):
        raise RuntimeError("Failed to parse checkpoint config into a dictionary")

    # Repair MCP fields if OmegaConf artifacts left them malformed
    try:
        arch_cfg = loaded_cfg.get('arch', {}) if isinstance(loaded_cfg.get('arch', {}), dict) else {}
        # Default keys and costs
        default_mcp_keys = [
            'puzzle', 'halt', 'gate', 'headbias', 'routing', 'film', 'rope', 'sched', 'ponder', 'ntm',
            'h_cycles', 'l_cycles', 'mlp_expand', 'heads_active', 'min_steps', 'max_steps'
        ]
        default_mcp_costs = {'ntm': 1e-4, 'routing': 5e-5, 'film': 5e-5}

        # Keys
        keys_val = arch_cfg.get('mcp_feature_keys', None)
        if not isinstance(keys_val, list) or not all(isinstance(x, (str, int, float)) for x in keys_val):
            arch_cfg['mcp_feature_keys'] = default_mcp_keys

        # Costs
        costs_val = arch_cfg.get('mcp_feature_costs', None)
        fixed_costs = {}
        if isinstance(costs_val, dict):
            for k in ('ntm', 'routing', 'film'):
                v = costs_val.get(k, default_mcp_costs[k])
                try:
                    fixed_costs[k] = float(v)
                except Exception:
                    fixed_costs[k] = default_mcp_costs[k]
        else:
            fixed_costs = default_mcp_costs
        arch_cfg['mcp_feature_costs'] = fixed_costs

        loaded_cfg['arch'] = arch_cfg
    except Exception:
        # Leave as-is if any unexpected structure; PretrainConfig may still coerce
        pass

    config = PretrainConfig(**loaded_cfg)  # type: ignore[arg-type]

    # Apply arbitrary CLI overrides to config (arch.* and top-level PretrainConfig keys)
    def _set_in_dict(d: dict, dotted_key: str, value):
        # remove optional leading + for Hydra-style additions
        if dotted_key.startswith('+'):
            dotted_key = dotted_key[1:]
        keys = dotted_key.split('.')
        cur = d
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = value

    # Build a mutable dict from current config and deep-merge CLI overrides
    base = config.model_dump()

    def _flatten_overrides(d: dict, prefix: str = ""):
        for k, v in (d or {}).items():
            if k in {"checkpoint", "save_outputs"}:
                continue
            dotted = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                yield from _flatten_overrides(v, dotted)
            else:
                yield dotted, v

    for dotted_key, value in _flatten_overrides(cli_all or {}):
        try:
            _set_in_dict(base, dotted_key, value)
        except Exception:
            pass
    # Re-instantiate config with overrides applied
    config = PretrainConfig(**base)  # type: ignore[arg-type]
    config.eval_save_outputs = eval_cfg.save_outputs
    config.checkpoint_path = cfg_dir

    # Dataloader
    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    # Models
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Robust checkpoint loading to handle compiled/uncompiled and DDP prefixes
    def _adapt_state_dict_keys(sd: dict, model: torch.nn.Module) -> dict:
        keys = list(sd.keys())
        if not len(keys):
            return sd
        # Strip common prefixes first
        def _strip_prefix(k: str, p: str) -> str:
            return k[len(p):] if k.startswith(p) else k
        new_sd = {}
        for k, v in sd.items():
            nk = _strip_prefix(k, 'module.')
            new_sd[nk] = v
        sd = new_sd
        # Align _orig_mod prefix based on target model expectation
        target_keys = list(model.state_dict().keys())
        expects_orig = any(k.startswith('_orig_mod.') for k in target_keys)
        has_orig = any(k.startswith('_orig_mod.') for k in sd.keys())
        if expects_orig and not has_orig:
            sd = {f"_orig_mod.{k}": v for k, v in sd.items()}
        elif not expects_orig and has_orig:
            sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
        return sd

    raw_sd = torch.load(eval_cfg.checkpoint, map_location="cuda")
    if not isinstance(raw_sd, dict):
        raise RuntimeError("Checkpoint is not a state_dict dictionary")
    adapted = _adapt_state_dict_keys(raw_sd, train_state.model)
    # Load permissively to tolerate architectural drift
    missing_unexpected = train_state.model.load_state_dict(adapted, strict=False)
    try:
        # If wrapped by torch.compile, also try to load into the original module when available
        if hasattr(train_state.model, '_orig_mod'):
            train_state.model._orig_mod.load_state_dict(_adapt_state_dict_keys(raw_sd, train_state.model._orig_mod), strict=False)  # type: ignore[attr-defined]
    except Exception:
        pass
    
    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("step_"):
        train_state.step = int(ckpt_filename.removeprefix("step_"))

    # Evaluate
    print ("Starting evaluation")
    
    train_state.model.eval()
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

    if metrics is not None:
        print (metrics)


if __name__ == "__main__":
    launch()
