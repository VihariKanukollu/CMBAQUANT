from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
import torch._dynamo as dynamo
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, YarnRotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


class PennyLanePQC(nn.Module):
    """GPU periodic MLP.

    Maps input_dim -> n_wires (via a learned projection), applies sinusoidal features
    and a small MLP for `n_layers`, then projects to output_dim. Stays on GPU.
    """
    def __init__(self, input_dim: int, output_dim: int, n_wires: int, n_layers: int):
        super().__init__()
        assert input_dim > 0 and output_dim > 0 and n_wires > 0 and n_layers > 0
        self.n_wires = n_wires
        self.n_layers = n_layers

        self.input_proj = CastedLinear(input_dim, n_wires, bias=True)
        hidden_dim = 2 * n_wires  # sin and cos features
        self.hidden = nn.ModuleList([
            CastedLinear(hidden_dim, hidden_dim, bias=True) for _ in range(max(0, n_layers - 1))
        ])
        self.readout = CastedLinear(hidden_dim, output_dim, bias=True)
        self.act = nn.SiLU()
        # Per-dimension frequency scale; cast at runtime to match input dtype
        self.freq = nn.Parameter(torch.ones(n_wires))

    def _fourier(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, n_wires]
        f = self.freq.to(h.dtype).view(1, -1)
        h = h * f
        return torch.cat([torch.sin(h), torch.cos(h)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        z = self._fourier(h)
        for layer in self.hidden:
            z = self.act(layer(z))
        return self.readout(z)

    def zero_last_linear(self, bias_fill: Optional[float] = None):
        with torch.no_grad():
            self.readout.weight.zero_()
            if (bias_fill is not None) and (self.readout.bias is not None):
                self.readout.bias.fill_(bias_fill)  # type: ignore[arg-type]


class PQCPuzzleEmbedding(nn.Module):
    """PQC-based embedding for puzzle identifiers with optional eval-time caching.

    Builds lightweight features from id, projects to wires, runs PQC, then to emb_dim.
    """
    def __init__(self, num_ids: int, emb_dim: int, cast_to: torch.dtype,
                 n_wires: int, n_layers: int, cache_eval: bool = True, cache_size: int = 4096):
        super().__init__()
        self.num_ids = max(1, int(num_ids))
        self.emb_dim = emb_dim
        self.cast_to = cast_to
        self.cache_eval = cache_eval
        self.cache_size = cache_size
        self._cache: Dict[int, torch.Tensor] = {}

        # Simple id features (normalized id + harmonics), then PQC
        self.feat_dim = 1 + 2 * 3
        self.feat_proj = CastedLinear(self.feat_dim, n_wires, bias=True)
        self.pqc = PennyLanePQC(n_wires, emb_dim, n_wires=n_wires, n_layers=n_layers)

    def _features(self, ids: torch.Tensor) -> torch.Tensor:
        ids_f = ids.to(torch.float32) / float(self.num_ids)
        two_pi = 6.283185307179586
        f1 = two_pi * ids_f
        f2 = two_pi * 2.0 * ids_f
        f3 = two_pi * 4.0 * ids_f
        feats = torch.stack([
            ids_f,
            torch.sin(f1), torch.cos(f1),
            torch.sin(f2), torch.cos(f2),
            torch.sin(f3), torch.cos(f3),
        ], dim=-1)
        return feats

    def forward(self, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        ids = puzzle_identifiers.view(-1).to(torch.long)
        if (not self.training) and self.cache_eval:
            out_list: List[torch.Tensor] = []
            for i in ids.tolist():
                if i in self._cache:
                    out_list.append(self._cache[i])
                else:
                    feats = self._features(torch.tensor([i], device=ids.device, dtype=torch.long))
                    x = self.feat_proj(feats.to(self.cast_to))
                    emb = self.pqc(x).to(self.cast_to)
                    if len(self._cache) >= self.cache_size:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[i] = emb.squeeze(0).detach().to(self.cast_to)
                    out_list.append(self._cache[i])
            out = torch.stack(out_list, dim=0)
        else:
            feats = self._features(ids)
            x = self.feat_proj(feats.to(self.cast_to))
            out = self.pqc(x).to(self.cast_to)
        return out.view(puzzle_identifiers.shape[0], self.emb_dim)


class QuantumGatingHead(nn.Module):
    """Produces small gating vector (e.g., 2 scalars for attn/mlp) via PQC (no fallback)."""
    def __init__(self, input_dim: int, gate_dim: int, n_wires: int, n_layers: int):
        super().__init__()
        self.gate_dim = gate_dim
        self.core = PennyLanePQC(input_dim, gate_dim, n_wires=n_wires, n_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


class SharedPQCTrunk(nn.Module):
    """Single PQC trunk shared by multiple quantum heads.

    Adapters are small linear maps on top of the trunk output.
    """
    def __init__(self, input_dim: int, trunk_dim: int, n_wires: int, n_layers: int):
        super().__init__()
        # trunk_dim is the output of trunk before small adapters
        self.trunk = PennyLanePQC(input_dim=input_dim, output_dim=trunk_dim, n_wires=n_wires, n_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class SharedPQCAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.map = CastedLinear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.map(x)


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor
    # Optional NTM scratchpad state
    ntm_memory: Optional[torch.Tensor] = None  # [B, N, M]
    ntm_read_w: Optional[torch.Tensor] = None  # [B, N]
    ntm_write_w: Optional[torch.Tensor] = None  # [B, N]


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    # Ponder-style state: hazard history per step [B, halt_max_steps]
    hazard_history: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    # Causal attention for autoregressive text (True in chat/text mode)
    causal: bool = False
    # If <=0, defaults to num_heads (no GQA). If < num_heads, enables GQA.
    num_kv_heads: int = 0
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    # YARN long-context options (used when pos_encodings=="rope_yarn")
    rope_original_seq_len: int = 0  # if <=0, defaults to seq_len
    rope_factor: float = 1.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_mscale_base: float = 1.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float
    # New: enforce a minimum number of thinking steps during training
    halt_min_steps: int = 0
    # New: small penalty added to loss when the policy attempts to halt before the floor
    act_min_step_penalty: float = 0.0

    forward_dtype: str = "bfloat16"

    # Halting head options
    halt_head_type: str = "linear"  # "linear" | "mlp" | "pqc"
    halt_proj_dim: int = 0  # if >0, project z_H[:,0] before halt head
    halt_head_hidden: int = 128

    # PQC common params
    pqc_n_wires: int = 8
    pqc_n_layers: int = 2

    # Puzzle embedding options
    puzzle_emb_type: str = "table"  # "table" | "pqc"
    puzzle_emb_cache_eval: bool = True
    puzzle_emb_cache_size: int = 4096

    # Quantum gating options
    quantum_gate_enabled: bool = False
    quantum_gate_dim: int = 2  # gates for [attn, mlp]
    quantum_gate_last_h_block_only: bool = False
    quantum_gate_proj_dim: int = 0

    # ACT scheduler PQC
    act_sched_enabled: bool = False
    act_sched_proj_dim: int = 32  # if <=0, uses concat hidden sizes directly
    act_sched_bias_scale: float = 1.0

    # Per-head attention bias/gating (last H block)
    per_head_bias_enabled: bool = False
    per_head_bias_scale: float = 1.0

    # Token routing mask (last H block)
    token_routing_enabled: bool = False
    token_routing_keep_ratio: float = 1.0  # 0..1, applied via sigmoid thresholding

    # Shared PQC trunk and adapters (within a block)
    pqc_shared: bool = False

    # FiLM conditioning (last H block)
    film_enabled: bool = False
    film_groups: int = 32
    film_scale: float = 1.0

    # RoPE phase bias (last H block)
    rope_phase_bias_enabled: bool = False
    rope_phase_bias_per_head: bool = True
    rope_phase_bias_scale: float = 1.0

    # MCP controller
    mcp_enabled: bool = False
    mcp_backend: str = "mlp"  # "mlp" | "pqc"
    mcp_temp: float = 1.0
    mcp_hard_eval: bool = True
    mcp_cost_coef: float = 0.0
    mcp_entropy_coef: float = 0.0
    # MCP central feature controller
    mcp_feature_keys: List[str] = [
        "puzzle", "halt", "gate", "headbias", "routing", "film", "rope", "sched", "ponder", "ntm",
        "h_cycles", "l_cycles", "mlp_expand", "heads_active", "min_steps", "max_steps"
    ]
    # Optional per-feature costs for cost-aware regularization
    mcp_feature_costs: Dict[str, float] = {}
    # Auto features enables short-circuiting heavy paths in eval
    mcp_auto_features: bool = True
    mcp_eval_threshold: float = 0.5
    # Optional high-level profile to seed MCP biases
    mcp_feature_profile: str = "balanced"  # "fast" | "balanced" | "algorithmic"
    # Dynamic ceilings for MCP-controlled compute
    max_h_cycles: int = 0  # if <=0, use H_cycles
    max_l_cycles: int = 0  # if <=0, use L_cycles
    max_expansion: float = 0.0  # if <=0, use expansion
    halt_min_steps_ceiling: int = 0  # if <=0, use halt_min_steps

    # NTM scratchpad (external memory) options
    ntm_enabled: bool = False
    ntm_rows: int = 128  # N
    ntm_dim: int = 128   # M
    ntm_num_read_heads: int = 1
    ntm_num_write_heads: int = 1
    ntm_rw_order: str = "write_then_read"  # or "read_then_write"
    ntm_proj_dim: int = 0  # if >0, project controller summary before heads
    ntm_inject_mode: str = "add"  # "add" | "concat"
    ntm_gate_from_mcp: bool = True
    ntm_entropy_reg: float = 0.0
    ntm_erase_reg: float = 0.0

    # Ponder-style halting options (optional, default off)
    ponder_enabled: bool = False
    ponder_lambda_p: float = 0.2
    ponder_epsilon: float = 0.05
    ponder_kl_div_loss_weight: float = 0.0
    # Deterministic eval early-exit by cumulative halt prob threshold
    ponder_eval_deterministic: bool = True
    ponder_eval_threshold: float = 0.5
    # If true, during eval halt when current step is the argmax of halt distribution so far
    ponder_eval_best_step: bool = False


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=(config.num_kv_heads if getattr(config, 'num_kv_heads', 0) and config.num_kv_heads > 0 else config.num_heads),
            causal=bool(getattr(config, 'causal', False))
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

        # Optional quantum gating head
        self.quantum_gate = None
        self.quantum_gate_last_h_block_only = config.quantum_gate_last_h_block_only
        self.gate_proj = None
        _mcp_keys = set(getattr(config, 'mcp_feature_keys', []))
        _mcp_en = bool(getattr(config, 'mcp_enabled', False))
        if (_mcp_en and ('gate' in _mcp_keys)) and config.quantum_gate_dim > 0:
            gate_in_dim = config.hidden_size if config.quantum_gate_proj_dim <= 0 else config.quantum_gate_proj_dim
            if config.quantum_gate_proj_dim > 0:
                self.gate_proj = CastedLinear(config.hidden_size, config.quantum_gate_proj_dim, bias=False)
            self.quantum_gate = QuantumGatingHead(
                input_dim=gate_in_dim,
                gate_dim=config.quantum_gate_dim,
                n_wires=config.pqc_n_wires,
                n_layers=config.pqc_n_layers,
            )

        # Optional per-head bias and token routing (PQC on summary)
        self.per_head_bias = None
        self.token_router = None
        # Wire config scalars directly so getattr() in forward reads correct values
        self.per_head_bias_scale = getattr(config, "per_head_bias_scale", 1.0)
        self.token_routing_keep_ratio = getattr(config, "token_routing_keep_ratio", 1.0)
        if _mcp_en and ('headbias' in _mcp_keys):
            self.per_head_bias = QuantumGatingHead(
                input_dim=(config.quantum_gate_proj_dim if config.quantum_gate_proj_dim > 0 else config.hidden_size),
                gate_dim=config.num_heads,
                n_wires=config.pqc_n_wires,
                n_layers=config.pqc_n_layers,
            )
        if _mcp_en and ('routing' in _mcp_keys):
            self.token_router = QuantumGatingHead(
                input_dim=(config.quantum_gate_proj_dim if config.quantum_gate_proj_dim > 0 else config.hidden_size),
                gate_dim=1,  # produce a keep score per token based on similarity later
                n_wires=config.pqc_n_wires,
                n_layers=config.pqc_n_layers,
            )

        # Shared PQC trunk and adapters
        self.shared_trunk = None
        self.adapter_gate = None
        self.adapter_headbias = None
        self.adapter_router = None
        self.adapter_film = None
        self.adapter_rope = None
        trunk_dim = 64
        if getattr(config, "pqc_shared", False):
            in_dim = (config.quantum_gate_proj_dim if config.quantum_gate_proj_dim > 0 else config.hidden_size)
            self.shared_trunk = SharedPQCTrunk(input_dim=in_dim, trunk_dim=trunk_dim, n_wires=config.pqc_n_wires, n_layers=config.pqc_n_layers)
            # Build adapters on demand
            if self.quantum_gate is not None:
                self.adapter_gate = SharedPQCAdapter(trunk_dim, config.quantum_gate_dim)
            if getattr(config, "per_head_bias_enabled", False):
                self.adapter_headbias = SharedPQCAdapter(trunk_dim, config.num_heads)
            if getattr(config, "token_routing_enabled", False):
                self.adapter_router = SharedPQCAdapter(trunk_dim, 1)
            if getattr(config, "film_enabled", False):
                self.adapter_film = SharedPQCAdapter(trunk_dim, 2 * max(1, int(config.film_groups)))
            if getattr(config, "rope_phase_bias_enabled", False):
                rope_dim = (config.num_heads if config.rope_phase_bias_per_head else 1)
                self.adapter_rope = SharedPQCAdapter(trunk_dim, rope_dim)

        # FiLM conditioning head (gamma,beta per group)
        self.film_head = None
        if (_mcp_en and ('film' in _mcp_keys)) and not getattr(config, "pqc_shared", False):
            groups = max(1, int(config.film_groups))
            self.film_groups = groups
            self.film_scale = getattr(config, "film_scale", 1.0)
            self.film_head = QuantumGatingHead(
                input_dim=(config.quantum_gate_proj_dim if config.quantum_gate_proj_dim > 0 else config.hidden_size),
                gate_dim=2 * groups,
                n_wires=config.pqc_n_wires,
                n_layers=config.pqc_n_layers,
            )

        # RoPE phase bias head
        self.rope_phase_head = None
        if (_mcp_en and ('rope' in _mcp_keys)) and not getattr(config, "pqc_shared", False):
            gate_dim = (config.num_heads if config.rope_phase_bias_per_head else 1)
            self.rope_phase_per_head = config.rope_phase_bias_per_head
            self.rope_phase_scale = getattr(config, "rope_phase_bias_scale", 1.0)
            self.rope_phase_head = QuantumGatingHead(
                input_dim=(config.quantum_gate_proj_dim if config.quantum_gate_proj_dim > 0 else config.hidden_size),
                gate_dim=gate_dim,
                n_wires=config.pqc_n_wires,
                n_layers=config.pqc_n_layers,
            )
        
        # Unified feature gate wrapper
        # compute_fn receives gate tensor or None and must return a value or None
        # If short_circuit and gate==0 (eval), returns None without calling compute_fn
        
        def _use_feature(mcp, key: str, compute_fn, short_circuit: bool = True):
            g = None if (mcp is None) else mcp.get(key, None)
            if g is None:
                return compute_fn(None)
            if (not self.training) and short_circuit and getattr(self.config, 'mcp_auto_features', True) and getattr(self.config, 'mcp_hard_eval', True):
                try:
                    if (g == 0).all().item():
                        return None
                except Exception:
                    pass
            return compute_fn(g)
        
        self._use_feature = _use_feature
    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, *, block_index: int = 0, num_blocks: int = 1, module_role: str = "", mcp_gates: Optional[Dict[str, torch.Tensor]] = None, rope_mscale: Optional[float] = None) -> torch.Tensor:
        # Post Norm
        # Compute optional gating from summary token
        gate_vals = None
        apply_gate = (self.quantum_gate is not None)
        if apply_gate and self.quantum_gate_last_h_block_only:
            apply_gate = (module_role == "H" and (block_index == (num_blocks - 1)))

        # Per-head bias and token routing only for last H block
        is_last_H = (module_role == "H" and (block_index == (num_blocks - 1)))
        apply_head_bias = ((self.per_head_bias is not None) or (self.adapter_headbias is not None)) and is_last_H
        apply_token_routing = ((self.token_router is not None) or (self.adapter_router is not None)) and is_last_H
        # MCP gates
        g_gate = g_head = g_route = g_film = g_rope = None
        if (mcp_gates is not None) and is_last_H:
            g_gate  = mcp_gates.get('gate',    None)
            g_head  = mcp_gates.get('headbias',None)
            g_route = mcp_gates.get('routing', None)
            g_film  = mcp_gates.get('film',    None)
            g_rope  = mcp_gates.get('rope',    None)

        trunk_cache = None
        def _trunk(summary):
            nonlocal trunk_cache
            if trunk_cache is None:
                trunk_cache = self.shared_trunk(summary)
            return trunk_cache

        if apply_gate:
            summary = hidden_states[:, 0]
            if self.gate_proj is not None:
                summary = self.gate_proj(summary)
            if self.shared_trunk is not None and self.adapter_gate is not None:
                gate_vals = torch.sigmoid(self.adapter_gate(_trunk(summary)))
            else:
                gate_vals = torch.sigmoid(self.quantum_gate(summary))  # [B, gate_dim]
            if (g_gate is not None) and (gate_vals is not None):
                gate_vals = gate_vals * g_gate.view(-1, 1).to(gate_vals.dtype)

        per_head_scale = None
        if apply_head_bias:
            def _compute_headbias(g):
                summary = hidden_states[:, 0]
                if self.gate_proj is not None:
                    summary = self.gate_proj(summary)
                raw = self.adapter_headbias(_trunk(summary)) if (self.shared_trunk is not None and self.adapter_headbias is not None) else self.per_head_bias(summary)
                head_bias = torch.tanh(raw) * getattr(self, 'per_head_bias_scale', 1.0)
                if g is not None:
                    head_bias = head_bias * g.view(-1, 1).to(head_bias.dtype)
                return (1.0 + head_bias)
            res = self._use_feature(mcp_gates, 'headbias', _compute_headbias)
            if res is not None:
                per_head_scale = res

        # Heads active gating (by simple prefix mask)
        if (mcp_gates is not None) and ('heads_active' in mcp_gates):
            H = getattr(self.self_attn, 'num_heads', None)
            if isinstance(H, int) and H > 0:
                g_heads = mcp_gates['heads_active'].to(hidden_states.dtype).view(-1)
                k = torch.clamp((g_heads * H).round().to(torch.int64), min=1, max=H)
                # build mask [B,H]: ones for first k heads, zeros for the rest
                ar = torch.arange(H, device=hidden_states.device).view(1, -1)
                mask = (ar < k.view(-1, 1)).to(hidden_states.dtype)
                if per_head_scale is None:
                    per_head_scale = mask
                else:
                    per_head_scale = per_head_scale * mask

        if apply_token_routing:
            def _compute_routing(g):
                summary = hidden_states[:, 0]
                tokens = hidden_states
                if self.gate_proj is not None:
                    summary = self.gate_proj(summary)
                    tokens = self.gate_proj(hidden_states)
                sim = torch.einsum('bd,btd->bt', summary, tokens)
                sim = sim / (hidden_states.shape[-1] ** 0.5)
                if self.shared_trunk is not None and self.adapter_router is not None:
                    router_bias = self.adapter_router(_trunk(summary)).squeeze(-1)
                    if g is not None:
                        sim = sim + (g.view(-1,1) * router_bias.unsqueeze(-1))
                    else:
                        sim = sim + router_bias.unsqueeze(-1)
                keep = torch.sigmoid(sim)
                keep_ratio = getattr(self, 'token_routing_keep_ratio', 1.0)
                if keep_ratio < 1.0:
                    keep_f = keep.to(torch.float32)
                    thresh = torch.quantile(keep_f, q=(1.0 - keep_ratio), dim=1, keepdim=True)
                    keep = (keep >= thresh).to(hidden_states.dtype)
                keep_mask = keep.view(keep.shape[0], keep.shape[1], 1)
                return hidden_states * keep_mask
            res = self._use_feature(mcp_gates, 'routing', _compute_routing)
            if res is not None:
                hidden_states = res

        # Self Attention with optional gate on residual
        # Optional RoPE phase bias
        rope_phase = None
        if (module_role == "H" and (block_index == (num_blocks - 1))):
            def _compute_rope(g):
                if self.rope_phase_head is not None:
                    summary = hidden_states[:, 0]
                    if self.gate_proj is not None:
                        summary = self.gate_proj(summary)
                    raw = self.rope_phase_head(summary)
                    if not self.rope_phase_per_head:
                        raw = raw.expand(-1, self.self_attn.num_heads)
                    rp = raw * getattr(self, 'rope_phase_scale', 1.0)
                elif self.shared_trunk is not None and self.adapter_rope is not None:
                    summary = hidden_states[:, 0]
                    if self.gate_proj is not None:
                        summary = self.gate_proj(summary)
                    raw = self.adapter_rope(_trunk(summary))
                    raw = raw if raw.shape[-1] == self.self_attn.num_heads else raw.expand(-1, self.self_attn.num_heads)
                    rp = raw * getattr(self, 'rope_phase_scale', 1.0)
                else:
                    return None
                if g is not None:
                    rp = rp * g.view(-1,1).to(rp.dtype)
                return rp
            res = self._use_feature(mcp_gates, 'rope', _compute_rope)
            if res is not None:
                rope_phase = res

        attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states, per_head_scale=per_head_scale, per_head_phase=rope_phase, rope_mscale=rope_mscale)
        if gate_vals is not None and gate_vals.shape[-1] >= 1:
            gate_attn = gate_vals[..., 0].view(-1, 1, 1)
            hidden_states = rms_norm(hidden_states + gate_attn * attn_out, variance_epsilon=self.norm_eps)
        else:
            hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)

        # Fully Connected with optional gate on residual
        # Dynamic MLP expansion scaling via MCP
        mlp_out = self.mlp(hidden_states)
        # Optionally scale MLP residual by MCP expansion gate
        # Effective expansion ~ 1..max_expansion; we approximate by scaling residual
        try:
            max_exp = float(getattr(self.config, 'max_expansion', 0.0)) or float(getattr(self.config, 'expansion', 4.0))
            if mcp_gates is not None and ('mlp_expand' in mcp_gates) and (max_exp > 1.0):
                scale = 1.0 + (max_exp - 1.0) * mcp_gates['mlp_expand'].view(-1, 1, 1).to(mlp_out.dtype)
                mlp_out = mlp_out * scale
        except Exception:
            pass
        if gate_vals is not None and gate_vals.shape[-1] >= 2:
            gate_mlp = gate_vals[..., 1].view(-1, 1, 1)
            hidden_states = rms_norm(hidden_states + gate_mlp * mlp_out, variance_epsilon=self.norm_eps)
        else:
            hidden_states = rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)

        # FiLM conditioning on last H block
        if (module_role == "H" and (block_index == (num_blocks - 1))):
            def _compute_film(g):
                film = None
                if self.film_head is not None:
                    summary = hidden_states[:, 0]
                    if self.gate_proj is not None:
                        summary = self.gate_proj(summary)
                    film = self.film_head(summary)
                elif self.shared_trunk is not None and self.adapter_film is not None:
                    summary = hidden_states[:, 0]
                    if self.gate_proj is not None:
                        summary = self.gate_proj(summary)
                    film = self.adapter_film(_trunk(summary))
                if film is None:
                    return None
                G_cfg = max(1, int(getattr(self, 'film_groups', 32)))
                Hdim = hidden_states.shape[-1]
                G = G_cfg
                if Hdim % G != 0:
                    import math as _math
                    G = _math.gcd(Hdim, G_cfg) or 1
                self.film_groups = G
                self.film_scale = getattr(self, 'film_scale', 1.0)
                gamma, beta = film[..., :G], film[..., G:]
                gamma = 1.0 + torch.tanh(gamma) * self.film_scale
                beta  =        torch.tanh(beta)  * self.film_scale
                if g is not None:
                    sf = g.view(-1, 1).to(hidden_states.dtype)
                    gamma = 1.0 + sf * (gamma - 1.0)
                    beta  =        sf * beta
                gamma = gamma.view(-1, 1, G)
                beta  = beta.view(-1, 1, G)
                group_size = Hdim // G
                if group_size > 0:
                    hs = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], G, group_size)
                    hs = hs * gamma.unsqueeze(-1) + beta.unsqueeze(-1)
                    return hs.view(hidden_states.shape[0], hidden_states.shape[1], Hdim)
                return hidden_states
            res = self._use_feature(mcp_gates, 'film', _compute_film)
            if res is not None:
                hidden_states = res
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block], role: str):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)
        self.role = role  # "H" or "L"

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        num_blocks = len(self.layers)
        mcp_gates = kwargs.get('mcp_gates', None)
        for idx, layer in enumerate(self.layers):
            # Per-layer gating via MCP (keys: h_layer_<idx>, l_layer_<idx>)
            layer_key = f"{'h' if self.role=='H' else 'l'}_layer_{idx}"
            if mcp_gates is not None and (layer_key in mcp_gates):
                if self.role and getattr(layer, '_should_skip', None) is None:
                    pass
                # short-circuit if gate==0 in eval
                g = mcp_gates[layer_key]
                if (not getattr(self, 'training', False)) and (g.ndim == 1) and bool((g == 0).all().item()):
                    continue
            hidden_states = layer(hidden_states=hidden_states, block_index=idx, num_blocks=num_blocks, module_role=self.role, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # MCP controller (gates for features)
        self.mcp_head = None
        if getattr(self.config, "mcp_enabled", False):
            mcp_in = self.config.hidden_size
            # Dynamic feature keys
            self.mcp_feature_keys = list(getattr(self.config, "mcp_feature_keys", [
                "puzzle", "halt", "gate", "headbias", "routing", "film", "rope", "sched", "ponder", "ntm"
            ]))
            mcp_out = max(1, int(len(self.mcp_feature_keys)))
            if self.config.mcp_backend == "pqc":
                self.mcp_head = PennyLanePQC(mcp_in, mcp_out, n_wires=self.config.pqc_n_wires, n_layers=self.config.pqc_n_layers)
            else:
                self.mcp_head = nn.Sequential(
                    CastedLinear(mcp_in, 128, bias=True), nn.SiLU(),
                    CastedLinear(128, mcp_out, bias=True),
                )
            # Seed MCP last-layer bias by feature_profile (MLP backend only)
            if isinstance(self.mcp_head, nn.Sequential) and isinstance(self.mcp_head[-1], CastedLinear):
                with torch.no_grad():
                    bias = self.mcp_head[-1].bias
                    if bias is not None:
                        if getattr(self.config, 'mcp_feature_profile', 'balanced') == 'fast':
                            # discourage heavy features
                            for i, k in enumerate(self.mcp_feature_keys):
                                if k in ("ntm", "routing", "film", "ponder"):
                                    bias[i] = bias[i] - 1.0
                        elif getattr(self.config, 'mcp_feature_profile', 'balanced') == 'algorithmic':
                            # encourage memory/routing/ponder
                            for i, k in enumerate(self.mcp_feature_keys):
                                if k in ("ntm", "routing", "sched", "ponder"):
                                    bias[i] = bias[i] + 1.0
        # Halting head: optional projection and linear/MLP/PQC
        self.halt_proj = None
        q_in_dim = self.config.hidden_size
        if getattr(self.config, "halt_proj_dim", 0) and self.config.halt_proj_dim > 0:
            self.halt_proj = CastedLinear(self.config.hidden_size, self.config.halt_proj_dim, bias=False)
            q_in_dim = self.config.halt_proj_dim
        halt_type = getattr(self.config, "halt_head_type", "linear")
        if halt_type == "mlp":
            self.q_head = nn.Sequential(
                CastedLinear(q_in_dim, self.config.halt_head_hidden, bias=True),
                nn.SiLU(),
                CastedLinear(self.config.halt_head_hidden, 2, bias=True),
            )
        elif halt_type == "pqc":
            self.q_head = PennyLanePQC(
                input_dim=q_in_dim,
                output_dim=2,
                n_wires=self.config.pqc_n_wires,
                n_layers=self.config.pqc_n_layers,
            )
        else:
            self.q_head = CastedLinear(q_in_dim, 2, bias=True)

        # MCP halting delta head (registered in init)
        self.halt_delta = None
        if getattr(self.config, "mcp_enabled", False):
            if getattr(self.config, 'mcp_backend', 'mlp') == 'pqc':
                self.halt_delta = PennyLanePQC(q_in_dim, 2, n_wires=self.config.pqc_n_wires, n_layers=self.config.pqc_n_layers)
            else:
                self.halt_delta = CastedLinear(q_in_dim, 2, bias=True)

        # Optional quantum kernel novelty feature and critic (advantage baseline)
        self.qkernel_feat = None
        self.qcritic = None
        if getattr(self.config, 'halt_head_type', 'linear') == 'pqc':
            # Produce 1-d novelty feature via PQC
            self.qkernel_feat = PennyLanePQC(
                input_dim=q_in_dim,
                output_dim=1,
                n_wires=self.config.pqc_n_wires,
                n_layers=self.config.pqc_n_layers,
            )
            # Critic baseline for halting logits
            self.qcritic = PennyLanePQC(
                input_dim=q_in_dim,
                output_dim=1,
                n_wires=self.config.pqc_n_wires,
                n_layers=self.config.pqc_n_layers,
            )

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            if getattr(self.config, "mcp_enabled", False):
                # Build both paths for blending
                self.puzzle_emb_table = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                        batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)
                self.puzzle_emb_pqc = PQCPuzzleEmbedding(
                    num_ids=self.config.num_puzzle_identifiers,
                    emb_dim=self.config.puzzle_emb_ndim,
                    cast_to=self.forward_dtype,
                    n_wires=self.config.pqc_n_wires,
                    n_layers=self.config.pqc_n_layers,
                    cache_eval=self.config.puzzle_emb_cache_eval,
                    cache_size=self.config.puzzle_emb_cache_size,
                )
            else:
                if getattr(self.config, "puzzle_emb_type", "table") == "pqc":
                    self.puzzle_emb = PQCPuzzleEmbedding(
                        num_ids=self.config.num_puzzle_identifiers,
                        emb_dim=self.config.puzzle_emb_ndim,
                        cast_to=self.forward_dtype,
                        n_wires=self.config.pqc_n_wires,
                        n_layers=self.config.pqc_n_layers,
                        cache_eval=self.config.puzzle_emb_cache_eval,
                        cache_size=self.config.puzzle_emb_cache_size,
                    )
                else:
                    # Zero init puzzle embeddings (table)
                    self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                            batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
            self.rope_mscale = 1.0
        elif self.config.pos_encodings == "rope_yarn":
            orig = (self.config.rope_original_seq_len if getattr(self.config, 'rope_original_seq_len', 0) and self.config.rope_original_seq_len > 0 else self.config.seq_len + self.puzzle_emb_len)
            self.rotary_emb = YarnRotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                                  max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                                  base=self.config.rope_theta,
                                                  original_seq_len=orig,
                                                  rope_factor=max(1.0, float(getattr(self.config, 'rope_factor', 1.0))),
                                                  beta_fast=int(getattr(self.config, 'rope_beta_fast', 32)),
                                                  beta_slow=int(getattr(self.config, 'rope_beta_slow', 1)),
                                                  mscale_base=float(getattr(self.config, 'rope_mscale_base', 1.0)))
            # Fetch mscale from embedding for attention rescale
            self.rope_mscale = float(getattr(self.rotary_emb, 'mscale', 1.0))
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)], role="H")
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)], role="L")

        # ACT scheduler (PQC) producing [gate_h2l, gate_l2h, halt_bias]
        self.sched_proj = None
        if self.config.act_sched_enabled:
            sched_in_dim = self.config.hidden_size * 2
            if self.config.act_sched_proj_dim and self.config.act_sched_proj_dim > 0:
                self.sched_proj = CastedLinear(sched_in_dim, self.config.act_sched_proj_dim, bias=False)
                sched_in_dim = self.config.act_sched_proj_dim
            self.act_scheduler = PennyLanePQC(
                input_dim=sched_in_dim,
                output_dim=3,
                n_wires=self.config.pqc_n_wires,
                n_layers=self.config.pqc_n_layers,
            )
        else:
            self.act_scheduler = None
        
        # Initial states [1,1,H] so they broadcast to [B,S,H]
        H0 = trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1)
        L0 = trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1)
        self.register_buffer("H_init", H0.view(1, 1, -1), persistent=True)
        self.register_buffer("L_init", L0.view(1, 1, -1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            if isinstance(self.q_head, CastedLinear):
                self.q_head.weight.zero_()
                if self.q_head.bias is not None:
                    self.q_head.bias.fill_(-5)  # type: ignore
            elif isinstance(self.q_head, nn.Sequential):
                last = self.q_head[-1]
                if isinstance(last, CastedLinear):
                    last.weight.zero_()
                    if last.bias is not None:
                        last.bias.fill_(-5)  # type: ignore
            elif isinstance(self.q_head, PennyLanePQC):
                self.q_head.zero_last_linear(bias_fill=-5.0)

        # NTM scratchpad setup
        self.ntm_enabled = bool(getattr(self.config, "ntm_enabled", False))
        if self.ntm_enabled:
            N = int(getattr(self.config, "ntm_rows", 128))
            M = int(getattr(self.config, "ntm_dim", 128))
            self.ntm_N = N
            self.ntm_M = M
            # Memory bias like NTM
            stdev = 1.0 / (float(N + M) ** 0.5)
            mem_bias = torch.empty(N, M, dtype=self.forward_dtype)
            nn.init.uniform_(mem_bias, -stdev, stdev)
            self.register_buffer("ntm_mem_bias", mem_bias, persistent=True)

            # Controller projection for head params
            controller_in = self.config.hidden_size
            if getattr(self.config, "ntm_proj_dim", 0) and self.config.ntm_proj_dim > 0:
                self.ntm_proj = CastedLinear(self.config.hidden_size, self.config.ntm_proj_dim, bias=False)
                controller_in = self.config.ntm_proj_dim
            else:
                self.ntm_proj = None

            # Read head params: k(M), beta(1), g(1), s(3), gamma(1) => M + 6
            self.ntm_read_fc = CastedLinear(controller_in, M + 6, bias=True)
            # Write head params: k(M), beta(1), g(1), s(3), gamma(1), e(M), a(M) => 3M + 6
            self.ntm_write_fc = CastedLinear(controller_in, 3 * M + 6, bias=True)
            # Read projection back to hidden size (for injection)
            self.ntm_read_to_hidden = CastedLinear(M, self.config.hidden_size, bias=False)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor, g_puzzle: Optional[torch.Tensor] = None):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            if getattr(self.config, "mcp_enabled", False):
                # Blend table and PQC via gate
                assert self.puzzle_emb_table is not None and self.puzzle_emb_pqc is not None
                table = self.puzzle_emb_table(puzzle_identifiers)
                pqc = self.puzzle_emb_pqc(puzzle_identifiers)
                if g_puzzle is None:
                    g = torch.ones((table.shape[0], 1), dtype=table.dtype, device=table.device)
                else:
                    g = g_puzzle.view(-1, 1).to(table.dtype)
                puzzle_embedding = g * pqc + (1 - g) * table
            else:
                puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            pos_table = self.embed_pos.weight.to(self.forward_dtype)  # [S_total, H]
            embedding = 0.707106781 * (embedding + pos_table.unsqueeze(0))  # [B,S,H] + [1,S,H]

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        dev = next(self.parameters()).device
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=dev),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=dev),
            ntm_memory=(self.ntm_mem_bias.unsqueeze(0).repeat(batch_size, 1, 1).to(dev) if getattr(self, "ntm_mem_bias", None) is not None else None),
            ntm_read_w=(torch.zeros(batch_size, getattr(self, "ntm_N", 1), dtype=self.forward_dtype, device=dev) if getattr(self, "ntm_enabled", False) else None),
            ntm_write_w=(torch.zeros(batch_size, getattr(self, "ntm_N", 1), dtype=self.forward_dtype, device=dev) if getattr(self, "ntm_enabled", False) else None),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        # Reset transformers states
        zH = torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H)
        zL = torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L)

        if self.ntm_enabled:
            # Reset memory and head weights on new episodes
            if carry.ntm_memory is None:
                ntm_mem = self.ntm_mem_bias.unsqueeze(0).repeat(zH.shape[0], 1, 1)
            else:
                ntm_mem = torch.where(reset_flag.view(-1, 1, 1), self.ntm_mem_bias, carry.ntm_memory)
            if carry.ntm_read_w is None:
                ntm_rw = torch.zeros(zH.shape[0], self.ntm_N, dtype=self.forward_dtype, device=zH.device)
            else:
                ntm_rw = torch.where(reset_flag.view(-1, 1), torch.zeros_like(carry.ntm_read_w), carry.ntm_read_w)
            if carry.ntm_write_w is None:
                ntm_ww = torch.zeros(zH.shape[0], self.ntm_N, dtype=self.forward_dtype, device=zH.device)
            else:
                ntm_ww = torch.where(reset_flag.view(-1, 1), torch.zeros_like(carry.ntm_write_w), carry.ntm_write_w)
        else:
            ntm_mem = None
            ntm_rw = None
            ntm_ww = None

        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=zH,
            z_L=zL,
            ntm_memory=ntm_mem,
            ntm_read_w=ntm_rw,
            ntm_write_w=ntm_ww,
        )

    @staticmethod
    def _cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-16) -> torch.Tensor:
        a_norm = a / (a.norm(dim=dim, keepdim=True) + eps)
        b_norm = b / (b.norm(dim=dim, keepdim=True) + eps)
        return (a_norm * b_norm).sum(dim=dim)

    def _ntm_address(self, memory: torch.Tensor, k: torch.Tensor, beta: torch.Tensor, g: torch.Tensor, s_logits: torch.Tensor, gamma: torch.Tensor, w_prev: torch.Tensor) -> torch.Tensor:
        # Activations
        beta = F.softplus(beta)
        g = torch.sigmoid(g).view(-1, 1)  # [B,1] for broadcasting over N
        s = F.softmax(s_logits, dim=-1)
        gamma = (1.0 + F.softplus(gamma)).view(-1, 1)  # [B,1] for broadcasting over N

        # Content focus (robust broadcasting)
        # memory: [B,N,M], k: [B,M]
        # Expand key across rows and compute cosine similarity per row
        k_exp = k.unsqueeze(1)                              # [B,1,M]
        k_rows = k_exp.expand(-1, memory.size(1), -1)       # [B,N,M]
        sim = F.cosine_similarity(memory, k_rows, dim=-1)   # [B,N]
        beta_ = beta.view(-1, 1)                            # [B,1]
        wc = F.softmax(beta_ * sim, dim=1)                  # [B,N]

        # Interpolate with previous weighting
        wg = g * wc + (1 - g) * w_prev  # [B,N]

        # Circular shift with 3 taps: [-1, 0, +1]
        s_m1 = s[:, 0:1]
        s_0 = s[:, 1:2]
        s_p1 = s[:, 2:3]
        w_shift = s_m1 * torch.roll(wg, shifts=1, dims=1) + s_0 * wg + s_p1 * torch.roll(wg, shifts=-1, dims=1)

        # Sharpen
        w = torch.clamp(w_shift, min=1e-20) ** gamma
        w = w / (w.sum(dim=1, keepdim=True) + 1e-20)
        return w

    @staticmethod
    def _ntm_read(memory: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # [B,N] x [B,N,M] -> [B,M]
        return torch.bmm(w.unsqueeze(1), memory).squeeze(1)

    @staticmethod
    def _ntm_write(memory: torch.Tensor, w: torch.Tensor, erase: torch.Tensor, add: torch.Tensor) -> torch.Tensor:
        # erase/add are [B,M]
        w_exp = w.unsqueeze(-1)            # [B,N,1]
        erase_mat = torch.sigmoid(erase).unsqueeze(1)   # [B,1,M]
        add_mat = add.unsqueeze(1)         # [B,1,M]
        mem = memory * (1 - w_exp * erase_mat) + (w_exp * add_mat)
        return mem

    # MCP helpers
    def _gate(self, mcp_gates: Optional[Dict[str, torch.Tensor]], key: str) -> Optional[torch.Tensor]:
        if mcp_gates is None:
            return None
        return mcp_gates.get(key, None)

    def _should_skip(self, mcp_gates: Optional[Dict[str, torch.Tensor]], key: str) -> bool:
        g = self._gate(mcp_gates, key)
        if g is None:
            return False
        if self.training:
            return False
        if not getattr(self.config, 'mcp_auto_features', True):
            return False
        if not getattr(self.config, 'mcp_hard_eval', True):
            return False
        try:
            return bool((g == 0).all().item())
        except Exception:
            return False

    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
            rope_mscale=float(getattr(self, 'rope_mscale', 1.0)) if hasattr(self, 'rope_mscale') else None,
        )

        # Input encoding
        # MCP gates (sample once)
        mcp_gates = None
        g_puzzle = None
        if self.mcp_head is not None:
            raw = self.mcp_head(carry.z_H[:, 0])
            # Probabilistic gate in train, deterministic in eval
            if self.training:
                u = torch.rand_like(raw)
                g_probs = torch.sigmoid((torch.log(u + 1e-9) - torch.log(1 - u + 1e-9) + raw) / max(self.config.mcp_temp, 1e-4))
                g = g_probs
            else:
                g_probs = torch.sigmoid(raw)
                if getattr(self.config, 'mcp_hard_eval', True):
                    thr = float(getattr(self.config, 'mcp_eval_threshold', 0.5))
                    g = (g_probs >= thr).to(g_probs.dtype)
                else:
                    g = g_probs
            # Map to named gates
            mcp_gates = {k: g[..., i] for i, k in enumerate(self.mcp_feature_keys[: g.shape[-1]])}
            # Backward-compat aliases if missing
            for k in ['puzzle','halt','gate','headbias','routing','film','rope','sched']:
                if k not in mcp_gates and g.shape[-1] > 0:
                    mcp_gates[k] = g[..., 0]
            g_puzzle = mcp_gates.get('puzzle', None)

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"], g_puzzle=g_puzzle)

        # (second MCP block removed)

        # Compute ACT scheduler gates (outside no-grad to allow learning via 1-step grad)
        if self.act_scheduler is not None:
            sched_in = torch.cat([carry.z_H[:, 0], carry.z_L[:, 0]], dim=-1)
            if self.sched_proj is not None:
                sched_in = self.sched_proj(sched_in)
            sched_raw = self.act_scheduler(sched_in)
            gate_h2l = torch.sigmoid(sched_raw[..., 0]).view(-1, 1, 1)
            gate_l2h = torch.sigmoid(sched_raw[..., 1]).view(-1, 1, 1)
            halt_bias = (self.config.act_sched_bias_scale * sched_raw[..., 2]).to(torch.float32)
            if mcp_gates is not None:
                scale = mcp_gates['sched'].view(-1, 1, 1).to(gate_h2l.dtype)
                gate_h2l = scale * gate_h2l
                gate_l2h = scale * gate_l2h
                halt_bias = mcp_gates['sched'].to(halt_bias.dtype) * halt_bias
        else:
            gate_h2l = torch.ones((batch["inputs"].shape[0], 1, 1), dtype=torch.float32, device=batch["inputs"].device)
            gate_l2h = torch.ones_like(gate_h2l)
            halt_bias = torch.zeros((batch["inputs"].shape[0],), dtype=torch.float32, device=batch["inputs"].device)

        # Cast gates to forward dtype to avoid implicit upcasts
        gate_h2l = gate_h2l.to(self.forward_dtype)
        gate_l2h = gate_l2h.to(self.forward_dtype)

        # Determine dynamic cycles from MCP (defaults to config if not present)
        dyn_h_max = int(getattr(self.config, 'max_h_cycles', 0)) or int(self.config.H_cycles)
        dyn_l_max = int(getattr(self.config, 'max_l_cycles', 0)) or int(self.config.L_cycles)
        if mcp_gates is not None:
            g_h = mcp_gates.get('h_cycles', None)
            g_l = mcp_gates.get('l_cycles', None)
            if g_h is not None:
                # map [0,1] -> [1, dyn_h_max]
                dyn_h = (1 + torch.clamp((g_h.mean() * (dyn_h_max - 1)).round(), min=0, max=dyn_h_max - 1)).to(torch.int32).item()
            else:
                dyn_h = dyn_h_max
            if g_l is not None:
                dyn_l = (1 + torch.clamp((g_l.mean() * (dyn_l_max - 1)).round(), min=0, max=dyn_l_max - 1)).to(torch.int32).item()
            else:
                dyn_l = dyn_l_max
        else:
            dyn_h = dyn_h_max
            dyn_l = dyn_l_max
        # record for telemetry
        dyn_h_tensor = torch.as_tensor(float(dyn_h), dtype=torch.float32, device=carry.z_H.device)
        dyn_l_tensor = torch.as_tensor(float(dyn_l), dtype=torch.float32, device=carry.z_H.device)

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(dyn_h):
                for _L_step in range(dyn_l):
                    if not ((_H_step == dyn_h - 1) and (_L_step == dyn_l - 1)):
                        z_L = self.L_level(z_L, (gate_h2l * z_H) + input_embeddings, **seq_info)

                if not (_H_step == dyn_h - 1):
                    z_H = self.H_level(z_H, (gate_l2h * z_L), **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, (gate_h2l * z_H) + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, (gate_l2h * z_L), **seq_info, mcp_gates=mcp_gates)

        # NTM scratchpad step (write/read) and inject into summary before LM head
        ntm_reg = None
        ntm_metrics = {}
        if self.ntm_enabled:
            # Controller embedding from summary
            ctrl = z_H[:, 0]
            if self.ntm_proj is not None:
                ctrl = self.ntm_proj(ctrl)

            # Unpack previous memory and weights
            ntm_mem = carry.ntm_memory if carry.ntm_memory is not None else self.ntm_mem_bias.unsqueeze(0).repeat(z_H.shape[0], 1, 1)
            read_w_prev = carry.ntm_read_w if carry.ntm_read_w is not None else torch.zeros(z_H.shape[0], self.ntm_N, dtype=self.forward_dtype, device=z_H.device)
            write_w_prev = carry.ntm_write_w if carry.ntm_write_w is not None else torch.zeros(z_H.shape[0], self.ntm_N, dtype=self.forward_dtype, device=z_H.device)

            # Write then Read (default)
            def read_head(ctrl_vec, mem, w_prev):
                o = self.ntm_read_fc(ctrl_vec.to(self.forward_dtype))
                k = o[..., : self.ntm_M]
                beta = o[..., self.ntm_M : self.ntm_M + 1].squeeze(-1)
                g = o[..., self.ntm_M + 1 : self.ntm_M + 2].squeeze(-1)
                s_logits = o[..., self.ntm_M + 2 : self.ntm_M + 5]
                gamma = o[..., self.ntm_M + 5 : self.ntm_M + 6].squeeze(-1)
                w = self._ntm_address(mem, k, beta, g, s_logits, gamma, w_prev)
                r = self._ntm_read(mem, w)
                return r, w

            def write_head(ctrl_vec, mem, w_prev):
                o = self.ntm_write_fc(ctrl_vec.to(self.forward_dtype))
                # splits
                off = 0
                k = o[..., off : off + self.ntm_M]; off += self.ntm_M
                beta = o[..., off : off + 1].squeeze(-1); off += 1
                g = o[..., off : off + 1].squeeze(-1); off += 1
                s_logits = o[..., off : off + 3]; off += 3
                gamma = o[..., off : off + 1].squeeze(-1); off += 1
                e = o[..., off : off + self.ntm_M]; off += self.ntm_M
                a = o[..., off : off + self.ntm_M]; off += self.ntm_M
                w = self._ntm_address(mem, k, beta, g, s_logits, gamma, w_prev)
                mem2 = self._ntm_write(mem, w, e.to(self.forward_dtype), a.to(self.forward_dtype))
                return mem2, w, e

            if getattr(self.config, "ntm_rw_order", "write_then_read") == "read_then_write":
                r, read_w = read_head(ctrl, ntm_mem, read_w_prev)
                ntm_mem2, write_w, erase_v = write_head(ctrl, ntm_mem, write_w_prev)
            else:
                ntm_mem2, write_w, erase_v = write_head(ctrl, ntm_mem, write_w_prev)
                r, read_w = read_head(ctrl, ntm_mem2, read_w_prev)

            # Inject read into summary token
            inject = self.ntm_read_to_hidden(r.to(self.forward_dtype))
            if getattr(self.config, "ntm_gate_from_mcp", True) and (mcp_gates is not None):
                # Prefer a dedicated NTM gate; fallback to 'routing' or 'gate'
                g_inj = mcp_gates.get('ntm', None)
                if g_inj is None:
                    g_inj = mcp_gates.get('routing', None)
                if g_inj is None:
                    g_inj = mcp_gates.get('gate', None)
                if g_inj is not None:
                    # Short-circuit heavy path in eval if auto features enabled
                    if (not self.training) and getattr(self.config, 'mcp_auto_features', True) and getattr(self.config, 'mcp_hard_eval', True):
                        if g_inj.ndim == 1 and bool((g_inj == 0).all().item()):
                            # Skip NTM entirely
                            inject = torch.zeros_like(inject)
                        else:
                            inject = inject * g_inj.view(-1, 1).to(inject.dtype)
                    else:
                        inject = inject * g_inj.view(-1, 1).to(inject.dtype)
            # Avoid in-place view assignment on z_H to keep autograd graph valid
            delta = torch.zeros_like(z_H)
            delta[:, 0, :] = inject
            z_H = z_H + delta

            # Regularizers and metrics
            eps = 1e-8
            read_ent = - (read_w.clamp_min(eps) * (read_w.clamp_min(eps).log())).sum(dim=1).mean()
            write_ent = - (write_w.clamp_min(eps) * (write_w.clamp_min(eps).log())).sum(dim=1).mean()
            erase_mag = torch.sigmoid(erase_v.to(torch.float32)).abs().mean()
            ent_coef = float(getattr(self.config, "ntm_entropy_reg", 0.0))
            erase_coef = float(getattr(self.config, "ntm_erase_reg", 0.0))
            # Per-feature cost: charge NTM usage
            feature_costs = getattr(self.config, 'mcp_feature_costs', {})
            ntm_cost = float(feature_costs.get('ntm', 0.0))
            ntm_reg = ent_coef * (read_ent + write_ent) + erase_coef * erase_mag + ntm_cost * 1.0
            ntm_metrics = {
                "ntm_read_entropy": read_ent.detach(),
                "ntm_write_entropy": write_ent.detach(),
                "ntm_erase_mag": erase_mag.detach(),
            }

            # Update new carry state for NTM
            new_ntm_mem = ntm_mem2.detach()
            new_read_w = read_w.detach()
            new_write_w = write_w.detach()
        
        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            ntm_memory=(new_ntm_mem if self.ntm_enabled else None),
            ntm_read_w=(new_read_w if self.ntm_enabled else None),
            ntm_write_w=(new_write_w if self.ntm_enabled else None),
        )
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_input = z_H[:, 0]
        if self.halt_proj is not None:
            q_input = self.halt_proj(q_input)
        q_logits = self.q_head(q_input).to(torch.float32)
        # MCP halting delta (do not zero logits)
        if getattr(self.config, 'mcp_enabled', False) and (self.halt_delta is not None) and (mcp_gates is not None):
                delta = self.halt_delta(q_input).to(torch.float32)
                q_logits = q_logits + mcp_gates['halt'].unsqueeze(-1).to(q_logits.dtype) * delta
        # Add novelty feature (difference to previous summary) if available
        if self.qkernel_feat is not None:
            prev_h = carry.z_H[:, 0]
            if self.halt_proj is not None:
                prev_h = self.halt_proj(prev_h)
            novelty = self.qkernel_feat((q_input - prev_h)).squeeze(-1).to(torch.float32)
            q_logits = q_logits + torch.stack([novelty, -novelty], dim=-1)
        # Subtract critic baseline from both logits to reduce variance
        if self.qcritic is not None:
            baseline = self.qcritic(q_input).squeeze(-1).to(torch.float32)
            q_logits = q_logits - baseline.unsqueeze(-1)
        # note: do not multiply logits by g_halt here (we add a delta above)
        if self.act_scheduler is not None:
            # Symmetric bias: +b to halt, -b to continue
            bias_vec = torch.stack([halt_bias, -halt_bias], dim=-1)
            q_logits = q_logits + bias_vec
        
        # MCP regularizer and per-feature costs
        mcp_cost = None
        extras = {}
        if mcp_gates is not None:
            # base cost + entropy
            if (self.config.mcp_cost_coef > 0 or self.config.mcp_entropy_coef > 0):
                gates = torch.stack(list(mcp_gates.values()), dim=-1).to(torch.float32)
                mcp_cost = self.config.mcp_cost_coef * gates.mean()
                if self.config.mcp_entropy_coef > 0:
                    p = gates.clamp(1e-6, 1-1e-6)
                    ent = - (p*torch.log(p) + (1-p)*torch.log(1-p)).mean()
                    mcp_cost = mcp_cost - self.config.mcp_entropy_coef * ent
            # FLOPs-aware approximate costs (normalized units)
            token_count = int(self.config.seq_len + getattr(self, 'puzzle_emb_len', 0))
            hidden = int(self.config.hidden_size)
            expansion = float(getattr(self.config, 'max_expansion', 0.0) or getattr(self.config, 'expansion', 4.0))
            H_layers = int(getattr(self.config, 'H_layers', 1))
            L_layers = int(getattr(self.config, 'L_layers', 1))
            base_h = int(getattr(self.config, 'H_cycles', 1))
            base_l = int(getattr(self.config, 'L_cycles', 1))
            # Relative per-layer cost ~ hidden*(1+expansion)
            per_layer_rel = hidden * (1.0 + expansion)
            cost_H_pass_rel = H_layers * per_layer_rel
            cost_L_pass_rel = L_layers * per_layer_rel
            extra_h = max(0, dyn_h - base_h)
            extra_l = max(0, dyn_l - base_l)
            # Normalize cycles cost to 0–1 scale based on extra cycles above base (robust to H-only or L-only)
            try:
                dyn_h_tensor = dyn_h_tensor.to(torch.float32)
                dyn_l_tensor = dyn_l_tensor.to(torch.float32)
            except Exception:
                dyn_h_tensor = torch.as_tensor(float(dyn_h), dtype=torch.float32, device=z_H.device)
                dyn_l_tensor = torch.as_tensor(float(dyn_l), dtype=torch.float32, device=z_H.device)
            max_h = int(getattr(self.config, 'max_h_cycles', 0))
            max_l = int(getattr(self.config, 'max_l_cycles', 0))
            h_on = 1 if max_h > 0 else 0
            l_on = 1 if max_l > 0 else 0
            h_denom = float(max(1, max_h - base_h)) if h_on else 1.0
            l_denom = float(max(1, max_l - base_l)) if l_on else 1.0
            h_rel = (torch.clamp(dyn_h_tensor - float(base_h), min=0.0) / h_denom) if h_on else torch.as_tensor(0.0, device=z_H.device)
            l_rel = (torch.clamp(dyn_l_tensor - float(base_l), min=0.0) / l_denom) if l_on else torch.as_tensor(0.0, device=z_H.device)
            denom = float(max(1, h_on + l_on))
            cycles_rel_t = (h_rel + l_rel) / denom
            # NTM relative cost ~ (N*M)/(hidden*token_count)
            # Keep NTM cost on ~0–1 scale via the gate mean
            ntm_rel_t = torch.as_tensor(0.0, dtype=torch.float32, device=z_H.device)
            if mcp_gates.get('ntm', None) is not None:
                ntm_rel_t = mcp_gates['ntm'].mean().to(torch.float32)
            # Routing block cost ~ token_count (normalized by token_count)
            routing_rel_t = torch.as_tensor(0.0, dtype=torch.float32, device=z_H.device)
            if mcp_gates.get('routing', None) is not None:
                routing_rel_t = mcp_gates['routing'].mean().to(torch.float32)
            # Sum
            addl_cost = cycles_rel_t + ntm_rel_t + routing_rel_t
            extras['mcp_cost_cycles'] = cycles_rel_t.detach()
            extras['mcp_cost_ntm_flops'] = ntm_rel_t.detach()
            extras['mcp_cost_routing'] = routing_rel_t.detach()
            if isinstance(addl_cost, torch.Tensor):
                mcp_cost = (mcp_cost + addl_cost) if (mcp_cost is not None) else addl_cost
            else:
                mcp_cost = (mcp_cost + torch.as_tensor(addl_cost, dtype=torch.float32, device=z_H.device)) if (mcp_cost is not None) else torch.as_tensor(addl_cost, dtype=torch.float32, device=z_H.device)
            # gate means
            for k, v in mcp_gates.items():
                extras[f'gate_{k}'] = v.mean().detach().to(torch.float32)
            # record chosen cycles
            extras['dyn_h_cycles'] = dyn_h_tensor.detach()
            extras['dyn_l_cycles'] = dyn_l_tensor.detach()
        # Attach extras (NTM metrics + MCP metrics)
        if ntm_reg is not None:
            extras['ntm_reg'] = ntm_reg if torch.is_tensor(ntm_reg) else torch.as_tensor(ntm_reg, dtype=torch.float32, device=z_H.device)
        extras.update(ntm_metrics)
        self._ntm_extras = extras if len(extras) > 0 else None
        # Ensure mcp_cost is a scalar (mean over batch) for consistent train/eval logging
        if isinstance(mcp_cost, torch.Tensor) and mcp_cost.ndim > 0:
            mcp_cost = mcp_cost.mean()
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), mcp_cost


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return getattr(self.inner, "puzzle_emb", None)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        dev = batch["inputs"].device

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=dev),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=dev),  # Default to halted
            
            hazard_history=torch.zeros((batch_size, self.config.halt_max_steps), dtype=torch.float32, device=dev),
            
            current_data={k: torch.empty_like(v, device=dev) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), mcp_cost = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        # propagate MCP cost if present
        if mcp_cost is not None:
            outputs["mcp_cost"] = mcp_cost
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            # Per-sample MCP-driven min/max steps
            max_steps_ceiling = int(getattr(self.config, 'halt_max_steps', 0))
            min_steps_ceiling = int(getattr(self.config, 'halt_min_steps_ceiling', 0)) or int(getattr(self.config, 'halt_min_steps', 0))
            eff_max = torch.full_like(new_steps, fill_value=max_steps_ceiling)
            eff_min = torch.full_like(new_steps, fill_value=min_steps_ceiling)
            _mg2 = locals().get('mcp_gates', None)
            if _mg2 is not None:
                if 'max_steps' in _mg2:
                    gmx = _mg2['max_steps'].to(torch.float32).view(-1)
                    eff_max = 1 + (gmx * (max_steps_ceiling - 1)).round().to(new_steps.dtype)
                if 'min_steps' in _mg2 and min_steps_ceiling > 0:
                    gmn = _mg2['min_steps'].to(torch.float32).view(-1)
                    eff_min = (gmn * min_steps_ceiling).round().to(new_steps.dtype)
            # ensure eff_min <= eff_max
            eff_min = torch.minimum(eff_min, eff_max)

            is_last_step = new_steps >= eff_max
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration (per-sample random min steps within [2, eff_max])
                explore_mask = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                eff_max_f = eff_max.to(torch.float32)
                u = torch.rand_like(eff_max_f)
                # 2 + floor(u * (eff_max - 1)) yields [2, eff_max] per-sample
                rand_steps = 2.0 + torch.floor(u * torch.clamp(eff_max_f - 1.0, min=0.0))
                rand_steps = torch.minimum(rand_steps, eff_max_f).to(new_steps.dtype)
                min_halt_steps = torch.where(explore_mask, rand_steps, torch.zeros_like(new_steps))
                # Enforce a training-time minimum-step floor (if configured)
                min_floor = eff_min if (int(getattr(self.config, 'halt_min_steps', 0)) > 0 or int(getattr(self.config, 'halt_min_steps_ceiling', 0)) > 0) else torch.zeros_like(new_steps)
                if (min_floor > 0).any():
                    # Require at least halt_min_steps before halting
                    min_halt_steps = torch.maximum(min_halt_steps, min_floor)
                    # Penalize attempted early halts (policy says halt before floor)
                    if self.config.act_min_step_penalty > 0.0:
                        early_mask = ((q_halt_logits > q_continue_logits) & (new_steps < min_floor))
                        early_penalty = early_mask.to(torch.float32).mean() * float(self.config.act_min_step_penalty)
                        if "mcp_cost" in outputs:
                            outputs["mcp_cost"] = outputs["mcp_cost"] + early_penalty
                        else:
                            outputs["mcp_cost"] = early_penalty

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                _, _, (next_q_halt_logits, next_q_continue_logits), _ = self.inner(new_inner_carry, new_current_data)
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

            # Ponder-style hazard tracking and deterministic eval halting
            # Compute per-step hazard p_t from Q logits
            p_t = torch.sigmoid((q_halt_logits - q_continue_logits).to(torch.float32))

            # Reset hazard history for sequences that were previously halted (new episodes)
            new_hazard_history = torch.where(
                carry.halted.view(-1, 1),
                torch.zeros_like(carry.hazard_history),
                carry.hazard_history
            )

            # Write p_t into history at index (new_steps - 1)
            step_idx = (new_steps - 1).clamp_min(0).clamp_max(self.config.halt_max_steps - 1).to(torch.long)
            new_hazard_history = new_hazard_history.clone()
            new_hazard_history.scatter_(1, step_idx.view(-1, 1), p_t.view(-1, 1))

            # Optional deterministic eval early-exit using cumulative halt prob (MCP-ponder gated)
            if (not self.training) and getattr(self.config, "ponder_eval_deterministic", False) and (self.config.halt_max_steps > 1):
                # cumulative halt prob up to current step: 1 - prod_{k<=t}(1 - p_k)
                p_hist = new_hazard_history.clamp(1e-6, 1 - 1e-6)
                log_survival = torch.log(1 - p_hist).cumsum(dim=1)
                cum_halt_prob = 1 - torch.exp(log_survival)
                cum_at_t = cum_halt_prob.gather(1, step_idx.view(-1, 1)).squeeze(1)
                base_thres = float(getattr(self.config, "ponder_eval_threshold", 0.5))
                thres_vec = torch.full_like(cum_at_t, fill_value=base_thres, dtype=cum_at_t.dtype)
                _mg = locals().get('mcp_gates', None)
                if (_mg is not None) and ('ponder' in _mg):
                    # Lower threshold when ponder gate is high (halt earlier)
                    g = _mg['ponder'].to(thres_vec.dtype).clamp(0, 1)
                    thres_vec = base_thres * (1.0 - 0.5 * g)
                halted = halted | (cum_at_t >= thres_vec)

                # Optional: best-step selection by halting when current step is argmax of halt distribution so far
                if getattr(self.config, 'ponder_eval_best_step', False):
                    # halt_dist[t] = (prod_{k<t}(1-p_k)) * p_t
                    log_1mp = torch.log(1 - p_hist)
                    excl_log_surv = F.pad(log_1mp.cumsum(dim=1)[:, :-1], (1, 0), value=0.0)
                    halt_dist = torch.exp(excl_log_surv) * p_hist
                    best_idx = halt_dist.argmax(dim=1)
                    halted = halted | (step_idx.view(-1) == best_idx)

            # If ponder KL is enabled, compute KL for newly halted sequences this step
            if getattr(self.config, "ponder_enabled", False) and (getattr(self.config, "ponder_kl_div_loss_weight", 0.0) > 0.0):
                newly_halted = halted & (~carry.halted)

                if newly_halted.any():
                    B, S = new_hazard_history.shape
                    # halting distribution from hazards: exclusive survival * hazard
                    p_hist = new_hazard_history.clamp(1e-6, 1 - 1e-6)
                    log_1mp = torch.log(1 - p_hist)
                    # exclusive cumsum: shift right by 1 with zeros in log-space
                    excl_log_surv = F.pad(log_1mp.cumsum(dim=1)[:, :-1], (1, 0), value=0.0)
                    halt_dist = torch.exp(excl_log_surv) * p_hist  # [B, S]

                    # geometric prior over steps 1..S
                    lam = float(getattr(self.config, "ponder_lambda_p", 0.2))
                    steps = torch.arange(S, device=halt_dist.device, dtype=torch.float32)
                    geom = ((1 - lam) ** steps) * lam  # t index 0..S-1 corresponds to step 1..S
                    geom = geom.view(1, -1).expand(B, -1)

                    # Mask tail beyond current step index per sample
                    # Build mask M[b, t] = 1 if t <= step_idx[b], else 0
                    t_idx = step_idx.view(-1, 1).to(torch.long)
                    rng = torch.arange(S, device=halt_dist.device).view(1, -1)
                    mask = (rng <= t_idx)

                    geom_m = geom * mask
                    halt_m = halt_dist * mask

                    eps = 1e-20
                    kl_per_sample = (geom_m * (torch.log(geom_m + eps) - torch.log(halt_m + eps))).sum(dim=1)
                    kl_value = kl_per_sample[newly_halted].mean()
                else:
                    kl_value = torch.tensor(0.0, device=new_hazard_history.device, dtype=torch.float32)

                # Optionally modulate by MCP ponder gate (defer to loss for stability)
                outputs["ponder_kl"] = kl_value

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_hazard_history, new_current_data), outputs
