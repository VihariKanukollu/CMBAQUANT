from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
import torch._dynamo as dynamo
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

# PennyLane import for PQC components (required in quant variant)
import pennylane as qml  # type: ignore[import]
from pennylane.qnn import TorchLayer as _PLTorchLayer  # type: ignore[import]


class PennyLanePQC(nn.Module):
    """Generic PennyLane PQC wrapper (no fallbacks in quant variant).

    Maps input_dim -> n_wires (via a learned projection), runs a PQC with n_layers,
    then projects n_wires -> output_dim.
    """
    def __init__(self, input_dim: int, output_dim: int, n_wires: int, n_layers: int):
        super().__init__()
        assert input_dim > 0 and output_dim > 0 and n_wires > 0 and n_layers > 0
        self.n_wires = n_wires
        self.n_layers = n_layers

        self.input_proj = CastedLinear(input_dim, n_wires, bias=True)

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="torch")
        def _circuit(inputs, weights):  # type: ignore[no-redef]
            qml.AngleEmbedding(inputs, wires=list(range(n_wires)), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=list(range(n_wires)))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        weight_shapes = {"weights": (n_layers, n_wires, 3)}
        self.pqc = _PLTorchLayer(_circuit, weight_shapes)

        self.readout = CastedLinear(n_wires, output_dim, bias=True)

    @dynamo.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        orig_device = x.device
        h = self.input_proj(x.to(orig_dtype))
        # PennyLane TorchLayer expects float32 inputs typically and generally runs on CPU
        h32 = h.to(torch.float32)
        z = self.pqc(h32.cpu()).to(orig_device)
        return self.readout(z.to(orig_dtype))

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


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
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
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

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


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
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
        if config.quantum_gate_enabled and config.quantum_gate_dim > 0:
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
        if getattr(config, "per_head_bias_enabled", False):
            self.per_head_bias = QuantumGatingHead(
                input_dim=(config.quantum_gate_proj_dim if config.quantum_gate_proj_dim > 0 else config.hidden_size),
                gate_dim=config.num_heads,
                n_wires=config.pqc_n_wires,
                n_layers=config.pqc_n_layers,
            )
        if getattr(config, "token_routing_enabled", False):
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
        if getattr(config, "film_enabled", False) and not getattr(config, "pqc_shared", False):
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
        if getattr(config, "rope_phase_bias_enabled", False) and not getattr(config, "pqc_shared", False):
            gate_dim = (config.num_heads if config.rope_phase_bias_per_head else 1)
            self.rope_phase_per_head = config.rope_phase_bias_per_head
            self.rope_phase_scale = getattr(config, "rope_phase_bias_scale", 1.0)
            self.rope_phase_head = QuantumGatingHead(
                input_dim=(config.quantum_gate_proj_dim if config.quantum_gate_proj_dim > 0 else config.hidden_size),
                gate_dim=gate_dim,
                n_wires=config.pqc_n_wires,
                n_layers=config.pqc_n_layers,
            )
    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, *, block_index: int = 0, num_blocks: int = 1, module_role: str = "") -> torch.Tensor:
        # Post Norm
        # Compute optional gating from summary token
        gate_vals = None
        apply_gate = (self.quantum_gate is not None)
        if apply_gate and self.quantum_gate_last_h_block_only:
            apply_gate = (module_role == "H" and (block_index == (num_blocks - 1)))

        # Per-head bias and token routing only for last H block
        apply_head_bias = ((self.per_head_bias is not None) or (self.adapter_headbias is not None)) and (module_role == "H" and (block_index == (num_blocks - 1)))
        apply_token_routing = ((self.token_router is not None) or (self.adapter_router is not None)) and (module_role == "H" and (block_index == (num_blocks - 1)))

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

        per_head_scale = None
        if apply_head_bias:
            summary = hidden_states[:, 0]
            if self.gate_proj is not None:
                summary = self.gate_proj(summary)
            # map to per-head scale via tanh -> positive scaling
            if self.shared_trunk is not None and self.adapter_headbias is not None:
                raw = self.adapter_headbias(_trunk(summary))
            else:
                raw = self.per_head_bias(summary)
            head_bias = torch.tanh(raw) * getattr(self, 'per_head_bias_scale', 1.0)
            per_head_scale = (1.0 + head_bias)  # [B, H]

        if apply_token_routing:
            # Compute routing weights per token from similarity with summary
            summary = hidden_states[:, 0]
            if self.gate_proj is not None:
                summary = self.gate_proj(summary)
            # score per token = sigmoid(sim(summary, token))
            sim = torch.einsum('bd,btd->bt', summary, hidden_states)
            sim = sim / (hidden_states.shape[-1] ** 0.5)
            if self.shared_trunk is not None and self.adapter_router is not None:
                router_bias = self.adapter_router(_trunk(summary)).squeeze(-1)
                keep = torch.sigmoid(sim + router_bias)
            else:
                keep = torch.sigmoid(sim)
            # sharpen/threshold using keep_ratio
            keep_ratio = getattr(self, 'token_routing_keep_ratio', 1.0)
            if keep_ratio < 1.0:
                thresh = torch.quantile(keep, q=(1.0 - keep_ratio), dim=1, keepdim=True)
                keep = (keep >= thresh).to(hidden_states.dtype)
            keep = keep.view(keep.shape[0], keep.shape[1], 1)
            hidden_states = hidden_states * keep

        # Self Attention with optional gate on residual
        # Optional RoPE phase bias
        rope_phase = None
        if (module_role == "H" and (block_index == (num_blocks - 1))):
            if self.rope_phase_head is not None:
                summary = hidden_states[:, 0]
                if self.gate_proj is not None:
                    summary = self.gate_proj(summary)
                raw = self.rope_phase_head(summary)
                if not self.rope_phase_per_head:
                    raw = raw.expand(-1, self.self_attn.num_heads)
                rope_phase = raw * getattr(self, 'rope_phase_scale', 1.0)
            elif self.shared_trunk is not None and self.adapter_rope is not None:
                summary = hidden_states[:, 0]
                if self.gate_proj is not None:
                    summary = self.gate_proj(summary)
                raw = self.adapter_rope(_trunk(summary))
                raw = raw if raw.shape[-1] == self.self_attn.num_heads else raw.expand(-1, self.self_attn.num_heads)
                rope_phase = raw * getattr(self, 'rope_phase_scale', 1.0)

        attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states, per_head_scale=per_head_scale, per_head_phase=rope_phase)
        if gate_vals is not None and gate_vals.shape[-1] >= 1:
            gate_attn = gate_vals[..., 0].view(-1, 1, 1)
            hidden_states = rms_norm(hidden_states + gate_attn * attn_out, variance_epsilon=self.norm_eps)
        else:
            hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)

        # Fully Connected with optional gate on residual
        mlp_out = self.mlp(hidden_states)
        if gate_vals is not None and gate_vals.shape[-1] >= 2:
            gate_mlp = gate_vals[..., 1].view(-1, 1, 1)
            hidden_states = rms_norm(hidden_states + gate_mlp * mlp_out, variance_epsilon=self.norm_eps)
        else:
            hidden_states = rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)

        # FiLM conditioning on last H block
        if (module_role == "H" and (block_index == (num_blocks - 1))):
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
            else:
                film = None
            if film is not None:
                G_cfg = max(1, int(getattr(self, 'film_groups', 32)))
                Hdim = hidden_states.shape[-1]
                G = G_cfg
                if Hdim % G != 0:
                    import math as _math
                    G = _math.gcd(Hdim, G_cfg) or 1
                self.film_groups = G
                self.film_scale = getattr(self, 'film_scale', 1.0)
                gamma, beta = film[..., :G], film[..., G:]
                gamma = (1.0 + torch.tanh(gamma) * self.film_scale).view(-1, 1, G)
                beta = (torch.tanh(beta) * self.film_scale).view(-1, 1, G)
                group_size = Hdim // G
                if group_size > 0:
                    hs = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], G, group_size)
                    hs = hs * gamma.unsqueeze(-1) + beta.unsqueeze(-1)
                    hidden_states = hs.view(hidden_states.shape[0], hidden_states.shape[1], Hdim)
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
        for idx, layer in enumerate(self.layers):
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

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
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
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Compute ACT scheduler gates (outside no-grad to allow learning via 1-step grad)
        if self.act_scheduler is not None:
            sched_in = torch.cat([carry.z_H[:, 0], carry.z_L[:, 0]], dim=-1)
            if self.sched_proj is not None:
                sched_in = self.sched_proj(sched_in)
            sched_raw = self.act_scheduler(sched_in)
            gate_h2l = torch.sigmoid(sched_raw[..., 0]).view(-1, 1, 1)
            gate_l2h = torch.sigmoid(sched_raw[..., 1]).view(-1, 1, 1)
            halt_bias = (self.config.act_sched_bias_scale * sched_raw[..., 2]).to(torch.float32)
        else:
            gate_h2l = torch.ones((batch["inputs"].shape[0], 1, 1), dtype=torch.float32, device=batch["inputs"].device)
            gate_l2h = torch.ones_like(gate_h2l)
            halt_bias = torch.zeros((batch["inputs"].shape[0],), dtype=torch.float32, device=batch["inputs"].device)

        # Cast gates to forward dtype to avoid implicit upcasts
        gate_h2l = gate_h2l.to(self.forward_dtype)
        gate_l2h = gate_l2h.to(self.forward_dtype)

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, (gate_h2l * z_H) + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, (gate_l2h * z_L), **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, (gate_h2l * z_H) + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, (gate_l2h * z_L), **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_input = z_H[:, 0]
        if self.halt_proj is not None:
            q_input = self.halt_proj(q_input)
        q_logits = self.q_head(q_input).to(torch.float32)
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
        if self.act_scheduler is not None:
            # Symmetric bias: +b to halt, -b to continue
            bias_vec = torch.stack([halt_bias, -halt_bias], dim=-1)
            q_logits = q_logits + bias_vec
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        dev = batch["inputs"].device

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=dev),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=dev),  # Default to halted
            
            current_data={k: torch.empty_like(v, device=dev) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
