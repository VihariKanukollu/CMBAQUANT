from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

try:
    from flash_attn_interface import flash_attn_func  # type: ignore[import]
except ImportError:
    # Fallback to FlashAttention 2
    from flash_attn import flash_attn_func  # type: ignore[import]

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def _repeat_kv(x: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """Repeat key/value heads to match query heads for GQA/MQA.

    x: [bs, seq_len, num_kv_heads, head_dim]
    returns: [bs, seq_len, num_heads, head_dim]
    """
    if num_kv_heads == num_heads:
        return x
    assert (num_heads % num_kv_heads) == 0, "num_heads must be multiple of num_kv_heads"
    n_rep = num_heads // num_kv_heads
    bs, slen, h, d = x.shape
    return (
        x.unsqueeze(3)
         .expand(bs, slen, h, n_rep, d)
         .reshape(bs, slen, h * n_rep, d)
    )


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class YarnRotaryEmbedding(nn.Module):
    """YARN-style RoPE extension with mscale stabilization for long context.

    Ref: DeepSeek and long-context RoPE smoothing. Stores `mscale` to optionally
    rescale attention (by scaling Q/K).
    """
    def __init__(self, dim: int, max_position_embeddings: int, base: float,
                 original_seq_len: int, rope_factor: float, beta_fast: int = 32, beta_slow: int = 1,
                 mscale_base: float = 1.0, device=None):
        super().__init__()
        self.mscale = float(0.1 * mscale_base * (torch.log(torch.tensor(rope_factor, dtype=torch.float32)).item()) + 1.0)

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        if max_position_embeddings > original_seq_len and rope_factor > 1.0:
            def find_correction_dim(num_rotations, dim, base, max_seq_len):
                return dim * torch.log(torch.tensor(max_seq_len / (num_rotations * 2 * torch.pi))) / (2 * torch.log(torch.tensor(base)))

            def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
                low = torch.floor(find_correction_dim(low_rot, dim, base, max_seq_len)).int().item()
                high = torch.ceil(find_correction_dim(high_rot, dim, base, max_seq_len)).int().item()
                return max(low, 0), min(high, dim - 1)

            def linear_ramp_factor(min_v, max_v, dim_v):
                if min_v == max_v:
                    max_v = max_v + 1e-3
                linear_func = (torch.arange(dim_v, dtype=torch.float32, device=device) - min_v) / (max_v - min_v)
                return torch.clamp(linear_func, 0, 1)

            low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            # Adjust frequencies across dims for extended window
            freqs_base = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            freqs_adj = freqs_base / rope_factor * (1 - smooth) + freqs_base * smooth
            freqs = torch.outer(t, freqs_adj)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, *, per_head_scale: Optional[torch.Tensor] = None, per_head_phase: Optional[torch.Tensor] = None, rope_mscale: Optional[float] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # Optional per-head phase pre-rotation (shape [B, H])
        if per_head_phase is not None:
            cosb = torch.cos(per_head_phase).view(batch_size, 1, self.num_heads, 1)
            sinb = torch.sin(per_head_phase).view(batch_size, 1, self.num_heads, 1)
            query = (query * cosb) + (rotate_half(query) * sinb)
            key = (key * cosb) + (rotate_half(key) * sinb)

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Optional YARN mscale scaling (applied to dot-product scale via Q/K)
        if rope_mscale is not None and rope_mscale != 1.0:
            scale = torch.tensor(rope_mscale, dtype=query.dtype, device=query.device)
            query = query * scale
            key = key * scale

        # Optional per-head scaling (shape: [B, H] or [B, H, 1, 1])
        if per_head_scale is not None:
            if per_head_scale.ndim == 2:
                scale = per_head_scale.view(batch_size, 1, self.num_heads, 1)
            else:
                scale = per_head_scale
            query = query * scale.to(query.dtype)

        # Repeat KV heads for GQA if needed
        if self.num_key_value_heads != self.num_heads:
            key = _repeat_kv(key, self.num_heads, self.num_key_value_heads)
            value = _repeat_kv(value, self.num_heads, self.num_key_value_heads)

        # flash attn
        attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
        if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
            attn_output = attn_output[0]

        # attn_output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
