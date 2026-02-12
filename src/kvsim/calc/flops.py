"""Transformer 每层 FLOPs 计算。

Prefill 阶段 (处理完整 prompt, 序列长度 s):
  Attention:
    Q 投影:   2 × s × h × (n_q × d_h)
    K 投影:   2 × s × h × (n_kv × d_h)
    V 投影:   2 × s × h × (n_kv × d_h)
    QK^T:     2 × n_q × s² × d_h
    Score×V:  2 × n_q × s² × d_h
    O 投影:   2 × s × (n_q × d_h) × h

  FFN (SwiGLU, 3 个线性层):
    Gate + Up + Down = 6 × s × h × d_ff

Decode 阶段 (逐 token, 上下文长度 s_ctx):
    与 Prefill 相同结构，但 s=1，注意力中 K/V 的序列长度为 s_ctx

参考: DESIGN.md 第 4.3 节
"""

from __future__ import annotations

from kvsim.models.config import ModelConfig


def attn_flops_per_layer(model: ModelConfig, seq_len: int) -> float:
    """单层 Attention 的 FLOPs (Prefill 阶段)。

    = 2s×h×(n_q + 2×n_kv)×d_h + 4×n_q×s²×d_h + 2s×h×n_q×d_h
    其中最后一项为 O 投影, 当 n_q×d_h = h 时等价于 2s×h²
    """
    s = seq_len
    h = model.hidden_size
    n_q = model.num_attention_heads
    n_kv = model.effective_kv_heads
    d_h = model.effective_head_dim

    # QKV 投影
    qkv = 2.0 * s * h * (n_q + 2 * n_kv) * d_h
    # 注意力分数 QK^T + Score×V
    attn_score = 4.0 * n_q * s * s * d_h
    # Output 投影
    o_proj = 2.0 * s * (n_q * d_h) * h

    return qkv + attn_score + o_proj


def ffn_flops_per_layer(model: ModelConfig, seq_len: int) -> float:
    """单层 FFN 的 FLOPs (SwiGLU: Gate + Up + Down = 3 个矩阵)。

    = 6 × s × h × d_ff
    """
    return 6.0 * seq_len * model.hidden_size * model.intermediate_size


def total_flops_per_layer(model: ModelConfig, seq_len: int) -> float:
    """单层总 FLOPs (Prefill 阶段) = Attention + FFN。"""
    return attn_flops_per_layer(model, seq_len) + ffn_flops_per_layer(model, seq_len)


def attn_flops_per_layer_decode(model: ModelConfig, context_len: int) -> float:
    """单层 Attention 的 FLOPs (Decode 阶段, 生成 1 个新 token)。

    QKV 投影: s=1
    注意力: Q (1×d_h) × K^T (d_h×s_ctx) → 2×n_q×1×s_ctx×d_h
    """
    h = model.hidden_size
    n_q = model.num_attention_heads
    n_kv = model.effective_kv_heads
    d_h = model.effective_head_dim

    qkv = 2.0 * h * (n_q + 2 * n_kv) * d_h
    attn_score = 4.0 * n_q * context_len * d_h
    o_proj = 2.0 * (n_q * d_h) * h

    return qkv + attn_score + o_proj


def ffn_flops_per_layer_decode(model: ModelConfig) -> float:
    """单层 FFN 的 FLOPs (Decode 阶段, s=1)。"""
    return 6.0 * model.hidden_size * model.intermediate_size


def total_flops_per_layer_decode(model: ModelConfig, context_len: int) -> float:
    """单层总 FLOPs (Decode 阶段)。"""
    return (
        attn_flops_per_layer_decode(model, context_len)
        + ffn_flops_per_layer_decode(model)
    )
