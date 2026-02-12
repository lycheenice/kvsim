"""KV Cache 显存用量计算。

核心公式:
  标准 Transformer (MHA/GQA/MQA):
    kv_per_token_per_layer = 2 × num_kv_heads × head_dim × dtype_bytes
    kv_total = num_layers × seq_len × kv_per_token_per_layer

  MLA (DeepSeek-V2/V3):
    kv_per_token_per_layer = (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
    kv_total = num_layers × seq_len × kv_per_token_per_layer

参考: DESIGN.md 第 3 节
"""

from __future__ import annotations

from kvsim.models.config import DTYPE_BYTES, KVCacheEstimation, ModelConfig


def kv_per_token_per_layer_bytes(model: ModelConfig, dtype: str) -> float:
    """计算单层单 token 的 KV Cache 字节数。

    MLA: (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
    标准: 2 × num_kv_heads × head_dim × dtype_bytes
    """
    dbytes = DTYPE_BYTES[dtype]

    if model.attention_type == "mla":
        if model.kv_lora_rank is None or model.qk_rope_head_dim is None:
            raise ValueError(
                f"模型 {model.name} 为 MLA 类型，但缺少 kv_lora_rank 或 qk_rope_head_dim"
            )
        return (model.kv_lora_rank + model.qk_rope_head_dim) * dbytes

    return 2.0 * model.effective_kv_heads * model.effective_head_dim * dbytes


def estimate_kv_memory(
    model: ModelConfig,
    seq_len: int,
    dtype: str = "fp16",
    tp_size: int = 1,
) -> KVCacheEstimation:
    """估算 KV Cache 总显存。

    Args:
        model: 模型配置
        seq_len: 序列长度 (prompt + generated tokens)
        dtype: 数据类型 ("fp16", "bf16", "fp8", "int8", "int4")
        tp_size: Tensor Parallelism 大小，KV Cache 按 tp_size 切分
    """
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"不支持的 dtype: {dtype}，可选: {list(DTYPE_BYTES.keys())}")

    per_tok = kv_per_token_per_layer_bytes(model, dtype)
    per_layer = per_tok * seq_len
    total = per_layer * model.num_hidden_layers

    # Tensor Parallelism: KV Cache 按 KV heads 切分到各卡
    if tp_size > 1:
        total /= tp_size
        per_layer /= tp_size

    return KVCacheEstimation(
        model_name=model.name,
        seq_len=seq_len,
        dtype=dtype,
        tp_size=tp_size,
        kv_per_token_per_layer_bytes=per_tok,
        kv_per_layer_bytes=per_layer,
        kv_total_bytes=total,
        kv_total_gb=total / (1024**3),
    )
