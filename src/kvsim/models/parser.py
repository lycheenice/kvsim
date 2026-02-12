"""HuggingFace config.json 解析器。

支持的模型架构:
- Llama 系列 (model_type: "llama")
- Qwen2 系列 (model_type: "qwen2")
- ChatGLM 系列 (model_type: "chatglm", "glm4")
- Mistral 系列 (model_type: "mistral")
- DeepSeek-V2/V3 (model_type: "deepseek_v2")
- 其他遵循标准字段命名的模型
"""

from __future__ import annotations

import json
from pathlib import Path

from kvsim.models.config import ModelConfig


def parse_hf_config(config_path: str | Path) -> ModelConfig:
    """从 HuggingFace config.json 解析模型配置。

    解析策略:
    - num_key_value_heads: 优先读取 num_key_value_heads，
      回退到 multi_query_group_num (ChatGLM)，再回退到 num_attention_heads (MHA)
    - head_dim: 优先读取显式字段，否则 hidden_size // num_attention_heads
    - MLA 检测: 通过 kv_lora_rank 字段存在性判断
    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "unknown")

    # --- 层数 ---
    num_layers: int = cfg.get("num_hidden_layers") or cfg.get("num_layers")
    if num_layers is None:
        raise ValueError("config.json 中缺少 num_hidden_layers 或 num_layers 字段")

    # --- hidden_size ---
    hidden_size: int = cfg["hidden_size"]

    # --- num_attention_heads ---
    num_attention_heads: int = cfg["num_attention_heads"]

    # --- num_key_value_heads (按优先级回退) ---
    num_kv_heads: int = (
        cfg.get("num_key_value_heads")
        or cfg.get("multi_query_group_num")  # ChatGLM 系列
        or num_attention_heads  # MHA
    )

    # --- head_dim ---
    head_dim: int | None = cfg.get("head_dim")
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads

    # --- intermediate_size ---
    intermediate_size: int = cfg["intermediate_size"]

    # --- 注意力类型判断 ---
    kv_lora_rank: int | None = cfg.get("kv_lora_rank")
    qk_rope_head_dim: int | None = cfg.get("qk_rope_head_dim")

    if kv_lora_rank is not None or model_type in ("deepseek_v2", "deepseek_v3"):
        attention_type = "mla"
    elif num_kv_heads == 1:
        attention_type = "mqa"
    elif num_kv_heads < num_attention_heads:
        attention_type = "gqa"
    else:
        attention_type = "mha"

    # --- 模型名称 ---
    name = cfg.get("_name_or_path", config_path.parent.name) or "custom"

    return ModelConfig(
        name=name,
        model_type=model_type,
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        attention_type=attention_type,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )
