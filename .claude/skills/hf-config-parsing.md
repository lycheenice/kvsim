# HuggingFace config.json 字段映射

本文件记录各主流模型家族在 HuggingFace config.json 中与 KV Cache 计算相关字段的命名差异。

## 通用字段（大多数模型）

| 语义 | 标准字段名 | 说明 |
|------|-----------|------|
| 模型类型标识 | `model_type` | "llama", "qwen2", "chatglm", "mistral", "deepseek_v2" 等 |
| Transformer 层数 | `num_hidden_layers` | 通用 |
| 隐藏层维度 | `hidden_size` | 通用 |
| Query 头数 | `num_attention_heads` | 通用 |
| KV 头数 | `num_key_value_heads` | GQA/MQA 模型提供，MHA 模型可能缺失 |
| 注意力头维度 | `head_dim` | 部分模型显式提供 |
| FFN 中间维度 | `intermediate_size` | 通用 |
| 词表大小 | `vocab_size` | 不影响 KV Cache，但影响 embedding 显存 |

## 各模型家族字段差异

### Llama 系列 (model_type: "llama")
- 适用: Llama-2, Llama-3, Llama-3.1, Code Llama
- 字段完全遵循标准命名
- `num_key_value_heads`: Llama-2-7B = 32 (MHA), Llama-3.1-8B = 8 (GQA)
- `head_dim`: 不显式提供，需计算 hidden_size // num_attention_heads = 128

```json
{
  "model_type": "llama",
  "num_hidden_layers": 80,
  "hidden_size": 8192,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "intermediate_size": 28672
}
```

### Qwen2 系列 (model_type: "qwen2")
- 适用: Qwen2, Qwen2.5
- 字段遵循标准命名
- 提供显式 `head_dim` 和 `use_sliding_window`

```json
{
  "model_type": "qwen2",
  "num_hidden_layers": 80,
  "hidden_size": 8192,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 29568
}
```

### ChatGLM 系列 (model_type: "chatglm" 或 "glm4")
- 适用: GLM-4, GLM-4.7, ChatGLM3
- **关键差异**: KV 头数字段名为 `multi_query_group_num` 而非 `num_key_value_heads`
- `padded_vocab_size` 代替 `vocab_size`
- 可能使用 `num_layers` 代替 `num_hidden_layers`
- `head_dim` 不显式提供

```json
{
  "model_type": "glm4",
  "num_hidden_layers": 40,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "multi_query_group_num": 2,
  "intermediate_size": 13696
}
```

**解析注意**: 读取 KV heads 时需按优先级:
1. `num_key_value_heads`
2. `multi_query_group_num`
3. 回退到 `num_attention_heads`（视为 MHA）

### Mistral 系列 (model_type: "mistral")
- 适用: Mistral-7B, Mixtral
- 字段遵循标准命名
- 特有 `sliding_window` 字段（影响实际 KV Cache 上限）

```json
{
  "model_type": "mistral",
  "num_hidden_layers": 32,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 14336,
  "sliding_window": 4096
}
```

**注意**: 当 `sliding_window` 存在且 seq_len > sliding_window 时，实际 KV Cache 可按 sliding_window 计算（仅在使用滑动窗口注意力时）。

### DeepSeek-V2/V3 (model_type: "deepseek_v2")
- **完全不同的 KV Cache 机制**: MLA (Multi-head Latent Attention)
- 不存储完整 K/V，存储压缩后的联合潜在向量
- 关键字段: `kv_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim`

```json
{
  "model_type": "deepseek_v2",
  "num_hidden_layers": 61,
  "hidden_size": 7168,
  "num_attention_heads": 128,
  "kv_lora_rank": 512,
  "qk_rope_head_dim": 64,
  "qk_nope_head_dim": 128,
  "v_head_dim": 128,
  "intermediate_size": 18432,
  "n_routed_experts": 256,
  "n_shared_experts": 1
}
```

**MLA KV Cache 公式**:
```
kv_per_token_per_layer = (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
                       = (512 + 64) × 2 = 1152 bytes (fp16)
```

### Phi 系列 (model_type: "phi3" 或 "phi")
- 字段遵循标准命名
- 有些版本使用 `num_key_value_heads`

### Yi 系列 (model_type: "llama")
- model_type 为 "llama"（基于 Llama 架构）
- 字段与 Llama 完全一致

### InternLM 系列 (model_type: "internlm2")
- 字段基本遵循标准命名
- 使用 GQA

## 解析策略总结

```python
# 伪代码：从 config.json 提取 KV Cache 相关参数
def extract_kv_params(config: dict):
    model_type = config.get("model_type", "unknown")

    # 层数
    num_layers = config.get("num_hidden_layers") or config.get("num_layers")

    # KV heads（按优先级）
    num_kv_heads = (
        config.get("num_key_value_heads")
        or config.get("multi_query_group_num")  # ChatGLM
        or config.get("num_attention_heads")     # MHA fallback
    )

    # Head dim
    head_dim = config.get("head_dim") or (
        config["hidden_size"] // config["num_attention_heads"]
    )

    # MLA 检测
    is_mla = "kv_lora_rank" in config

    # Sliding window
    sliding_window = config.get("sliding_window")  # None 表示无限

    return {
        "model_type": model_type,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "is_mla": is_mla,
        "kv_lora_rank": config.get("kv_lora_rank"),
        "qk_rope_head_dim": config.get("qk_rope_head_dim"),
        "sliding_window": sliding_window,
    }
```

## 需要验证的事项

> **重要**: 以上字段映射基于已知信息整理，在实际实现时需要下载真实的 config.json 进行验证，特别是:
> 1. GLM-4.7 (zai-org/GLM-4.7) 的确切字段名和值
> 2. DeepSeek-V3 的 MLA 相关字段的确切命名
> 3. Qwen2.5 是否所有尺寸都提供 head_dim
> 4. 各模型 intermediate_size 对于 MoE 模型的处理（single expert vs total）
