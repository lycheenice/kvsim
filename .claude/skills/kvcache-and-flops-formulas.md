# KV Cache 与 FLOPs 计算公式

本文件记录 KV Cache 显存计算和每层 FLOPs 计算的完整公式推导。

## 1. KV Cache 显存计算

### 1.1 标准 Transformer（MHA / GQA / MQA）

每个 token 在每一层需要存储 Key 和 Value 两个张量:

```
K shape per token: [num_kv_heads, head_dim]
V shape per token: [num_kv_heads, head_dim]
```

**单层单 token KV Cache**:
```
kv_per_token_per_layer = 2 × num_kv_heads × head_dim × dtype_bytes
```

**单层 KV Cache（seq_len 个 token）**:
```
kv_per_layer = seq_len × 2 × num_kv_heads × head_dim × dtype_bytes
```

**总 KV Cache**:
```
kv_total = num_layers × seq_len × 2 × num_kv_heads × head_dim × dtype_bytes
```

**注意力类型对 num_kv_heads 的影响**:
| 类型 | num_kv_heads | 说明 |
|------|-------------|------|
| MHA | = num_attention_heads | 每个 Q head 对应独立的 KV head |
| GQA | < num_attention_heads, > 1 | 多个 Q head 共享一组 KV head |
| MQA | = 1 | 所有 Q head 共享一组 KV head |

### 1.2 MLA（Multi-head Latent Attention, DeepSeek-V2/V3）

MLA 将 KV 压缩到低秩潜在空间:
- 不存储完整 K 和 V
- 存储联合压缩向量 c_kv (维度 = kv_lora_rank)
- 加上 RoPE 部分的 key (维度 = qk_rope_head_dim)

```
kv_per_token_per_layer = (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
```

> 对比：DeepSeek-V3 标准 KV (假设 MHA) = 2 × 128 × 128 × 2 = 65536 bytes/token/layer
> 实际 MLA = (512 + 64) × 2 = 1152 bytes/token/layer → **压缩约 57 倍**

### 1.3 Sliding Window Attention

当模型使用滑动窗口注意力（如 Mistral）时:
```
effective_seq_len = min(seq_len, sliding_window)
kv_per_layer = effective_seq_len × 2 × num_kv_heads × head_dim × dtype_bytes
```

注意：部分模型（如 Qwen2.5）只在部分层使用滑动窗口，需要逐层计算。

### 1.4 dtype_bytes 对照

| 数据类型 | 字节数 | 说明 |
|---------|--------|------|
| fp32 | 4 | 训练用，推理少见 |
| fp16 | 2 | 推理默认 |
| bf16 | 2 | 推理默认 |
| fp8 (e4m3/e5m2) | 1 | H100+ 支持 |
| int8 | 1 | 量化 KV Cache |
| int4 | 0.5 | 激进量化 |

### 1.5 Tensor Parallelism 下的 KV Cache

```
kv_per_gpu = kv_total / tp_size  (当 num_kv_heads >= tp_size)
```

当 `num_kv_heads < tp_size` 时，某些 TP 实现会复制 KV heads，此时:
```
kv_per_gpu = kv_total × ceil(tp_size / num_kv_heads) / tp_size
```

一般推荐 `num_kv_heads` 能被 `tp_size` 整除。

## 2. FLOPs 计算

以下公式中:
- `s` = 序列长度 (seq_len)
- `h` = hidden_size
- `n_q` = num_attention_heads
- `n_kv` = num_key_value_heads
- `d_h` = head_dim
- `d_ff` = intermediate_size

矩阵乘法 `[M, K] × [K, N]` 的 FLOPs = `2 × M × K × N`（乘加各算一次）。

### 2.1 Prefill 阶段（处理完整 prompt）

#### Attention 部分

```
Q 投影:   2 × s × h × (n_q × d_h)          # [s, h] × [h, n_q*d_h]
K 投影:   2 × s × h × (n_kv × d_h)         # [s, h] × [h, n_kv*d_h]
V 投影:   2 × s × h × (n_kv × d_h)         # [s, h] × [h, n_kv*d_h]
QK^T:     2 × n_q × s × s × d_h            # 每个 head: [s, d_h] × [d_h, s]
Score×V:  2 × n_q × s × s × d_h            # 每个 head: [s, s] × [s, d_h]
O 投影:   2 × s × (n_q × d_h) × h          # [s, n_q*d_h] × [n_q*d_h, h]
```

简化（当 `n_q × d_h = h`）：
```
Attn_FLOPs = 4×s×h² + 4×s×h×n_kv×d_h + 4×n_q×s²×d_h
```

当 `n_kv = n_q`（MHA）时进一步简化:
```
Attn_FLOPs = 8×s×h² + 4×n_q×s²×d_h
```

#### FFN 部分

**SwiGLU**（Llama, Qwen, GLM-4 等主流模型）— 3 个线性层:
```
Gate: 2 × s × h × d_ff     # [s, h] × [h, d_ff]
Up:   2 × s × h × d_ff     # [s, h] × [h, d_ff]
Down: 2 × s × d_ff × h     # [s, d_ff] × [d_ff, h]

FFN_FLOPs = 6 × s × h × d_ff
```

**标准 FFN**（ReLU/GELU，2 个线性层）:
```
FFN_FLOPs = 4 × s × h × d_ff
```

#### 每层总 FLOPs（Prefill）:
```
Layer_FLOPs_prefill = Attn_FLOPs + FFN_FLOPs
                    = 4×s×h² + 4×s×h×n_kv×d_h + 4×n_q×s²×d_h + 6×s×h×d_ff
```

### 2.2 Decode 阶段（逐 token 生成）

Decode 每步只处理 1 个新 token，但需要访问完整 KV Cache:
```
s_new = 1
s_ctx = context_length (已有 token 数)
```

#### Attention 部分
```
Q 投影:   2 × 1 × h × (n_q × d_h)
K 投影:   2 × 1 × h × (n_kv × d_h)
V 投影:   2 × 1 × h × (n_kv × d_h)
QK^T:     2 × n_q × 1 × s_ctx × d_h      # [1, d_h] × [d_h, s_ctx]
Score×V:  2 × n_q × 1 × s_ctx × d_h      # [1, s_ctx] × [s_ctx, d_h]
O 投影:   2 × 1 × (n_q × d_h) × h

Attn_FLOPs_decode = 4×h² + 4×h×n_kv×d_h + 4×n_q×s_ctx×d_h
```

#### FFN 部分（与 prefill 相同公式，s=1）:
```
FFN_FLOPs_decode = 6 × h × d_ff
```

#### 每层总 FLOPs（Decode）:
```
Layer_FLOPs_decode = 4×h² + 4×h×n_kv×d_h + 4×n_q×s_ctx×d_h + 6×h×d_ff
```

> 注: Decode 阶段是 memory-bound，实际瓶颈是显存带宽而非算力，但 FLOPs 计算仍然需要用于带宽估算。

### 2.3 MoE 模型的 FFN FLOPs

对于 MoE 模型（如 DeepSeek-V3, Mixtral）:
```
# 每个 token 只激活 top-k 个 expert
FFN_FLOPs_moe = top_k × 6 × s × h × expert_intermediate_size + shared_expert_FLOPs

# DeepSeek-V3 例:
# top_k = 8, expert_intermediate_size = 2048, n_shared = 1, shared_intermediate = 18432
# FFN_FLOPs = 8 × 6 × s × 7168 × 2048 + 6 × s × 7168 × 18432
```

## 3. 计算时间估算

```
compute_time_per_layer = Layer_FLOPs / effective_FLOPS
effective_FLOPS = peak_FLOPS × utilization

# 默认 utilization = 0.5（按需求文档）
```

## 4. 数值验证示例

### Llama-3.1-70B, fp16, seq_len=4096

```
KV Cache:
  per_token_per_layer = 2 × 8 × 128 × 2 = 4096 bytes = 4 KB
  per_layer = 4096 × 4096 = 16,777,216 bytes = 16 MB
  total = 80 × 16 MB = 1,280 MB = 1.25 GB ✓

FLOPs per layer (Prefill, s=4096, h=8192, n_q=64, n_kv=8, d_h=128, d_ff=28672):
  Attn = 4×4096×8192² + 4×4096×8192×8×128 + 4×64×4096²×128
       = 1.10T + 0.137T + 0.549T = 1.786 TFLOP
  FFN = 6 × 4096 × 8192 × 28672 = 5.780 TFLOP
  Total = 7.566 TFLOP per layer

On H100 (989 TFLOPS × 0.5 = 494.5 TFLOPS effective):
  compute_time = 7.566T / 494.5T = 15.3 ms per layer

PD bandwidth:
  KV per layer = 16 MB
  min_bandwidth = 16 MB / 15.3 ms = 1.045 GB/s
  → 远低于 InfiniBand HDR (25 GB/s)，异步传输完全可以掩盖延迟
```
