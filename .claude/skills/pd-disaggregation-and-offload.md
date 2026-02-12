# PD 分离与 KV Cache Offload 技术

本文件记录 Prefill-Decode 分离架构和 KV Cache Offload 的技术背景、带宽模型和估算方法。

## 1. PD 分离 (Prefill-Decode Disaggregation)

### 1.1 架构概述

传统 LLM 推理将 prefill 和 decode 放在同一 GPU 上，但两个阶段特征差异巨大:

| 特征 | Prefill | Decode |
|------|---------|--------|
| 计算模式 | 批量矩阵乘，compute-bound | 逐 token，memory-bound |
| 算力利用率 | 高 (60-80%) | 低 (< 5%) |
| 延迟特征 | 首 token 延迟 (TTFT) | 每 token 延迟 (TPOT) |
| 显存访问 | 高带宽利用 | KV Cache 随机读 |

PD 分离将两个阶段部署在不同的 GPU 池上:
- **Prefill 池**: 优化 TTFT，高算力利用
- **Decode 池**: 优化 TPOT，高显存带宽利用
- **KV Transfer**: Prefill 完成后将 KV Cache 通过高速网络传输到 Decode 节点

### 1.2 相关系统

| 系统 | 出处 | 核心思路 |
|------|------|---------|
| **DistServe** | OSDI'24 | 首次提出 PD 分离，通过将 prefill 和 decode 置于不同 GPU 集群来消除相互干扰 |
| **Splitwise** | ISCA'24 | 类似思路，分析了不同 GPU 类型适合不同阶段 |
| **Mooncake** | Moonshot AI, 2024 | KV Cache 中心化架构，KV Cache 存储在分布式缓存中，prefill/decode 均从缓存读写 |
| **TetriInfer** | 2024 | 将 prefill 拆分为多个 chunk，与 decode 在同一 GPU 上交错执行 |

### 1.3 逐层异步传输带宽模型

#### 流水线工作方式

```
时间 →
Layer 0: [====Compute====]
Layer 1:                   [====Compute====]
Layer 2:                                     [====Compute====]
Transfer:         [--Send L0 KV--]  [--Send L1 KV--]  [--Send L2 KV--]
```

Layer 0 计算完成后立即开始传输，同时 Layer 1 开始计算。

#### 约束条件

为使传输能被计算完全掩盖:
```
transfer_time(Layer_i) ≤ compute_time(Layer_{i+1})
```

由于所有层结构相同（除首末层可能有 embedding/lm_head），简化为:
```
transfer_time ≤ compute_time  (逐层)
```

#### 计算公式

```
KV_data_per_layer = 2 × num_kv_heads × head_dim × seq_len × dtype_bytes  (bytes)

compute_time_per_layer = Layer_FLOPs / (peak_FLOPS × utilization)  (seconds)

min_bandwidth = KV_data_per_layer / compute_time_per_layer  (bytes/s)
             = KV_data_per_layer × peak_FLOPS × utilization / Layer_FLOPs
```

#### 特殊考虑

1. **首层和末层**: 首层计算前没有传输可重叠；末层传输后没有计算可掩盖。总延迟 = pipeline_latency + 首层计算时间 + 末层传输时间。
2. **MLA 模型**: KV_data_per_layer 大幅减小，带宽需求极低。
3. **多 batch**: 若多个请求同时做 prefill→decode transfer，需累加带宽。
4. **网络协议开销**: RDMA 近乎零开销，TCP 需考虑 5-15% 额外开销。

### 1.4 端到端 KV Transfer 延迟

```
不使用流水线:
  total_transfer_time = num_layers × KV_data_per_layer / bandwidth

使用逐层流水线:
  total_transfer_time = compute_time_all_layers + KV_data_per_layer / bandwidth
                      = num_layers × compute_time_per_layer + transfer_time_last_layer
  其中 transfer_time ≤ compute_time 时，传输完全被隐藏

  有效 overhead = transfer_time_last_layer  (仅最后一层的传输不能被计算掩盖)
```

## 2. KV Cache Offload

### 2.1 场景分类

#### 场景 A: GPU → CPU 全量 Offload

所有 KV Cache 都放在 CPU 内存，推理时逐层加载到 GPU:
- **适用**: 单卡显存极度不足，或超长上下文
- **瓶颈**: PCIe 带宽
- **代表**: FlexGen

#### 场景 B: GPU → CPU 部分 Offload

GPU 保留部分层的 KV Cache，其余放在 CPU:
- **适用**: 显存不足但有余量
- **优化**: 选择保留哪些层（靠近当前计算的层）
- **策略**: 双缓冲 — 计算第 i 层时预取第 i+1 层

#### 场景 C: 请求级 Offload

按请求粒度 offload，热请求的 KV 在 GPU，冷请求的 KV 在 CPU:
- **适用**: 多请求并发，部分请求暂时不活跃
- **代表**: vLLM 的 swap 机制
- **触发**: GPU KV Cache 空间不足时 evict 最不活跃的请求

#### 场景 D: GPU → CPU → SSD 三级 Offload

进一步将冷数据从 CPU 卸载到 SSD:
- **适用**: 超大规模并发或超长上下文
- **代表**: FlexGen
- **瓶颈**: SSD 带宽 (~7 GB/s) 远低于 PCIe

### 2.2 带宽计算模型

#### 场景 A: 全量逐层预取

```
# Decode 阶段每步（生成一个 token）需要遍历所有层
# 计算第 i 层时预取第 i+1 层的 KV 到 GPU

KV_per_layer = 2 × num_kv_heads × head_dim × context_len × batch_size × dtype_bytes

# Decode 每层计算时间（memory-bound，实际受显存带宽限制）
# 但我们使用 FLOPs 估算:
decode_compute_per_layer = Layer_FLOPs_decode / (peak_FLOPS × utilization)

# 约束: 预取时间 ≤ 计算时间
min_pcie_bandwidth = KV_per_layer / decode_compute_per_layer

# 注意: Decode 阶段计算量很小，对 PCIe 带宽要求高
# 这是 offload 的核心挑战
```

#### 场景 B: 部分 offload

```
# GPU 保留 k 层，CPU 存储 (L - k) 层
# 计算 GPU 上 k 层时，有时间预取 CPU 上的层

# 可用预取时间 = k 层的计算时间
available_time = k × decode_compute_per_layer

# 需要预取的数据量
data_to_prefetch = (L - k) × KV_per_layer

# 同时需要写回（evict GPU 上的旧层到 CPU）
data_to_evict = (L - k) × KV_per_layer

# 双向总带宽需求（假设可以双向同时传输）
min_pcie_bandwidth = data_to_prefetch / available_time
# 若不能双向同时，需要 × 2
```

#### 场景 C: 请求级 swap

```
# swap 发生在请求调度时，不在层计算的关键路径上
# 带宽需求取决于调度策略

swap_in_data = num_layers × KV_per_layer_per_request × context_len
swap_in_time = swap_in_data / pcie_bandwidth

# 这段时间该请求无法生成，影响 latency
# 通常在处理其他请求时在后台 swap in
```

### 2.3 Decode 阶段的特殊挑战

Decode 阶段是 **memory-bound**:
- 每步只做 1 个 token 的计算，FLOPs 很少
- 但需要读取所有层的 KV Cache + 模型权重
- 计算时间短 → prefetch 窗口小 → 对 PCIe 带宽要求极高

**数值示例 (Llama-3.1-70B, fp16, context=4096, H100)**:
```
KV per layer = 2 × 8 × 128 × 4096 × 2 = 16 MB
Decode FLOPs per layer ≈ 4×8192² + 4×8192×8×128 + 4×64×4096×128 + 6×8192×28672
                       ≈ 0.269G + 0.034G + 0.134G + 1.41G = 1.847 GFLOP
Decode compute time = 1.847G / (989T × 0.5) = 0.00374 ms = 3.74 μs

min_pcie_bandwidth = 16 MB / 3.74 μs = 4,278 GB/s

→ 远超 PCIe Gen5 (128 GB/s)！全量 offload 在 decode 阶段不可行！
```

> **结论**: Decode 阶段全量 offload 对绝大多数模型不可行。实际方案需要:
> 1. 仅 prefill 阶段 offload（prefill 计算量大，有足够时间）
> 2. 请求级 swap（非关键路径）
> 3. 部分 offload + 大量 GPU 缓存

### 2.4 Prefill 阶段的 Offload

反直觉的是，offload 在 prefill 阶段更可行:
```
Prefill FLOPs per layer (s=4096) ≈ 7.566 TFLOP  (Llama-3.1-70B)
Prefill compute time = 7.566T / (989T × 0.5) = 15.3 ms

# 假设需要从 CPU 加载本层的模型权重（offload 权重场景）
# 或加载上一次 prefill 遗留的 KV Cache
KV per layer = 16 MB
load_time = 16 MB / 64 GB/s (PCIe Gen4) = 0.25 ms

15.3 ms >> 0.25 ms → 轻松覆盖
```

### 2.5 开放问题与讨论方向

1. **Decode offload 是否有意义**:
   - 如上分析，全量 offload 不可行
   - 但部分 offload（如只 offload 20% 的层）可能可行
   - 或配合 KV Cache 量化（int4）减少传输量

2. **Offload 粒度选择**:
   - 按层: 简单，但 decode 阶段窗口太小
   - 按请求: 灵活，但 swap 延迟影响用户体验
   - 按 token 段: 如只 offload 早期 token 的 KV（注意力集中在近期 token）

3. **与 PagedAttention 的交互**:
   - vLLM 的 PagedAttention 将 KV Cache 分为固定大小的 block
   - Offload 可以以 block 为粒度，更灵活
   - swap 的单位是 block sequence 而非整个请求

4. **KV Cache 压缩**:
   - 传输前量化: fp16 → int4 可减少 4x 传输量
   - 但增加压缩/解压 overhead
   - CacheGen 提出了专门的 KV Cache 编码方案

5. **建模建议（v0.1）**:
   - 先实现 prefill 阶段的逐层 offload 带宽计算（与 PD 分离类似）
   - 给出 decode 阶段的带宽需求数值，让用户自行判断可行性
   - 提供不同 offload 比例的参数化计算
