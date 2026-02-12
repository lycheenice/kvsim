# KVCache Simulator v0.1 架构设计文档

## 1. 背景与目标

在大模型推理过程中，KV Cache 是影响显存占用和推理性能的关键因素。随着模型规模增长和长上下文需求增加，KV Cache 的显存管理、跨节点传输（PD 分离）、以及 GPU-CPU 卸载（Offload）成为工程中必须量化评估的问题。

本项目旨在构建一个 **KV Cache 模拟器**，提供以下能力：

1. **KV Cache 显存用量估算**：根据模型架构和序列长度，精确计算 KV Cache 显存占用
2. **PD 分离带宽估算**：在 Prefill-Decode 分离架构下，评估逐层异步传输 KV Cache 所需的最低网络带宽
3. **KV Cache Offload 带宽估算**（调研阶段）：评估 GPU → CPU/SSD 卸载场景下的带宽需求
4. **CLI 优先，前后端分离**：v0.1 实现命令行工具，后续版本扩展 Web 界面

---

## 2. 已有方案调研

### 2.1 已有工具与项目

| 项目 | 描述 | 局限性 |
|------|------|--------|
| **vLLM** | PagedAttention 管理 KV Cache，内部有 block 级显存计算 | 耦合在推理引擎中，无法独立使用 |
| **SGLang** | RadixAttention，内部有 KV Cache 显存管理 | 同上，无法独立估算 |
| **LLM-Viewer** | 可视化模型显存分布 | 侧重整体显存，不专注 KV Cache 分析 |
| **HuggingFace Model Memory Calculator** | 估算模型加载显存 | 仅估算权重显存，不计算 KV Cache |
| **DeepSpeed Inference** | 支持 KV Cache offload 到 CPU | 计算逻辑嵌入框架，无独立估算工具 |

**结论**：目前缺少一个独立的、支持多模型架构的 KV Cache 用量与带宽估算工具。

### 2.2 关键论文

| 论文 | 关联 |
|------|------|
| **DistServe** (OSDI'24) | PD 分离架构，prefill 和 decode 在不同 GPU 上执行，KV Cache 通过网络传输 |
| **Splitwise** (ISCA'24) | 类似 PD 分离，分析了 KV transfer 的带宽需求 |
| **Mooncake** (Moonshot AI) | KV Cache 中心化的分离架构，使用 RDMA 传输 KV Cache |
| **FlexGen** (ICML'23) | GPU-CPU-SSD 三级 offload，分析了各层级带宽需求 |
| **InfiniGen** (2024) | 带预测性预取的 KV Cache offload |
| **CacheGen** (2024) | KV Cache 压缩传输，降低带宽需求 |

---

## 3. KV Cache 显存用量估算

### 3.1 核心公式

对于标准 Transformer 架构（MHA / GQA / MQA）：

```
单层单 token 的 KV Cache 大小：
  kv_per_token_per_layer = 2 × num_kv_heads × head_dim × dtype_bytes

总 KV Cache 大小：
  kv_total = num_layers × seq_len × kv_per_token_per_layer
           = num_layers × seq_len × 2 × num_kv_heads × head_dim × dtype_bytes
```

其中：
- `num_kv_heads`：KV 头数（MHA = num_attention_heads，GQA < num_attention_heads，MQA = 1）
- `head_dim`：每个注意力头的维度（通常 = hidden_size / num_attention_heads）
- `dtype_bytes`：数据类型字节数（fp16=2, bf16=2, fp8=1, int8=1, int4=0.5）
- `seq_len`：序列长度（prompt + generated tokens）

### 3.2 特殊架构：MLA（Multi-head Latent Attention）

DeepSeek-V2/V3 使用 MLA，KV Cache 被压缩为低秩潜在表示：

```
单层单 token 的 KV Cache 大小：
  kv_per_token_per_layer = (kv_lora_rank + qk_rope_head_dim) × dtype_bytes

注意：MLA 不存储完整的 K/V，而是存储压缩后的联合潜在向量 c_kv
```

config.json 中对应字段为 `kv_lora_rank` 和 `qk_rope_head_dim`。

### 3.3 HuggingFace config.json 解析

需要从 `config.json` 中提取的关键字段：

| 字段 | 说明 | 备注 |
|------|------|------|
| `model_type` | 模型架构类型 | 用于识别特殊架构（如 deepseek_v2） |
| `num_hidden_layers` | Transformer 层数 | 所有模型通用 |
| `hidden_size` | 隐藏层维度 | 所有模型通用 |
| `num_attention_heads` | Query 头数 | 所有模型通用 |
| `num_key_value_heads` | KV 头数 | GQA 模型此值 < num_attention_heads |
| `head_dim` | 注意力头维度 | 部分模型显式提供，否则 = hidden_size / num_attention_heads |
| `intermediate_size` | FFN 中间维度 | 用于计算 FFN 算力（PD 分离估算需要） |
| `kv_lora_rank` | MLA 压缩维度 | 仅 DeepSeek-V2/V3 |
| `qk_rope_head_dim` | MLA RoPE 维度 | 仅 DeepSeek-V2/V3 |

**不同模型的字段差异处理策略**：
- 优先读取 `num_key_value_heads`，若缺失则回退到 `num_attention_heads`（MHA）
- 优先读取 `head_dim`，若缺失则计算 `hidden_size // num_attention_heads`
- 通过 `model_type` 判断是否为 MLA 架构

### 3.4 内置模型预设

v0.1 内置以下主流模型的预设参数，用户无需手动提供 config.json：

| 模型 | 层数 | hidden_size | Q heads | KV heads | head_dim | FFN size | 注意力类型 |
|------|------|-------------|---------|----------|----------|----------|-----------|
| Llama-3.1-8B | 32 | 4096 | 32 | 8 | 128 | 14336 | GQA |
| Llama-3.1-70B | 80 | 8192 | 64 | 8 | 128 | 28672 | GQA |
| Llama-3.1-405B | 126 | 16384 | 128 | 8 | 128 | 53248 | GQA |
| Qwen2.5-7B | 28 | 3584 | 28 | 4 | 128 | 18944 | GQA |
| Qwen2.5-72B | 80 | 8192 | 64 | 8 | 128 | 29568 | GQA |
| GLM-4-9B | 40 | 4096 | 32 | 2 | 128 | 13696 | GQA |
| DeepSeek-V3 | 61 | 7168 | 128 | - | - | 18432 | MLA |
| Mistral-7B | 32 | 4096 | 32 | 8 | 128 | 14336 | GQA |

> 注：DeepSeek-V3 使用 MLA，kv_lora_rank=512, qk_rope_head_dim=64

---

## 4. PD 分离带宽估算

### 4.1 PD 分离架构概述

在 Prefill-Decode 分离（PD Disaggregation）架构中：
- **Prefill 节点**：处理用户输入的 prompt，逐层计算生成 KV Cache
- **Decode 节点**：接收 KV Cache，逐 token 自回归生成
- **KV Transfer**：Prefill 计算完成后，将 KV Cache 通过网络（通常是 RDMA）传输到 Decode 节点

### 4.2 逐层异步传输模型

逐层流水线传输策略：
```
时间线：
Prefill:  [计算 Layer 0] [计算 Layer 1] [计算 Layer 2] ...
Transfer:            [传输 Layer 0 KV] [传输 Layer 1 KV] ...
```

**约束条件**：Layer i 的 KV 传输必须在 Layer i+1 计算完成前完成，即：

```
transfer_time(layer_i) ≤ compute_time(layer_{i+1})
```

### 4.3 计算时间估算

每层的计算量 FLOPs（Prefill 阶段，序列长度 s）：

**Attention 部分**：
```
Q 投影:    2 × s × hidden_size × num_heads × head_dim
K 投影:    2 × s × hidden_size × num_kv_heads × head_dim
V 投影:    2 × s × hidden_size × num_kv_heads × head_dim
QK^T:      2 × num_heads × s² × head_dim
Score×V:   2 × num_heads × s² × head_dim
Output 投影: 2 × s × num_heads × head_dim × hidden_size

Attn_FLOPs = 2s × h × (n_q + 2 × n_kv) × d_h + 4 × n_q × s² × d_h + 2s × h²
  其中 h = hidden_size, n_q = num_heads, n_kv = num_kv_heads, d_h = head_dim
  简化（当 n_q × d_h = h）: Attn_FLOPs ≈ 4s×h² + 4×n_kv×s×h×d_h + 4×n_q×s²×d_h
```

**FFN 部分**（SwiGLU，3 个矩阵）：
```
Gate 投影:  2 × s × hidden_size × intermediate_size
Up 投影:    2 × s × hidden_size × intermediate_size
Down 投影:  2 × s × intermediate_size × hidden_size

FFN_FLOPs = 6 × s × hidden_size × intermediate_size
```

**每层总 FLOPs**：
```
Layer_FLOPs = Attn_FLOPs + FFN_FLOPs
```

**计算时间**：
```
compute_time_per_layer = Layer_FLOPs / (peak_FLOPS × utilization)
  其中 utilization = 0.5（按需求文档）
```

### 4.4 最低带宽计算

```
每层 KV 数据量:
  KV_data_per_layer = 2 × num_kv_heads × head_dim × seq_len × dtype_bytes

最低网络带宽要求:
  min_bandwidth = KV_data_per_layer / compute_time_per_layer
               = KV_data_per_layer × peak_FLOPS × utilization / Layer_FLOPs
```

### 4.5 GPU 算力参考值

| GPU | FP16 峰值算力 (TFLOPS) | FP8 峰值算力 (TFLOPS) |
|-----|------------------------|----------------------|
| A100 80GB | 312 | - |
| H100 80GB | 989 | 1979 |
| H800 80GB | 989 | 1979 |
| A800 80GB | 312 | - |
| L40S | 362 | 733 |

> 注：实际计算使用 `peak_FLOPS × 0.5` 作为有效算力。

---

## 5. KV Cache Offload 带宽估算（调研）

### 5.1 Offload 场景概述

当单卡 GPU 显存不足以容纳所有活跃请求的 KV Cache 时，可以将部分 KV Cache 卸载到 CPU 内存或 SSD：

```
层级结构:  GPU HBM  ←→  CPU DRAM  ←→  NVMe SSD
带宽:      ~3.35 TB/s     ~64 GB/s      ~7 GB/s
          (HBM3)       (PCIe 5.0 x16)  (NVMe Gen4)
```

### 5.2 现有研究梳理

#### 5.2.1 FlexGen 策略（吞吐导向）
- 将模型权重、KV Cache、激活值分别在 GPU/CPU/SSD 之间调度
- 以最大化吞吐为目标，通过线性规划求解最优 offload 策略
- KV Cache 按 layer 粒度 offload，计算时按需加载
- **带宽模型**：在计算 layer i 时预取 layer i+1 的 KV Cache，要求预取时间 ≤ 计算时间

#### 5.2.2 InfiniGen 策略（延迟敏感）
- Prefill 阶段将 KV Cache 卸载到 CPU
- Decode 时用轻量级模型预测哪些 KV 需要取回 GPU
- 仅取回高注意力分数对应的 KV 子集，减少传输量
- **带宽模型**：传输量 = 选中的 KV 数量 × per-token KV 大小

#### 5.2.3 层级预取策略（通用）
- 与 PD 分离类似的流水线思路：计算 layer i 时预取 layer i+1 的 KV 到 GPU
- 约束：`prefetch_time(layer_{i+1}_KV) ≤ compute_time(layer_i)`
- 这是最基本的 offload 带宽下界

### 5.3 Offload 带宽计算方法（草案）

**场景 A：全量逐层预取**
```
假设所有 KV Cache 都在 CPU 中，推理时逐层预取:
  KV_per_layer = 2 × num_kv_heads × head_dim × context_len × batch_size × dtype_bytes
  prefetch_time ≤ decode_compute_time_per_layer
  min_bandwidth = KV_per_layer / decode_compute_time_per_layer
```

注意 decode 阶段是 **token-by-token**，每步计算量远小于 prefill，因此对带宽需求更高。

**场景 B：部分卸载**
```
假设 GPU 保留 k 层 KV Cache，其余 (num_layers - k) 层在 CPU:
  每步需传输: (num_layers - k) × per_layer_KV
  可用时间: k 层的 GPU 计算时间（这些层无需加载）
  min_bandwidth = [(num_layers - k) × per_layer_KV] / [k × decode_time_per_layer]
```

**场景 C：batch 级调度**
```
当有多个请求时，GPU 显存中保留热请求的 KV Cache，冷请求的卸载到 CPU:
  换入一个请求的 KV: num_layers × per_layer_KV × request_context_len
  可用时间: 处理其他请求的时间
  这是一个调度优化问题
```

### 5.4 待讨论的开放问题

1. **Offload 粒度**：按层 offload？按请求 offload？按 token 段 offload？
2. **预取策略**：全量预取 vs 按需预取 vs 预测性预取，对带宽需求差异很大
3. **Decode vs Prefill offload**：Prefill 阶段计算密集，更容易隐藏传输延迟；Decode 阶段访存密集，带宽压力大
4. **KV Cache 压缩**：是否在传输前进行量化/压缩（如 CacheGen），可大幅降低带宽需求
5. **双向传输**：写回（evict）和读取（prefetch）的带宽是否需要同时考虑

> **建议**：v0.1 先实现场景 A（全量逐层预取）作为基础，场景 B 作为可选参数，场景 C 留待后续版本。

---

## 6. 系统架构设计

### 6.1 整体架构

```
┌──────────────────────────────────────────────────────┐
│                    用户接口层                          │
│  ┌─────────────┐            ┌─────────────────────┐  │
│  │  CLI (v0.1) │            │  Web (v0.2+, 预留)   │  │
│  └──────┬──────┘            └──────────┬──────────┘  │
│         │                              │              │
├─────────┴──────────────────────────────┴──────────────┤
│                    API / 核心接口层                     │
│  ┌────────────────────────────────────────────────┐   │
│  │            SimulatorEngine (核心引擎)            │   │
│  │  - estimate_kv_memory(model, seq_len, dtype)   │   │
│  │  - estimate_pd_bandwidth(model, seq_len, gpu)  │   │
│  │  - estimate_offload_bandwidth(model, ...)      │   │
│  └───────────────────┬────────────────────────────┘   │
│                      │                                 │
├──────────────────────┴─────────────────────────────────┤
│                    计算层                               │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐  │
│  │ KVCacheCalc  │ │ FLOPsCalc    │ │ BandwidthCalc │  │
│  │ - per_layer  │ │ - attn_flops │ │ - pd_transfer │  │
│  │ - total      │ │ - ffn_flops  │ │ - offload     │  │
│  │ - per_token  │ │ - per_layer  │ │ - min_bw      │  │
│  └──────────────┘ └──────────────┘ └───────────────┘  │
│                                                        │
├────────────────────────────────────────────────────────┤
│                    模型层                               │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐  │
│  │ ModelConfig  │ │ GPUConfig    │ │ ConfigParser  │  │
│  │ - presets    │ │ - peak_flops │ │ - from_json   │  │
│  │ - custom     │ │ - hbm_bw     │ │ - from_preset │  │
│  │ - registry   │ │ - pcie_bw    │ │ - validate    │  │
│  └──────────────┘ └──────────────┘ └───────────────┘  │
└────────────────────────────────────────────────────────┘
```

### 6.2 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 语言 | **Python 3.10+** | 生态丰富，AI 领域标准，快速开发 |
| CLI 框架 | **click** 或 **typer** | 类型安全，自动生成帮助文档 |
| 输出格式化 | **rich** | 美观的表格和彩色输出 |
| 配置解析 | **pydantic** | 类型校验，JSON schema 兼容 |
| 未来 Web | **FastAPI** + 前端框架 | 与 pydantic 天然集成，前后端分离 |
| 包管理 | **pyproject.toml** + **uv/pip** | 现代 Python 项目标准 |

### 6.3 目录结构

```
kvsim/
├── pyproject.toml              # 项目配置
├── README.md
├── DESIGN.md                   # 本设计文档
├── DRAFT.md                    # 需求草案
├── src/
│   └── kvsim/
│       ├── __init__.py
│       ├── cli.py              # CLI 入口
│       ├── engine.py           # 核心引擎，组装各计算模块
│       ├── models/
│       │   ├── __init__.py
│       │   ├── config.py       # ModelConfig, GPUConfig 数据模型
│       │   ├── presets.py      # 内置模型预设
│       │   └── parser.py       # HuggingFace config.json 解析器
│       ├── calc/
│       │   ├── __init__.py
│       │   ├── kvcache.py      # KV Cache 显存计算
│       │   ├── flops.py        # FLOPs 计算
│       │   └── bandwidth.py    # 带宽计算（PD + Offload）
│       └── output/
│           ├── __init__.py
│           ├── formatter.py    # 输出格式化（表格、JSON）
│           └── report.py       # 综合报告生成
└── tests/
    ├── test_kvcache.py
    ├── test_flops.py
    ├── test_bandwidth.py
    └── test_parser.py
```

---

## 7. 数据模型

### 7.1 ModelConfig

```python
from pydantic import BaseModel
from typing import Optional, Literal

class ModelConfig(BaseModel):
    name: str
    model_type: str  # "llama", "qwen2", "chatglm", "deepseek_v2", ...
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None  # None → MHA
    head_dim: Optional[int] = None  # None → hidden_size // num_attention_heads
    intermediate_size: int
    attention_type: Literal["mha", "gqa", "mqa", "mla"] = "gqa"

    # MLA 专用字段
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None

    @property
    def effective_kv_heads(self) -> int:
        return self.num_key_value_heads or self.num_attention_heads

    @property
    def effective_head_dim(self) -> int:
        return self.head_dim or (self.hidden_size // self.num_attention_heads)
```

### 7.2 GPUConfig

```python
class GPUConfig(BaseModel):
    name: str
    peak_flops_fp16: float   # TFLOPS
    peak_flops_fp8: Optional[float] = None
    hbm_bandwidth: float     # GB/s
    pcie_bandwidth: float    # GB/s (for offload)
    memory_capacity: float   # GB
```

### 7.3 EstimationResult

```python
class KVCacheEstimation(BaseModel):
    model_name: str
    seq_len: int
    dtype: str
    kv_per_token_per_layer_bytes: int
    kv_per_layer_bytes: int
    kv_total_bytes: int
    kv_total_gb: float

class PDBandwidthEstimation(BaseModel):
    model_name: str
    gpu_name: str
    seq_len: int
    dtype: str
    kv_per_layer_bytes: int
    flops_per_layer: float
    compute_time_per_layer_ms: float
    transfer_time_per_layer_ms: float  # at given bandwidth
    min_bandwidth_gbps: float          # 最低带宽要求 (GB/s)
    can_overlap: bool                  # 是否能完全隐藏延迟
```

---

## 8. CLI 设计

### 8.1 命令结构

```bash
# KV Cache 显存估算
kvsim memory --model llama-3.1-70b --seq-len 4096 --dtype fp16
kvsim memory --config /path/to/config.json --seq-len 4096 --dtype fp16
kvsim memory --model llama-3.1-70b --seq-len 1024,2048,4096,8192,16384 --dtype fp16

# PD 分离带宽估算
kvsim pd-bandwidth --model llama-3.1-70b --seq-len 4096 --gpu h100 --dtype fp16
kvsim pd-bandwidth --config /path/to/config.json --seq-len 4096 --gpu h100 --utilization 0.5

# KV Cache Offload 估算
kvsim offload --model llama-3.1-70b --seq-len 4096 --gpu h100 --pcie-bw 64

# 列出可用预设
kvsim list-models
kvsim list-gpus
```

### 8.2 输出示例

```
$ kvsim memory --model llama-3.1-70b --seq-len 4096 --dtype fp16

╭──────────────────────────────────────────────────────────╮
│            KV Cache Memory Estimation                    │
├──────────────────────────────────────────────────────────┤
│  Model:           Llama-3.1-70B                          │
│  Attention Type:  GQA (8 KV heads)                       │
│  Layers:          80                                     │
│  Sequence Length:  4,096                                  │
│  Data Type:       fp16 (2 bytes)                         │
├──────────────────────────────────────────────────────────┤
│  Per Token Per Layer:   2 × 8 × 128 × 2 = 4,096 B      │
│  Per Layer (4096 tok):  4,096 × 4,096 = 16.0 MB         │
│  Total (80 layers):     80 × 16.0 MB = 1,280.0 MB       │
│                         = 1.25 GB                        │
╰──────────────────────────────────────────────────────────╯

$ kvsim memory --model llama-3.1-70b --seq-len 1024,4096,16384,131072 --dtype fp16

┌─────────────┬───────────┬────────────┬───────────┐
│  Seq Length  │  Per Layer │   Total    │  % of 80G │
├─────────────┼───────────┼────────────┼───────────┤
│      1,024  │    4.0 MB │   320.0 MB │    0.39%  │
│      4,096  │   16.0 MB │  1,280.0 MB│    1.56%  │
│     16,384  │   64.0 MB │  5,120.0 MB│    6.25%  │
│    131,072  │  512.0 MB │ 40,960.0 MB│   50.00%  │
└─────────────┴───────────┴────────────┴───────────┘
```

---

## 9. 关键实现细节

### 9.1 config.json 自动解析

```python
def parse_hf_config(config_path: str) -> ModelConfig:
    """从 HuggingFace config.json 自动解析模型配置"""
    with open(config_path) as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "unknown")

    # 处理不同模型的字段名差异
    num_kv_heads = (
        cfg.get("num_key_value_heads")
        or cfg.get("multi_query_group_num")  # ChatGLM 系列
        or cfg.get("num_attention_heads")     # 回退到 MHA
    )

    # 判断注意力类型
    if model_type in ("deepseek_v2", "deepseek_v3"):
        attention_type = "mla"
    elif num_kv_heads == 1:
        attention_type = "mqa"
    elif num_kv_heads < cfg["num_attention_heads"]:
        attention_type = "gqa"
    else:
        attention_type = "mha"

    return ModelConfig(
        name=cfg.get("_name_or_path", "custom"),
        model_type=model_type,
        num_hidden_layers=cfg["num_hidden_layers"],
        hidden_size=cfg["hidden_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=num_kv_heads,
        head_dim=cfg.get("head_dim"),
        intermediate_size=cfg["intermediate_size"],
        attention_type=attention_type,
        kv_lora_rank=cfg.get("kv_lora_rank"),
        qk_rope_head_dim=cfg.get("qk_rope_head_dim"),
    )
```

### 9.2 Tensor Parallelism 支持

当使用张量并行时，KV Cache 按 KV head 切分到不同 GPU：

```
单卡 KV Cache = kv_total / tp_size  (当 num_kv_heads >= tp_size)
```

CLI 中通过 `--tp` 参数指定：
```bash
kvsim memory --model llama-3.1-70b --seq-len 4096 --tp 8
```

---

## 10. v0.1 交付范围与后续规划

### v0.1 交付内容

- [x] KV Cache 显存估算（单请求）
- [x] 支持 MHA / GQA / MQA / MLA 四种注意力类型
- [x] HuggingFace config.json 自动解析
- [x] 8+ 主流模型内置预设
- [x] PD 分离逐层异步传输最低带宽估算
- [x] CLI 命令行工具
- [x] KV Cache Offload 带宽估算（场景 A：全量逐层预取）
- [x] 单元测试

### v0.2 规划

- [ ] Web 前端界面（FastAPI + React/Vue）
- [ ] 批量请求场景估算（concurrent requests）
- [ ] KV Cache Offload 场景 B/C
- [ ] KV Cache 压缩量化对带宽的影响分析
- [ ] 可视化图表（显存随 seq_len 变化的曲线等）

### v0.3+ 展望

- [ ] 端到端推理性能模拟（latency modeling）
- [ ] 多机多卡部署方案推荐
- [ ] 与 vLLM/SGLang 配置参数联动
- [ ] 成本估算

---

## 附录 A：KV Cache 速算参考

**常见模型单请求 KV Cache（fp16，序列长度 4096）**：

| 模型 | KV heads | 单层 | 总计 |
|------|---------|------|------|
| Llama-3.1-8B | 8 | 16 MB | 512 MB |
| Llama-3.1-70B | 8 | 16 MB | 1.25 GB |
| Llama-3.1-405B | 8 | 16 MB | 1.97 GB |
| Qwen2.5-72B | 8 | 16 MB | 1.25 GB |
| DeepSeek-V3 (MLA) | - | 4.5 MB | 274 MB |
