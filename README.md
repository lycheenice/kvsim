# kvsim — KV Cache Simulator

LLM 推理 KV Cache 显存用量与带宽估算工具。

支持 KV Cache **显存占用估算**、**PD 分离（Prefill-Decode Disaggregation）传输带宽估算**、以及 **KV Cache Offload（GPU→CPU）带宽估算**，涵盖 MHA / GQA / MQA / MLA 四种注意力架构。

## 功能特性

- **KV Cache 显存估算** — 根据模型架构和序列长度精确计算 KV Cache 显存占用，支持多种量化精度（fp16/bf16/fp8/int8/int4）和 Tensor Parallelism
- **PD 分离带宽估算** — 在 Prefill-Decode 分离架构下，计算逐层异步传输 KV Cache 的最低网络带宽，判断计算能否掩盖传输延迟
- **KV Cache Offload 带宽估算** — 计算 GPU→CPU 全量逐层预取场景下的最低 PCIe 带宽需求，分别评估 Prefill 和 Decode 阶段的可行性
- **自定义模型** — 自动解析 HuggingFace `config.json`，兼容 Llama / Qwen / ChatGLM / Mistral / DeepSeek 等模型家族的字段差异
- **内置预设** — 8 个主流模型（Llama-3.1、Qwen2.5、GLM-4、DeepSeek-V3、Mistral）+ 5 款 GPU（H100/H800/A100/A800/L40S）开箱即用

## 安装

### 从源码安装

```bash
git clone <repo-url> kvsim
cd kvsim
pip install -e .
```

### 安装开发依赖（含 pytest）

```bash
pip install -e ".[dev]"
```

安装后即可使用 `kvsim` 命令，或通过 `python -m kvsim.cli` 调用。

## 快速开始

### 1. 估算 KV Cache 显存

```bash
# 单个序列长度
kvsim memory --model llama-3.1-70b --seq-len 4096 --dtype fp16

# 多个序列长度对比，并显示 GPU 显存占比
kvsim memory --model llama-3.1-70b --seq-len 1024,4096,16384,131072 --gpu-mem 80
```

输出示例：

```
╭───────────────────────── KV Cache Memory Estimation ─────────────────────────╮
│   Model:             Llama-3.1-70B                                           │
│   Attention Type:    GQA (8 KV heads)                                        │
│   Sequence Length:   4,096                                                   │
│   Data Type:         fp16                                                    │
│                                                                              │
│   Per Token/Layer:   4.0 KB                                                  │
│   Per Layer:         16.0 MB                                                 │
│   Total:             1.25 GB                                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```

多序列长度表格：

```
     KV Cache Memory — Llama-3.1-70B (fp16)
┏━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Seq Length ┃ Per Layer ┃    Total ┃ % of 80G ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│      1,024 │    4.0 MB │ 320.0 MB │    0.39% │
│      4,096 │   16.0 MB │  1.25 GB │    1.56% │
│     16,384 │   64.0 MB │  5.00 GB │    6.25% │
│    131,072 │  512.0 MB │ 40.00 GB │   50.00% │
└────────────┴───────────┴──────────┴──────────┘
```

### 2. 估算 PD 分离带宽

```bash
# 基本用法
kvsim pd-bandwidth --model llama-3.1-70b --seq-len 4096 --gpu h100

# 指定网络带宽，判断能否掩盖传输延迟
kvsim pd-bandwidth --model llama-3.1-70b --seq-len 4096 --gpu h100 --network-bw 25

# 多序列长度对比
kvsim pd-bandwidth --model llama-3.1-70b --seq-len 1024,4096,16384,131072 --gpu h100
```

输出示例：

```
╭─────────────────── PD Disaggregation Bandwidth Estimation ───────────────────╮
│   Model:                Llama-3.1-70B                                        │
│   GPU (Prefill):        H100 SXM 80GB                                        │
│   Sequence Length:      4,096                                                │
│   Data Type:            fp16                                                 │
│   Utilization:          50%                                                  │
│                                                                              │
│   KV per Layer:         16.0 MB                                              │
│   Attn FLOPs/Layer:     1.787 TFLOP                                          │
│   FFN  FLOPs/Layer:     5.772 TFLOP                                          │
│   Total FLOPs/Layer:    7.559 TFLOP                                          │
│                                                                              │
│   Compute Time/Layer:   15.280 ms                                            │
│   Min Bandwidth:        1.10 GB/s                                            │
│   Network Bandwidth:    25.0 GB/s                                            │
│   Transfer Time/Layer:  0.671 ms                                             │
│   Can Fully Overlap:    YES                                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

多序列长度表格：

```
              PD Disaggregation Bandwidth — Llama-3.1-70B
┏━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Seq Length ┃ KV/Layer ┃   FLOPs/Layer ┃ Compute (ms) ┃ Min BW (GB/s) ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│      1,024 │   4.0 MB │   1.787 TFLOP │        3.612 │          1.16 │
│      4,096 │  16.0 MB │   7.559 TFLOP │       15.280 │          1.10 │
│     16,384 │  64.0 MB │  36.834 TFLOP │       74.457 │          0.90 │
│    131,072 │ 512.0 MB │ 787.250 TFLOP │     1591.369 │          0.34 │
└────────────┴──────────┴───────────────┴──────────────┴───────────────┘
```

### 3. 估算 KV Cache Offload 带宽

```bash
# Prefill 阶段（通常可行）
kvsim offload --model llama-3.1-70b --seq-len 4096 --gpu h100 --phase prefill

# Decode 阶段（通常不可行 — 计算量太少，无法掩盖传输延迟）
kvsim offload --model llama-3.1-70b --seq-len 4096 --gpu h100 --phase decode --context-len 4096
```

输出示例（Decode 阶段）：

```
╭─────────────────── KV Cache Offload Bandwidth Estimation ────────────────────╮
│   Model:                Llama-3.1-70B                                        │
│   GPU:                  H100 SXM 80GB                                        │
│   Phase:                decode                                               │
│   Sequence Length:      4,096                                                │
│   Context Length:       4,096                                                │
│   Data Type:            fp16                                                 │
│   Utilization:          50%                                                  │
│                                                                              │
│   KV per Layer:         16.0 MB                                              │
│   FLOPs per Layer:      1.85 GFLOP                                           │
│   Compute Time/Layer:   0.0037 ms                                            │
│                                                                              │
│   Min PCIe BW Needed:   4497.27 GB/s                                         │
│   Actual PCIe BW:       128.0 GB/s                                           │
│   Feasible:             NO                                                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```

> Decode 阶段全量 Offload 需要 ~4497 GB/s，远超 PCIe Gen5 的 128 GB/s，因此不可行。这一结论与学术界的分析一致——Decode 阶段是 memory-bound，计算窗口极小，无法通过逐层预取隐藏传输延迟。

### 4. 使用自定义模型

通过 `--config` 指定 HuggingFace `config.json` 路径，自动解析模型架构参数：

```bash
kvsim memory --config /path/to/config.json --seq-len 4096
kvsim pd-bandwidth --config /path/to/config.json --seq-len 4096 --gpu h100
```

支持的 `config.json` 字段自动兼容：
- 标准字段：`num_key_value_heads`、`head_dim`
- ChatGLM 变体：`multi_query_group_num`
- DeepSeek MLA：`kv_lora_rank`、`qk_rope_head_dim`

### 5. 查看内置预设

```bash
kvsim list-models    # 列出所有内置模型预设
kvsim list-gpus      # 列出所有内置 GPU 预设
```

## 命令参考

| 命令 | 说明 |
|------|------|
| `kvsim memory` | KV Cache 显存估算 |
| `kvsim pd-bandwidth` | PD 分离带宽估算 |
| `kvsim offload` | KV Cache Offload 带宽估算 |
| `kvsim list-models` | 列出内置模型预设 |
| `kvsim list-gpus` | 列出内置 GPU 预设 |

### 通用选项

| 选项 | 简写 | 说明 |
|------|------|------|
| `--model` | `-m` | 内置模型预设名称 |
| `--config` | `-c` | HuggingFace config.json 路径（与 `--model` 二选一） |
| `--seq-len` | `-s` | 序列长度，部分命令支持逗号分隔多个值 |
| `--dtype` | `-d` | 数据类型：fp16 / bf16 / fp8 / int8 / int4（默认 fp16） |
| `--gpu` | `-g` | GPU 预设名称（默认 h100） |
| `--utilization` | `-u` | 算力利用率 0~1（默认 0.5） |

## 内置预设

### 模型

| Key | 模型 | 注意力类型 | 层数 | KV Heads |
|-----|------|-----------|------|----------|
| `llama-3.1-8b` | Llama-3.1-8B | GQA | 32 | 8 |
| `llama-3.1-70b` | Llama-3.1-70B | GQA | 80 | 8 |
| `llama-3.1-405b` | Llama-3.1-405B | GQA | 126 | 8 |
| `qwen2.5-7b` | Qwen2.5-7B | GQA | 28 | 4 |
| `qwen2.5-72b` | Qwen2.5-72B | GQA | 80 | 8 |
| `glm-4-9b` | GLM-4-9B | GQA | 40 | 2 |
| `deepseek-v3` | DeepSeek-V3 | MLA | 61 | - |
| `mistral-7b` | Mistral-7B | GQA | 32 | 8 |

### GPU

| Key | GPU | FP16 (TFLOPS) | PCIe BW | 显存 |
|-----|-----|---------------|---------|------|
| `h100` | H100 SXM 80GB | 989.4 | 128 GB/s | 80 GB |
| `h800` | H800 SXM 80GB | 989.4 | 128 GB/s | 80 GB |
| `a100` | A100 SXM 80GB | 312.0 | 64 GB/s | 80 GB |
| `a800` | A800 SXM 80GB | 312.0 | 64 GB/s | 80 GB |
| `l40s` | L40S 48GB | 362.1 | 64 GB/s | 48 GB |

## 核心公式

### KV Cache 显存

**标准 Transformer (MHA/GQA/MQA)**:

```
kv_per_token_per_layer = 2 x num_kv_heads x head_dim x dtype_bytes
kv_total = num_layers x seq_len x kv_per_token_per_layer
```

**MLA (DeepSeek-V2/V3)**:

```
kv_per_token_per_layer = (kv_lora_rank + qk_rope_head_dim) x dtype_bytes
```

### PD 分离最低带宽

逐层流水线传输，要求传输时间不超过下一层计算时间：

```
min_bandwidth = kv_per_layer / compute_time_per_layer
compute_time  = layer_FLOPs / (peak_FLOPS x utilization)
```

### Offload 带宽

与 PD 分离类似，但使用 PCIe 带宽替代网络带宽，Decode 阶段使用 Decode FLOPs：

```
min_pcie_bw = kv_per_layer / decode_compute_time_per_layer
```

## 项目结构

```
src/kvsim/
  cli.py              CLI 入口 (typer)
  engine.py           核心引擎 (纯计算 API, 不依赖 IO)
  models/
    config.py          数据模型 (ModelConfig, GPUConfig, Result)
    presets.py          内置模型和 GPU 预设
    parser.py           HuggingFace config.json 解析器
  calc/
    kvcache.py          KV Cache 显存计算
    flops.py            FLOPs 计算 (Prefill + Decode)
    bandwidth.py        PD 分离 + Offload 带宽计算
  output/
    formatter.py        rich 格式化输出
tests/                  单元测试 (30 个)
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v
```

## 后续规划

- **v0.2**: Web 前端界面 (FastAPI + 前端框架)、批量请求场景估算、可视化图表
- **v0.3+**: 端到端推理性能模拟、多机多卡部署方案推荐、与 vLLM/SGLang 配置联动

## License

TBD
