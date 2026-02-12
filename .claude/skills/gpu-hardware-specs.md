# GPU 硬件规格参考

本文件记录主流推理 GPU 的计算能力、显存带宽、PCIe 带宽等关键参数，用于 PD 分离和 Offload 带宽估算。

## NVIDIA 数据中心 GPU

### H100 / H800 系列

| 参数 | H100 SXM | H100 PCIe | H800 SXM | H800 PCIe |
|------|----------|-----------|----------|-----------|
| 显存容量 | 80 GB HBM3 | 80 GB HBM3 | 80 GB HBM3 | 80 GB HBM3 |
| 显存带宽 | 3.35 TB/s | 2.0 TB/s | 3.35 TB/s | 2.0 TB/s |
| FP16 Tensor Core | 989.4 TFLOPS | 756 TFLOPS | 989.4 TFLOPS | 756 TFLOPS |
| BF16 Tensor Core | 989.4 TFLOPS | 756 TFLOPS | 989.4 TFLOPS | 756 TFLOPS |
| FP8 Tensor Core | 1978.9 TFLOPS | 1513 TFLOPS | 1978.9 TFLOPS | 1513 TFLOPS |
| NVLink 带宽 | 900 GB/s | - | 400 GB/s | - |
| PCIe | Gen5 x16 (128 GB/s) | Gen5 x16 (128 GB/s) | Gen5 x16 (128 GB/s) | Gen5 x16 (128 GB/s) |
| TDP | 700W | 350W | 700W | 350W |

> 注: H800 是 H100 的中国特供版本，NVLink 带宽从 900 降到 400 GB/s，计算能力相同。

### A100 / A800 系列

| 参数 | A100 SXM | A100 PCIe | A800 SXM | A800 PCIe |
|------|----------|-----------|----------|-----------|
| 显存容量 | 80 GB HBM2e | 80 GB HBM2e | 80 GB HBM2e | 80 GB HBM2e |
| 显存带宽 | 2.0 TB/s | 2.0 TB/s | 2.0 TB/s | 2.0 TB/s |
| FP16 Tensor Core | 312 TFLOPS | 312 TFLOPS | 312 TFLOPS | 312 TFLOPS |
| BF16 Tensor Core | 312 TFLOPS | 312 TFLOPS | 312 TFLOPS | 312 TFLOPS |
| TF32 Tensor Core | 156 TFLOPS | 156 TFLOPS | 156 TFLOPS | 156 TFLOPS |
| NVLink 带宽 | 600 GB/s | - | 400 GB/s | - |
| PCIe | Gen4 x16 (64 GB/s) | Gen4 x16 (64 GB/s) | Gen4 x16 (64 GB/s) | Gen4 x16 (64 GB/s) |
| TDP | 400W | 300W | 400W | 300W |

### L40S / L40

| 参数 | L40S | L40 |
|------|------|-----|
| 显存容量 | 48 GB GDDR6 | 48 GB GDDR6 |
| 显存带宽 | 864 GB/s | 864 GB/s |
| FP16 Tensor Core | 362.05 TFLOPS | 181.05 TFLOPS |
| BF16 Tensor Core | 362.05 TFLOPS | 181.05 TFLOPS |
| FP8 Tensor Core | 733 TFLOPS | 362 TFLOPS |
| PCIe | Gen4 x16 (64 GB/s) | Gen4 x16 (64 GB/s) |
| TDP | 350W | 300W |

### 其他常见 GPU

| GPU | 显存 | 显存带宽 | FP16 TFLOPS | PCIe |
|-----|------|---------|-------------|------|
| RTX 4090 | 24 GB GDDR6X | 1.01 TB/s | 82.6 | Gen4 x16 |
| RTX 3090 | 24 GB GDDR6X | 936 GB/s | 35.6 | Gen4 x16 |
| V100 SXM | 32 GB HBM2 | 900 GB/s | 125 | Gen3 x16 |
| T4 | 16 GB GDDR6 | 320 GB/s | 65 | Gen3 x16 |

## 互联带宽

### 节点内（用于 Tensor Parallelism）

| 互联 | 单向带宽 | 双向带宽 | 适用场景 |
|------|---------|---------|---------|
| NVLink 4.0 (H100) | 450 GB/s | 900 GB/s | 同节点 GPU 间 |
| NVLink 3.0 (A100) | 300 GB/s | 600 GB/s | 同节点 GPU 间 |
| NVLink (H800) | 200 GB/s | 400 GB/s | 同节点 GPU 间 |
| PCIe Gen5 x16 | 64 GB/s | 128 GB/s | GPU-CPU |
| PCIe Gen4 x16 | 32 GB/s | 64 GB/s | GPU-CPU |

### 节点间（用于 PD 分离 KV Transfer）

| 互联 | 单向带宽 | 说明 |
|------|---------|------|
| InfiniBand NDR | 50 GB/s (400 Gbps) | 单端口 |
| InfiniBand HDR | 25 GB/s (200 Gbps) | 单端口 |
| RoCE v2 (100GbE) | 12.5 GB/s | 单端口 |
| RoCE v2 (200GbE) | 25 GB/s | 单端口 |
| TCP/IP (100GbE) | ~11 GB/s | 有协议开销 |
| TCP/IP (25GbE) | ~2.8 GB/s | 有协议开销 |

> PD 分离场景中 KV Cache 传输通常使用 RDMA (InfiniBand / RoCE)，延迟更低、CPU 开销更小。

## 存储带宽（用于 Offload 到 SSD）

| 存储类型 | 顺序读带宽 | 顺序写带宽 | 说明 |
|---------|-----------|-----------|------|
| NVMe Gen4 x4 | ~7 GB/s | ~5 GB/s | 单盘 |
| NVMe Gen5 x4 | ~14 GB/s | ~12 GB/s | 单盘 |
| NVMe RAID 0 (4盘) | ~28 GB/s | ~20 GB/s | 4 × Gen4 |

## CPU DRAM 带宽（用于 Offload 到 CPU 内存）

| 平台 | 内存带宽 | 说明 |
|------|---------|------|
| DDR5-4800 (8通道) | ~307 GB/s | 服务器典型配置 |
| DDR4-3200 (8通道) | ~205 GB/s | 上一代服务器 |

> 注: CPU 内存带宽通常不是瓶颈，PCIe 带宽是 GPU-CPU offload 的实际瓶颈。

## 在本项目中的使用

计算时取 `peak_flops × utilization` 作为有效算力，默认 utilization = 0.5:
```
effective_flops = gpu.peak_flops_fp16 × 0.5
compute_time = flops_per_layer / effective_flops
```

带宽估算时使用的是链路带宽（PCIe / NVLink / InfiniBand），不是显存带宽:
- PD 分离: 使用节点间网络带宽（InfiniBand / RoCE）
- Offload: 使用 PCIe 带宽（GPU ↔ CPU）

## 需要验证的事项

> 1. H100/H800 的 FP8 TFLOPS 在 sparsity 和 non-sparsity 模式下不同，上表为 non-sparsity
> 2. 实际 PCIe 带宽受 NUMA 拓扑、DMA 引擎效率的影响，通常能达到理论值的 85-95%
> 3. NVLink 带宽为所有 link 聚合值，单 link 带宽需查具体拓扑
