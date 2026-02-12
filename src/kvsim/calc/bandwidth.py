"""PD 分离带宽估算和 KV Cache Offload 带宽估算。

PD 分离 — 逐层异步传输:
  约束: transfer_time(layer_i) ≤ compute_time(layer_{i+1})
  min_bandwidth = kv_per_layer / compute_time_per_layer
               = kv_per_layer × effective_flops / layer_flops

Offload — 全量逐层预取 (场景 A):
  Prefill: min_pcie_bw = kv_per_layer / prefill_compute_time_per_layer
  Decode:  min_pcie_bw = kv_per_layer / decode_compute_time_per_layer

参考: DESIGN.md 第 4、5 节
"""

from __future__ import annotations

from kvsim.calc.flops import (
    total_flops_per_layer,
    total_flops_per_layer_decode,
)
from kvsim.calc.kvcache import kv_per_token_per_layer_bytes
from kvsim.models.config import (
    DTYPE_BYTES,
    GPUConfig,
    ModelConfig,
    OffloadBandwidthEstimation,
    PDBandwidthEstimation,
)


def _effective_flops(gpu: GPUConfig, dtype: str, utilization: float) -> float:
    """有效算力 (FLOPS), 返回单位为 FLOP/s。

    peak_flops 单位是 TFLOPS = 10^12 FLOPS
    """
    if dtype == "fp8" and gpu.peak_flops_fp8 is not None:
        peak = gpu.peak_flops_fp8
    else:
        peak = gpu.peak_flops_fp16
    return peak * 1e12 * utilization


def estimate_pd_bandwidth(
    model: ModelConfig,
    gpu: GPUConfig,
    seq_len: int,
    dtype: str = "fp16",
    utilization: float = 0.5,
    network_bandwidth_gbps: float | None = None,
) -> PDBandwidthEstimation:
    """PD 分离逐层异步传输的最低带宽估算。

    Args:
        model: 模型配置
        gpu: GPU 配置 (Prefill 节点)
        seq_len: 序列长度
        dtype: KV Cache 数据类型
        utilization: 算力利用率, 默认 0.5
        network_bandwidth_gbps: 用户指定的网络带宽 (GB/s), 用于判断 can_overlap
    """
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"不支持的 dtype: {dtype}")

    from kvsim.calc.flops import attn_flops_per_layer, ffn_flops_per_layer

    # 1) KV 数据量 (bytes per layer)
    kv_bytes = kv_per_token_per_layer_bytes(model, dtype) * seq_len

    # 2) 每层 FLOPs
    attn_f = attn_flops_per_layer(model, seq_len)
    ffn_f = ffn_flops_per_layer(model, seq_len)
    layer_f = attn_f + ffn_f

    # 3) 计算时间
    eff_flops = _effective_flops(gpu, dtype, utilization)
    compute_time_s = layer_f / eff_flops  # 秒
    compute_time_ms = compute_time_s * 1000.0

    # 4) 最低带宽 (GB/s)
    min_bw = kv_bytes / compute_time_s / 1e9  # bytes/s → GB/s

    # 5) 在指定带宽下的传输时间
    if network_bandwidth_gbps is not None and network_bandwidth_gbps > 0:
        transfer_time_s = kv_bytes / (network_bandwidth_gbps * 1e9)
        transfer_time_ms = transfer_time_s * 1000.0
        can_overlap = transfer_time_ms <= compute_time_ms
    else:
        # 用最低带宽算传输时间，即恰好等于计算时间
        transfer_time_ms = compute_time_ms
        can_overlap = None

    return PDBandwidthEstimation(
        model_name=model.name,
        gpu_name=gpu.name,
        seq_len=seq_len,
        dtype=dtype,
        utilization=utilization,
        kv_per_layer_bytes=kv_bytes,
        attn_flops_per_layer=attn_f,
        ffn_flops_per_layer=ffn_f,
        total_flops_per_layer=layer_f,
        compute_time_per_layer_ms=compute_time_ms,
        transfer_time_per_layer_ms=transfer_time_ms,
        min_bandwidth_gbps=min_bw,
        network_bandwidth_gbps=network_bandwidth_gbps,
        can_overlap=can_overlap,
    )


def estimate_offload_bandwidth(
    model: ModelConfig,
    gpu: GPUConfig,
    seq_len: int,
    context_len: int | None = None,
    dtype: str = "fp16",
    utilization: float = 0.5,
    phase: str = "prefill",
) -> OffloadBandwidthEstimation:
    """KV Cache Offload 带宽估算（场景 A: 全量逐层预取）。

    计算在 prefill 或 decode 阶段，逐层从 CPU 预取 KV Cache 到 GPU
    所需的最低 PCIe 带宽。

    Args:
        model: 模型配置
        gpu: GPU 配置
        seq_len: 序列长度 (prefill 阶段) 或不使用 (decode 阶段)
        context_len: decode 阶段的上下文长度, prefill 时可不指定
        dtype: KV Cache 数据类型
        utilization: 算力利用率
        phase: "prefill" 或 "decode"
    """
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"不支持的 dtype: {dtype}")
    if phase not in ("prefill", "decode"):
        raise ValueError(f"phase 必须为 'prefill' 或 'decode', 收到: {phase}")

    if phase == "decode":
        if context_len is None:
            raise ValueError("decode 阶段必须指定 context_len")
        # Decode: KV Cache 大小按 context_len 计算
        kv_bytes = kv_per_token_per_layer_bytes(model, dtype) * context_len
        layer_flops = total_flops_per_layer_decode(model, context_len)
    else:
        # Prefill: 使用 seq_len
        if context_len is None:
            context_len = seq_len
        kv_bytes = kv_per_token_per_layer_bytes(model, dtype) * seq_len
        layer_flops = total_flops_per_layer(model, seq_len)

    eff_flops = _effective_flops(gpu, dtype, utilization)
    compute_time_s = layer_flops / eff_flops
    compute_time_ms = compute_time_s * 1000.0

    min_pcie_bw = kv_bytes / compute_time_s / 1e9  # GB/s

    return OffloadBandwidthEstimation(
        model_name=model.name,
        gpu_name=gpu.name,
        seq_len=seq_len,
        context_len=context_len,
        dtype=dtype,
        utilization=utilization,
        phase=phase,
        kv_per_layer_bytes=kv_bytes,
        flops_per_layer=layer_flops,
        compute_time_per_layer_ms=compute_time_ms,
        min_pcie_bandwidth_gbps=min_pcie_bw,
        actual_pcie_bandwidth_gbps=gpu.pcie_bandwidth,
        feasible=min_pcie_bw <= gpu.pcie_bandwidth,
    )
