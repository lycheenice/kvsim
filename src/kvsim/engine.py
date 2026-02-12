"""核心引擎 — 组装各计算模块，提供统一的 API。

engine.py 为纯计算层，不依赖任何 IO / CLI / Web，方便前后端分离调用。
"""

from __future__ import annotations

from kvsim.calc.bandwidth import estimate_offload_bandwidth, estimate_pd_bandwidth
from kvsim.calc.kvcache import estimate_kv_memory
from kvsim.models.config import (
    GPUConfig,
    KVCacheEstimation,
    ModelConfig,
    OffloadBandwidthEstimation,
    PDBandwidthEstimation,
)
from kvsim.models.parser import parse_hf_config
from kvsim.models.presets import get_gpu, get_model


def resolve_model(model_name: str | None, config_path: str | None) -> ModelConfig:
    """解析模型来源: 预设名称或 config.json 路径。"""
    if config_path:
        return parse_hf_config(config_path)
    if model_name:
        return get_model(model_name)
    raise ValueError("必须指定 --model 或 --config")


def resolve_gpu(gpu_name: str) -> GPUConfig:
    """解析 GPU 预设名称。"""
    return get_gpu(gpu_name)


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------

def compute_kv_memory(
    model: ModelConfig,
    seq_lens: list[int],
    dtype: str = "fp16",
    tp_size: int = 1,
) -> list[KVCacheEstimation]:
    """批量计算多个序列长度的 KV Cache 显存。"""
    return [
        estimate_kv_memory(model, sl, dtype=dtype, tp_size=tp_size)
        for sl in seq_lens
    ]


def compute_pd_bandwidth(
    model: ModelConfig,
    gpu: GPUConfig,
    seq_lens: list[int],
    dtype: str = "fp16",
    utilization: float = 0.5,
    network_bandwidth_gbps: float | None = None,
) -> list[PDBandwidthEstimation]:
    """批量计算多个序列长度的 PD 分离带宽需求。"""
    return [
        estimate_pd_bandwidth(
            model, gpu, sl,
            dtype=dtype,
            utilization=utilization,
            network_bandwidth_gbps=network_bandwidth_gbps,
        )
        for sl in seq_lens
    ]


def compute_offload_bandwidth(
    model: ModelConfig,
    gpu: GPUConfig,
    seq_len: int,
    context_len: int | None = None,
    dtype: str = "fp16",
    utilization: float = 0.5,
    phase: str = "prefill",
) -> OffloadBandwidthEstimation:
    """计算 KV Cache Offload 带宽需求。"""
    return estimate_offload_bandwidth(
        model, gpu, seq_len,
        context_len=context_len,
        dtype=dtype,
        utilization=utilization,
        phase=phase,
    )
