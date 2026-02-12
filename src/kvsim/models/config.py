"""数据模型定义：ModelConfig, GPUConfig, 以及各类估算结果。"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, model_validator


# ---------------------------------------------------------------------------
# dtype 字节数映射
# ---------------------------------------------------------------------------

DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}


# ---------------------------------------------------------------------------
# ModelConfig — 描述一个 Transformer 模型的 KV Cache 相关参数
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Transformer 模型配置，包含计算 KV Cache 和 FLOPs 所需的全部参数。"""

    name: str
    model_type: str  # "llama", "qwen2", "chatglm", "deepseek_v2" 等

    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None  # None 则视为 MHA
    head_dim: Optional[int] = None  # None 则由 hidden_size // num_attention_heads 推导
    intermediate_size: int

    attention_type: Literal["mha", "gqa", "mqa", "mla"] = "gqa"

    # MLA 专用 (DeepSeek-V2/V3)
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None

    @model_validator(mode="after")
    def _fill_defaults(self) -> "ModelConfig":
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        return self

    @property
    def effective_kv_heads(self) -> int:
        """实际 KV 头数（已在 validator 中填充）。"""
        assert self.num_key_value_heads is not None
        return self.num_key_value_heads

    @property
    def effective_head_dim(self) -> int:
        """实际 head 维度（已在 validator 中填充）。"""
        assert self.head_dim is not None
        return self.head_dim


# ---------------------------------------------------------------------------
# GPUConfig — GPU 硬件参数
# ---------------------------------------------------------------------------

class GPUConfig(BaseModel):
    """GPU 硬件参数，用于算力和带宽估算。"""

    name: str
    peak_flops_fp16: float  # TFLOPS
    peak_flops_fp8: Optional[float] = None  # TFLOPS
    hbm_bandwidth: float  # GB/s
    pcie_bandwidth: float  # GB/s（GPU ↔ CPU）
    memory_capacity: float  # GB


# ---------------------------------------------------------------------------
# 估算结果
# ---------------------------------------------------------------------------

class KVCacheEstimation(BaseModel):
    """KV Cache 显存估算结果。"""

    model_name: str
    seq_len: int
    dtype: str
    tp_size: int = 1

    kv_per_token_per_layer_bytes: float
    kv_per_layer_bytes: float
    kv_total_bytes: float
    kv_total_gb: float


class PDBandwidthEstimation(BaseModel):
    """PD 分离带宽估算结果。"""

    model_name: str
    gpu_name: str
    seq_len: int
    dtype: str
    utilization: float

    kv_per_layer_bytes: float
    attn_flops_per_layer: float
    ffn_flops_per_layer: float
    total_flops_per_layer: float
    compute_time_per_layer_ms: float
    transfer_time_per_layer_ms: float  # 在给定带宽下
    min_bandwidth_gbps: float  # 最低带宽要求 GB/s
    network_bandwidth_gbps: Optional[float] = None  # 用户指定的网络带宽 GB/s
    can_overlap: Optional[bool] = None  # 是否能完全隐藏延迟


class OffloadBandwidthEstimation(BaseModel):
    """KV Cache Offload 带宽估算结果（场景 A: 全量逐层预取）。"""

    model_name: str
    gpu_name: str
    seq_len: int
    context_len: int  # decode 阶段已有上下文长度
    dtype: str
    utilization: float
    phase: Literal["prefill", "decode"]

    kv_per_layer_bytes: float
    flops_per_layer: float
    compute_time_per_layer_ms: float
    min_pcie_bandwidth_gbps: float  # 最低 PCIe 带宽要求 GB/s
    actual_pcie_bandwidth_gbps: float  # GPU 实际 PCIe 带宽
    feasible: bool  # 是否可行
