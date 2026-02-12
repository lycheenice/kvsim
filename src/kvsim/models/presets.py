"""内置模型预设和 GPU 预设。"""

from __future__ import annotations

from kvsim.models.config import GPUConfig, ModelConfig

# ---------------------------------------------------------------------------
# 模型预设
# ---------------------------------------------------------------------------

MODEL_PRESETS: dict[str, ModelConfig] = {
    "llama-3.1-8b": ModelConfig(
        name="Llama-3.1-8B",
        model_type="llama",
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=14336,
        attention_type="gqa",
    ),
    "llama-3.1-70b": ModelConfig(
        name="Llama-3.1-70B",
        model_type="llama",
        num_hidden_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=28672,
        attention_type="gqa",
    ),
    "llama-3.1-405b": ModelConfig(
        name="Llama-3.1-405B",
        model_type="llama",
        num_hidden_layers=126,
        hidden_size=16384,
        num_attention_heads=128,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=53248,
        attention_type="gqa",
    ),
    "qwen2.5-7b": ModelConfig(
        name="Qwen2.5-7B",
        model_type="qwen2",
        num_hidden_layers=28,
        hidden_size=3584,
        num_attention_heads=28,
        num_key_value_heads=4,
        head_dim=128,
        intermediate_size=18944,
        attention_type="gqa",
    ),
    "qwen2.5-72b": ModelConfig(
        name="Qwen2.5-72B",
        model_type="qwen2",
        num_hidden_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=29568,
        attention_type="gqa",
    ),
    "glm-4-9b": ModelConfig(
        name="GLM-4-9B",
        model_type="chatglm",
        num_hidden_layers=40,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=13696,
        attention_type="gqa",
    ),
    "deepseek-v3": ModelConfig(
        name="DeepSeek-V3",
        model_type="deepseek_v2",
        num_hidden_layers=61,
        hidden_size=7168,
        num_attention_heads=128,
        num_key_value_heads=128,  # MLA 不使用这个值；保留以满足字段约束
        head_dim=128,
        intermediate_size=18432,
        attention_type="mla",
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    ),
    "mistral-7b": ModelConfig(
        name="Mistral-7B",
        model_type="mistral",
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=14336,
        attention_type="gqa",
    ),
}

# ---------------------------------------------------------------------------
# GPU 预设
# ---------------------------------------------------------------------------

GPU_PRESETS: dict[str, GPUConfig] = {
    "h100": GPUConfig(
        name="H100 SXM 80GB",
        peak_flops_fp16=989.4,
        peak_flops_fp8=1978.9,
        hbm_bandwidth=3350.0,
        pcie_bandwidth=128.0,  # PCIe Gen5 x16 双向
        memory_capacity=80.0,
    ),
    "h800": GPUConfig(
        name="H800 SXM 80GB",
        peak_flops_fp16=989.4,
        peak_flops_fp8=1978.9,
        hbm_bandwidth=3350.0,
        pcie_bandwidth=128.0,
        memory_capacity=80.0,
    ),
    "a100": GPUConfig(
        name="A100 SXM 80GB",
        peak_flops_fp16=312.0,
        peak_flops_fp8=None,
        hbm_bandwidth=2000.0,
        pcie_bandwidth=64.0,  # PCIe Gen4 x16 双向
        memory_capacity=80.0,
    ),
    "a800": GPUConfig(
        name="A800 SXM 80GB",
        peak_flops_fp16=312.0,
        peak_flops_fp8=None,
        hbm_bandwidth=2000.0,
        pcie_bandwidth=64.0,
        memory_capacity=80.0,
    ),
    "l40s": GPUConfig(
        name="L40S 48GB",
        peak_flops_fp16=362.05,
        peak_flops_fp8=733.0,
        hbm_bandwidth=864.0,
        pcie_bandwidth=64.0,
        memory_capacity=48.0,
    ),
}


def get_model(name: str) -> ModelConfig:
    """根据名称获取模型预设，不区分大小写。"""
    key = name.lower().strip()
    if key not in MODEL_PRESETS:
        available = ", ".join(sorted(MODEL_PRESETS.keys()))
        raise ValueError(f"未知模型 '{name}'，可用预设: {available}")
    return MODEL_PRESETS[key]


def get_gpu(name: str) -> GPUConfig:
    """根据名称获取 GPU 预设，不区分大小写。"""
    key = name.lower().strip()
    if key not in GPU_PRESETS:
        available = ", ".join(sorted(GPU_PRESETS.keys()))
        raise ValueError(f"未知 GPU '{name}'，可用预设: {available}")
    return GPU_PRESETS[key]
