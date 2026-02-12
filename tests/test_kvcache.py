"""KV Cache 显存计算的单元测试。

验证数据来源: DESIGN.md 附录 A 的速算参考表。
"""

import pytest

from kvsim.calc.kvcache import estimate_kv_memory, kv_per_token_per_layer_bytes
from kvsim.models.presets import get_model


class TestKVPerTokenPerLayer:
    """测试单层单 token KV Cache 大小。"""

    def test_llama_3_1_70b_fp16(self):
        """Llama-3.1-70B, fp16: 2 × 8 × 128 × 2 = 4096 bytes."""
        model = get_model("llama-3.1-70b")
        result = kv_per_token_per_layer_bytes(model, "fp16")
        assert result == 2 * 8 * 128 * 2  # 4096 bytes

    def test_llama_3_1_8b_fp16(self):
        """Llama-3.1-8B, fp16: 2 × 8 × 128 × 2 = 4096 bytes."""
        model = get_model("llama-3.1-8b")
        result = kv_per_token_per_layer_bytes(model, "fp16")
        assert result == 4096

    def test_glm_4_9b_fp16(self):
        """GLM-4-9B, fp16: 2 × 2 × 128 × 2 = 1024 bytes."""
        model = get_model("glm-4-9b")
        result = kv_per_token_per_layer_bytes(model, "fp16")
        assert result == 2 * 2 * 128 * 2  # 1024 bytes

    def test_deepseek_v3_mla_fp16(self):
        """DeepSeek-V3, MLA, fp16: (512 + 64) × 2 = 1152 bytes."""
        model = get_model("deepseek-v3")
        result = kv_per_token_per_layer_bytes(model, "fp16")
        assert result == (512 + 64) * 2  # 1152 bytes

    def test_fp8_halves_size(self):
        """fp8 的 KV Cache 应该是 fp16 的一半。"""
        model = get_model("llama-3.1-70b")
        fp16 = kv_per_token_per_layer_bytes(model, "fp16")
        fp8 = kv_per_token_per_layer_bytes(model, "fp8")
        assert fp8 == fp16 / 2

    def test_int4_quarter_size(self):
        """int4 的 KV Cache 应该是 fp16 的四分之一。"""
        model = get_model("llama-3.1-70b")
        fp16 = kv_per_token_per_layer_bytes(model, "fp16")
        int4 = kv_per_token_per_layer_bytes(model, "int4")
        assert int4 == fp16 / 4


class TestEstimateKVMemory:
    """测试 KV Cache 总显存估算。"""

    def test_llama_3_1_70b_4096_fp16(self):
        """Llama-3.1-70B, seq_len=4096, fp16 → 1.25 GB (DESIGN.md 附录 A)。"""
        model = get_model("llama-3.1-70b")
        result = estimate_kv_memory(model, 4096, dtype="fp16")
        # 80 layers × 4096 tokens × 4096 bytes/token = 1,342,177,280 bytes
        assert result.kv_total_bytes == 80 * 4096 * 4096
        assert abs(result.kv_total_gb - 1.25) < 0.01

    def test_llama_3_1_8b_4096_fp16(self):
        """Llama-3.1-8B, seq_len=4096, fp16 → 512 MB。"""
        model = get_model("llama-3.1-8b")
        result = estimate_kv_memory(model, 4096, dtype="fp16")
        expected_bytes = 32 * 4096 * 4096  # 32 layers
        assert result.kv_total_bytes == expected_bytes
        expected_mb = expected_bytes / (1024**2)
        assert abs(expected_mb - 512.0) < 1.0

    def test_tensor_parallel(self):
        """TP=8 时 KV Cache 应该是 TP=1 的 1/8。"""
        model = get_model("llama-3.1-70b")
        r1 = estimate_kv_memory(model, 4096, dtype="fp16", tp_size=1)
        r8 = estimate_kv_memory(model, 4096, dtype="fp16", tp_size=8)
        assert abs(r8.kv_total_bytes - r1.kv_total_bytes / 8) < 1.0

    def test_seq_len_linear(self):
        """KV Cache 应与序列长度成正比。"""
        model = get_model("llama-3.1-70b")
        r1 = estimate_kv_memory(model, 1024, dtype="fp16")
        r4 = estimate_kv_memory(model, 4096, dtype="fp16")
        assert abs(r4.kv_total_bytes / r1.kv_total_bytes - 4.0) < 0.001

    def test_invalid_dtype(self):
        model = get_model("llama-3.1-8b")
        with pytest.raises(ValueError, match="不支持的 dtype"):
            estimate_kv_memory(model, 4096, dtype="fp3")
