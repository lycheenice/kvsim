"""FLOPs 计算的单元测试。

验证数据来源: .claude/skills/kvcache-and-flops-formulas.md 中的数值示例。
"""

from kvsim.calc.flops import (
    attn_flops_per_layer,
    ffn_flops_per_layer,
    total_flops_per_layer,
    total_flops_per_layer_decode,
)
from kvsim.models.presets import get_model


class TestPrefillFLOPs:
    """测试 Prefill 阶段每层 FLOPs。"""

    def test_llama_3_1_70b_prefill(self):
        """Llama-3.1-70B, seq_len=4096 的每层 FLOPs 验证。

        按 skill 文件中的手动计算:
          Attn ≈ 1.786 TFLOP
          FFN  ≈ 5.780 TFLOP
          Total ≈ 7.566 TFLOP

        这里验证 FFN 精确值: 6 × 4096 × 8192 × 28672
        """
        model = get_model("llama-3.1-70b")
        s = 4096

        ffn = ffn_flops_per_layer(model, s)
        expected_ffn = 6.0 * s * 8192 * 28672
        assert ffn == expected_ffn

        attn = attn_flops_per_layer(model, s)
        total = total_flops_per_layer(model, s)
        assert total == attn + ffn

        # 总 FLOPs 在 TFLOP 量级，约 7.5T
        total_tflop = total / 1e12
        assert 7.0 < total_tflop < 8.5

    def test_ffn_proportional_to_seq_len(self):
        """FFN FLOPs 应与序列长度成正比。"""
        model = get_model("llama-3.1-70b")
        f1 = ffn_flops_per_layer(model, 1024)
        f4 = ffn_flops_per_layer(model, 4096)
        assert abs(f4 / f1 - 4.0) < 0.001

    def test_attn_grows_quadratically_with_seq_len(self):
        """Attention FLOPs 在长序列时应接近 O(s²)，短序列时线性项占主导。"""
        model = get_model("llama-3.1-8b")
        # 对于很大的 s，s² 项 (4×n_q×s²×d_h) 应占主导
        a_large = attn_flops_per_layer(model, 65536)
        a_half = attn_flops_per_layer(model, 32768)
        # 如果是纯 O(s²)，ratio = 4；实际因线性项存在，ratio 略小于 4
        ratio = a_large / a_half
        assert 3.5 < ratio < 4.1


class TestDecodeFLOPs:
    """测试 Decode 阶段每层 FLOPs。"""

    def test_decode_much_smaller_than_prefill(self):
        """Decode 每步 FLOPs 应远小于 Prefill 每步 (相同上下文长度)。"""
        model = get_model("llama-3.1-70b")
        ctx = 4096
        prefill = total_flops_per_layer(model, ctx)
        decode = total_flops_per_layer_decode(model, ctx)
        # Decode 处理 1 个 token vs Prefill 处理 4096 个 token
        assert decode < prefill / 100

    def test_decode_flops_llama_70b(self):
        """Llama-3.1-70B decode FLOPs 数量级验证。

        按 skill 文件: ≈ 1.847 GFLOP
        """
        model = get_model("llama-3.1-70b")
        decode = total_flops_per_layer_decode(model, 4096)
        gflop = decode / 1e9
        assert 1.0 < gflop < 3.0
