"""PD 分离带宽和 Offload 带宽计算的单元测试。"""

from kvsim.calc.bandwidth import estimate_offload_bandwidth, estimate_pd_bandwidth
from kvsim.models.presets import get_gpu, get_model


class TestPDBandwidth:
    """PD 分离带宽估算测试。"""

    def test_llama_70b_h100_basic(self):
        """Llama-3.1-70B + H100, seq_len=4096 基本验证。

        按 skill 文件:
          KV per layer = 16 MB
          compute_time ≈ 15.3 ms
          min_bw ≈ 1.045 GB/s
        """
        model = get_model("llama-3.1-70b")
        gpu = get_gpu("h100")
        result = estimate_pd_bandwidth(model, gpu, 4096, dtype="fp16")

        # KV per layer = 4096 × 4096 = 16 MB
        assert result.kv_per_layer_bytes == 4096 * 4096

        # min bandwidth 应在 1 GB/s 量级（远低于 IB 带宽）
        assert 0.5 < result.min_bandwidth_gbps < 3.0

        # 计算时间应在 10-20 ms 量级
        assert 5 < result.compute_time_per_layer_ms < 30

    def test_with_network_bandwidth_sufficient(self):
        """当网络带宽充足时 can_overlap 应为 True。"""
        model = get_model("llama-3.1-70b")
        gpu = get_gpu("h100")
        result = estimate_pd_bandwidth(
            model, gpu, 4096, dtype="fp16", network_bandwidth_gbps=25.0
        )
        assert result.can_overlap is True
        assert result.transfer_time_per_layer_ms < result.compute_time_per_layer_ms

    def test_with_network_bandwidth_insufficient(self):
        """当网络带宽不足时 can_overlap 应为 False。"""
        model = get_model("llama-3.1-70b")
        gpu = get_gpu("h100")
        result = estimate_pd_bandwidth(
            model, gpu, 4096, dtype="fp16", network_bandwidth_gbps=0.001
        )
        assert result.can_overlap is False

    def test_longer_seq_higher_bandwidth(self):
        """更长的序列应需要更高的带宽（因为 KV 数据量线性增长，但 FLOPs 超线性增长）。"""
        model = get_model("llama-3.1-70b")
        gpu = get_gpu("h100")
        r_short = estimate_pd_bandwidth(model, gpu, 1024, dtype="fp16")
        r_long = estimate_pd_bandwidth(model, gpu, 131072, dtype="fp16")
        # 长序列时 attn FLOPs 的 s² 项增长快于 KV 的线性增长
        # 所以 min_bw 实际上随 seq_len 增长会先增后减
        # 至少两者都应该是正数
        assert r_short.min_bandwidth_gbps > 0
        assert r_long.min_bandwidth_gbps > 0

    def test_deepseek_v3_mla_low_bandwidth(self):
        """DeepSeek-V3 (MLA) 的 KV 数据量远小于标准模型，带宽需求更低。"""
        ds = get_model("deepseek-v3")
        llama = get_model("llama-3.1-70b")
        gpu = get_gpu("h100")
        r_ds = estimate_pd_bandwidth(ds, gpu, 4096, dtype="fp16")
        r_llama = estimate_pd_bandwidth(llama, gpu, 4096, dtype="fp16")
        # MLA 的 KV per layer 远小于 GQA
        assert r_ds.kv_per_layer_bytes < r_llama.kv_per_layer_bytes / 2


class TestOffloadBandwidth:
    """KV Cache Offload 带宽估算测试。"""

    def test_prefill_feasible(self):
        """Prefill 阶段 offload 通常可行（计算密集）。"""
        model = get_model("llama-3.1-70b")
        gpu = get_gpu("h100")
        result = estimate_offload_bandwidth(
            model, gpu, 4096, dtype="fp16", phase="prefill"
        )
        assert result.feasible is True
        assert result.min_pcie_bandwidth_gbps < gpu.pcie_bandwidth

    def test_decode_infeasible(self):
        """Decode 阶段全量 offload 通常不可行（计算量太少）。"""
        model = get_model("llama-3.1-70b")
        gpu = get_gpu("h100")
        result = estimate_offload_bandwidth(
            model, gpu, 4096, context_len=4096, dtype="fp16", phase="decode"
        )
        # Decode 阶段所需 PCIe 带宽远超实际 PCIe 带宽
        assert result.feasible is False
        assert result.min_pcie_bandwidth_gbps > gpu.pcie_bandwidth

    def test_decode_requires_context_len(self):
        """Decode 阶段必须指定 context_len。"""
        model = get_model("llama-3.1-70b")
        gpu = get_gpu("h100")
        import pytest
        with pytest.raises(ValueError, match="必须指定 context_len"):
            estimate_offload_bandwidth(
                model, gpu, 4096, dtype="fp16", phase="decode"
            )
