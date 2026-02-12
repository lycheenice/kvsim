"""CLI 入口 — 基于 typer 构建命令行工具。

命令:
  kvsim memory          KV Cache 显存估算
  kvsim pd-bandwidth    PD 分离带宽估算
  kvsim offload         KV Cache Offload 带宽估算
  kvsim list-models     列出内置模型预设
  kvsim list-gpus       列出内置 GPU 预设
"""

from __future__ import annotations

from typing import Optional

import typer

from kvsim.engine import (
    compute_kv_memory,
    compute_offload_bandwidth,
    compute_pd_bandwidth,
    resolve_gpu,
    resolve_model,
)
from kvsim.models.presets import GPU_PRESETS, MODEL_PRESETS
from kvsim.output.formatter import (
    print_gpu_list,
    print_kv_memory_single,
    print_kv_memory_table,
    print_model_list,
    print_offload_bandwidth,
    print_pd_bandwidth,
    print_pd_bandwidth_table,
)

app = typer.Typer(
    name="kvsim",
    help="KV Cache Simulator — LLM 推理 KV Cache 显存与带宽估算工具",
    no_args_is_help=True,
)


def _parse_seq_lens(value: str) -> list[int]:
    """解析逗号分隔的序列长度列表。"""
    parts = [p.strip() for p in value.split(",")]
    return [int(p) for p in parts if p]


# ---------------------------------------------------------------------------
# kvsim memory
# ---------------------------------------------------------------------------

@app.command("memory")
def cmd_memory(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="内置模型预设名称"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="HuggingFace config.json 路径"),
    seq_len: str = typer.Option("4096", "--seq-len", "-s", help="序列长度, 支持逗号分隔多个值"),
    dtype: str = typer.Option("fp16", "--dtype", "-d", help="数据类型: fp16, bf16, fp8, int8, int4"),
    tp_size: int = typer.Option(1, "--tp", help="Tensor Parallelism 大小"),
    gpu_mem: Optional[float] = typer.Option(None, "--gpu-mem", help="GPU 显存 (GB), 用于计算占比"),
) -> None:
    """估算 KV Cache 显存用量。"""
    model_cfg = resolve_model(model, config)
    seq_lens = _parse_seq_lens(seq_len)

    results = compute_kv_memory(model_cfg, seq_lens, dtype=dtype, tp_size=tp_size)

    if len(results) == 1:
        attn_info = f"{model_cfg.attention_type.upper()} ({model_cfg.effective_kv_heads} KV heads)"
        print_kv_memory_single(results[0], model_info=attn_info)
    else:
        print_kv_memory_table(results, gpu_memory_gb=gpu_mem)


# ---------------------------------------------------------------------------
# kvsim pd-bandwidth
# ---------------------------------------------------------------------------

@app.command("pd-bandwidth")
def cmd_pd_bandwidth(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="内置模型预设名称"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="HuggingFace config.json 路径"),
    seq_len: str = typer.Option("4096", "--seq-len", "-s", help="序列长度, 支持逗号分隔多个值"),
    gpu: str = typer.Option("h100", "--gpu", "-g", help="GPU 预设名称"),
    dtype: str = typer.Option("fp16", "--dtype", "-d", help="数据类型"),
    utilization: float = typer.Option(0.5, "--utilization", "-u", help="算力利用率 (0-1)"),
    network_bw: Optional[float] = typer.Option(None, "--network-bw", help="网络带宽 (GB/s), 用于判断能否掩盖延迟"),
) -> None:
    """估算 PD 分离逐层异步传输的最低带宽需求。"""
    model_cfg = resolve_model(model, config)
    gpu_cfg = resolve_gpu(gpu)
    seq_lens = _parse_seq_lens(seq_len)

    results = compute_pd_bandwidth(
        model_cfg, gpu_cfg, seq_lens,
        dtype=dtype,
        utilization=utilization,
        network_bandwidth_gbps=network_bw,
    )

    if len(results) == 1:
        print_pd_bandwidth(results[0])
    else:
        print_pd_bandwidth_table(results)


# ---------------------------------------------------------------------------
# kvsim offload
# ---------------------------------------------------------------------------

@app.command("offload")
def cmd_offload(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="内置模型预设名称"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="HuggingFace config.json 路径"),
    seq_len: int = typer.Option(4096, "--seq-len", "-s", help="序列长度"),
    context_len: Optional[int] = typer.Option(None, "--context-len", help="Decode 上下文长度, 默认等于 seq-len"),
    gpu: str = typer.Option("h100", "--gpu", "-g", help="GPU 预设名称"),
    dtype: str = typer.Option("fp16", "--dtype", "-d", help="数据类型"),
    utilization: float = typer.Option(0.5, "--utilization", "-u", help="算力利用率 (0-1)"),
    phase: str = typer.Option("prefill", "--phase", "-p", help="推理阶段: prefill 或 decode"),
) -> None:
    """估算 KV Cache Offload (GPU→CPU) 的最低 PCIe 带宽需求。"""
    model_cfg = resolve_model(model, config)
    gpu_cfg = resolve_gpu(gpu)

    result = compute_offload_bandwidth(
        model_cfg, gpu_cfg, seq_len,
        context_len=context_len,
        dtype=dtype,
        utilization=utilization,
        phase=phase,
    )
    print_offload_bandwidth(result)


# ---------------------------------------------------------------------------
# kvsim list-models / list-gpus
# ---------------------------------------------------------------------------

@app.command("list-models")
def cmd_list_models() -> None:
    """列出内置模型预设。"""
    print_model_list(MODEL_PRESETS)


@app.command("list-gpus")
def cmd_list_gpus() -> None:
    """列出内置 GPU 预设。"""
    print_gpu_list(GPU_PRESETS)


if __name__ == "__main__":
    app()
