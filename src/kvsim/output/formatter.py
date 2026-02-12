"""输出格式化 — 使用 rich 生成美观的 CLI 表格和面板。"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kvsim.models.config import (
    KVCacheEstimation,
    OffloadBandwidthEstimation,
    PDBandwidthEstimation,
)

console = Console()


def _fmt_bytes(b: float) -> str:
    """将字节数格式化为可读字符串。"""
    if b < 1024:
        return f"{b:.0f} B"
    elif b < 1024**2:
        return f"{b / 1024:.1f} KB"
    elif b < 1024**3:
        return f"{b / 1024**2:.1f} MB"
    else:
        return f"{b / 1024**3:.2f} GB"


def _fmt_flops(f: float) -> str:
    """将 FLOPs 格式化为可读字符串。"""
    if f < 1e9:
        return f"{f / 1e6:.1f} MFLOP"
    elif f < 1e12:
        return f"{f / 1e9:.2f} GFLOP"
    else:
        return f"{f / 1e12:.3f} TFLOP"


# ---------------------------------------------------------------------------
# KV Cache 显存估算
# ---------------------------------------------------------------------------

def print_kv_memory_single(result: KVCacheEstimation, model_info: str = "") -> None:
    """打印单个序列长度的 KV Cache 估算结果。"""
    lines = [
        f"  Model:             {result.model_name}",
    ]
    if model_info:
        lines.append(f"  Attention Type:    {model_info}")
    lines += [
        f"  Sequence Length:   {result.seq_len:,}",
        f"  Data Type:         {result.dtype}",
    ]
    if result.tp_size > 1:
        lines.append(f"  Tensor Parallel:   {result.tp_size}")

    lines.append("")
    lines.append(f"  Per Token/Layer:   {_fmt_bytes(result.kv_per_token_per_layer_bytes)}")
    lines.append(f"  Per Layer:         {_fmt_bytes(result.kv_per_layer_bytes)}")
    lines.append(f"  Total:             {_fmt_bytes(result.kv_total_bytes)}")

    panel = Panel(
        "\n".join(lines),
        title="KV Cache Memory Estimation",
        border_style="cyan",
    )
    console.print(panel)


def print_kv_memory_table(
    results: list[KVCacheEstimation],
    gpu_memory_gb: float | None = None,
) -> None:
    """打印多个序列长度的 KV Cache 估算对比表格。"""
    table = Table(title=f"KV Cache Memory — {results[0].model_name} ({results[0].dtype})")
    table.add_column("Seq Length", justify="right")
    table.add_column("Per Layer", justify="right")
    table.add_column("Total", justify="right")
    if gpu_memory_gb:
        table.add_column(f"% of {gpu_memory_gb:.0f}G", justify="right")

    for r in results:
        row = [
            f"{r.seq_len:,}",
            _fmt_bytes(r.kv_per_layer_bytes),
            _fmt_bytes(r.kv_total_bytes),
        ]
        if gpu_memory_gb:
            pct = r.kv_total_gb / gpu_memory_gb * 100
            row.append(f"{pct:.2f}%")
        table.add_row(*row)

    console.print(table)


# ---------------------------------------------------------------------------
# PD 分离带宽估算
# ---------------------------------------------------------------------------

def print_pd_bandwidth(result: PDBandwidthEstimation) -> None:
    """打印 PD 分离带宽估算结果。"""
    lines = [
        f"  Model:                {result.model_name}",
        f"  GPU (Prefill):        {result.gpu_name}",
        f"  Sequence Length:      {result.seq_len:,}",
        f"  Data Type:            {result.dtype}",
        f"  Utilization:          {result.utilization:.0%}",
        "",
        f"  KV per Layer:         {_fmt_bytes(result.kv_per_layer_bytes)}",
        f"  Attn FLOPs/Layer:     {_fmt_flops(result.attn_flops_per_layer)}",
        f"  FFN  FLOPs/Layer:     {_fmt_flops(result.ffn_flops_per_layer)}",
        f"  Total FLOPs/Layer:    {_fmt_flops(result.total_flops_per_layer)}",
        "",
        f"  Compute Time/Layer:   {result.compute_time_per_layer_ms:.3f} ms",
        f"  Min Bandwidth:        {result.min_bandwidth_gbps:.2f} GB/s",
    ]
    if result.network_bandwidth_gbps is not None:
        lines.append(f"  Network Bandwidth:    {result.network_bandwidth_gbps:.1f} GB/s")
        lines.append(f"  Transfer Time/Layer:  {result.transfer_time_per_layer_ms:.3f} ms")
        overlap_str = "[green]YES[/green]" if result.can_overlap else "[red]NO[/red]"
        lines.append(f"  Can Fully Overlap:    {overlap_str}")

    panel = Panel(
        "\n".join(lines),
        title="PD Disaggregation Bandwidth Estimation",
        border_style="yellow",
    )
    console.print(panel)


def print_pd_bandwidth_table(results: list[PDBandwidthEstimation]) -> None:
    """打印多个序列长度的 PD 带宽估算对比表格。"""
    table = Table(title="PD Disaggregation Bandwidth — " + results[0].model_name)
    table.add_column("Seq Length", justify="right")
    table.add_column("KV/Layer", justify="right")
    table.add_column("FLOPs/Layer", justify="right")
    table.add_column("Compute (ms)", justify="right")
    table.add_column("Min BW (GB/s)", justify="right")

    for r in results:
        table.add_row(
            f"{r.seq_len:,}",
            _fmt_bytes(r.kv_per_layer_bytes),
            _fmt_flops(r.total_flops_per_layer),
            f"{r.compute_time_per_layer_ms:.3f}",
            f"{r.min_bandwidth_gbps:.2f}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Offload 带宽估算
# ---------------------------------------------------------------------------

def print_offload_bandwidth(result: OffloadBandwidthEstimation) -> None:
    """打印 Offload 带宽估算结果。"""
    lines = [
        f"  Model:                {result.model_name}",
        f"  GPU:                  {result.gpu_name}",
        f"  Phase:                {result.phase}",
        f"  Sequence Length:      {result.seq_len:,}",
        f"  Context Length:       {result.context_len:,}",
        f"  Data Type:            {result.dtype}",
        f"  Utilization:          {result.utilization:.0%}",
        "",
        f"  KV per Layer:         {_fmt_bytes(result.kv_per_layer_bytes)}",
        f"  FLOPs per Layer:      {_fmt_flops(result.flops_per_layer)}",
        f"  Compute Time/Layer:   {result.compute_time_per_layer_ms:.4f} ms",
        "",
        f"  Min PCIe BW Needed:   {result.min_pcie_bandwidth_gbps:.2f} GB/s",
        f"  Actual PCIe BW:       {result.actual_pcie_bandwidth_gbps:.1f} GB/s",
    ]
    feasible_str = "[green]YES[/green]" if result.feasible else "[red]NO[/red]"
    lines.append(f"  Feasible:             {feasible_str}")

    panel = Panel(
        "\n".join(lines),
        title="KV Cache Offload Bandwidth Estimation",
        border_style="magenta",
    )
    console.print(panel)


# ---------------------------------------------------------------------------
# 模型/GPU 列表
# ---------------------------------------------------------------------------

def print_model_list(models: dict) -> None:
    """打印内置模型预设列表。"""
    table = Table(title="Available Model Presets")
    table.add_column("Key", style="cyan")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Layers", justify="right")
    table.add_column("Hidden", justify="right")
    table.add_column("Q Heads", justify="right")
    table.add_column("KV Heads", justify="right")
    table.add_column("Attn")

    for key, m in models.items():
        table.add_row(
            key,
            m.name,
            m.model_type,
            str(m.num_hidden_layers),
            str(m.hidden_size),
            str(m.num_attention_heads),
            str(m.effective_kv_heads),
            m.attention_type.upper(),
        )

    console.print(table)


def print_gpu_list(gpus: dict) -> None:
    """打印内置 GPU 预设列表。"""
    table = Table(title="Available GPU Presets")
    table.add_column("Key", style="cyan")
    table.add_column("Name")
    table.add_column("FP16 (TFLOPS)", justify="right")
    table.add_column("HBM BW (GB/s)", justify="right")
    table.add_column("PCIe BW (GB/s)", justify="right")
    table.add_column("Memory (GB)", justify="right")

    for key, g in gpus.items():
        table.add_row(
            key,
            g.name,
            f"{g.peak_flops_fp16:.1f}",
            f"{g.hbm_bandwidth:.0f}",
            f"{g.pcie_bandwidth:.0f}",
            f"{g.memory_capacity:.0f}",
        )

    console.print(table)
