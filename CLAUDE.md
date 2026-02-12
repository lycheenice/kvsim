# KVCache Simulator 项目规范

## 项目概述

KV Cache 模拟器 — 用于大模型推理中 KV Cache 显存用量、PD 分离带宽、Offload 带宽的估算工具。

## 技术栈

- Python 3.10+
- CLI: typer + rich
- 数据模型: pydantic v2
- 测试: pytest
- 包管理: pyproject.toml

## 代码规范

- 使用中文注释说明业务逻辑，英文命名变量和函数
- 所有计算公式必须在函数 docstring 中注明来源和推导
- 数值单位统一：显存用 bytes 内部计算，对外展示转换为 MB/GB；带宽用 GB/s；算力用 TFLOPS
- pydantic model 用于所有对外数据结构，内部计算可用 dataclass 或 plain dict
- 每个计算模块必须有对应的单元测试，用已知模型参数验证

## 目录结构

```
src/kvsim/           # 主包
  cli.py             # CLI 入口 (typer)
  engine.py          # 核心引擎
  models/            # 数据模型 (ModelConfig, GPUConfig, presets)
  calc/              # 计算模块 (kvcache, flops, bandwidth)
  output/            # 输出格式化
tests/               # pytest 测试
```

## 关键设计决策

1. **注意力类型**：支持 MHA / GQA / MQA / MLA 四种，通过 ModelConfig.attention_type 区分
2. **MLA 特殊处理**：DeepSeek-V2/V3 的 KV Cache 公式与标准 Transformer 不同，使用 kv_lora_rank + qk_rope_head_dim
3. **config.json 兼容**：不同模型家族的字段名不同，parser 需处理别名（如 ChatGLM 的 multi_query_group_num）
4. **前后端分离**：engine.py 提供纯计算 API，不依赖任何 IO，CLI 和未来 Web 都调用 engine

## 领域知识

详见 `.claude/skills/` 目录下的 skill 文件。
