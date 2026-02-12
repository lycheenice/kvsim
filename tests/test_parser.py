"""HuggingFace config.json 解析器的单元测试。"""

import json
import tempfile
from pathlib import Path

from kvsim.models.parser import parse_hf_config


def _write_config(cfg: dict) -> str:
    """将配置字典写入临时 JSON 文件，返回路径。"""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, f)
    f.close()
    return f.name


class TestParseHFConfig:
    """config.json 解析测试。"""

    def test_llama_style(self):
        """标准 Llama 风格 config.json。"""
        cfg = {
            "model_type": "llama",
            "num_hidden_layers": 80,
            "hidden_size": 8192,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "intermediate_size": 28672,
        }
        path = _write_config(cfg)
        model = parse_hf_config(path)
        assert model.num_hidden_layers == 80
        assert model.effective_kv_heads == 8
        assert model.effective_head_dim == 128  # 8192 // 64
        assert model.attention_type == "gqa"
        Path(path).unlink()

    def test_chatglm_style(self):
        """ChatGLM 风格: 使用 multi_query_group_num 代替 num_key_value_heads。"""
        cfg = {
            "model_type": "glm4",
            "num_hidden_layers": 40,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "multi_query_group_num": 2,
            "intermediate_size": 13696,
        }
        path = _write_config(cfg)
        model = parse_hf_config(path)
        assert model.effective_kv_heads == 2
        assert model.attention_type == "gqa"
        Path(path).unlink()

    def test_mha_fallback(self):
        """缺少 num_key_value_heads 时回退到 MHA。"""
        cfg = {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
        }
        path = _write_config(cfg)
        model = parse_hf_config(path)
        assert model.effective_kv_heads == 32
        assert model.attention_type == "mha"
        Path(path).unlink()

    def test_mqa_detection(self):
        """num_key_value_heads=1 时检测为 MQA。"""
        cfg = {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 1,
            "intermediate_size": 11008,
        }
        path = _write_config(cfg)
        model = parse_hf_config(path)
        assert model.effective_kv_heads == 1
        assert model.attention_type == "mqa"
        Path(path).unlink()

    def test_deepseek_mla(self):
        """DeepSeek-V2 风格: 检测 MLA 注意力类型。"""
        cfg = {
            "model_type": "deepseek_v2",
            "num_hidden_layers": 61,
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "intermediate_size": 18432,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
        }
        path = _write_config(cfg)
        model = parse_hf_config(path)
        assert model.attention_type == "mla"
        assert model.kv_lora_rank == 512
        assert model.qk_rope_head_dim == 64
        Path(path).unlink()

    def test_explicit_head_dim(self):
        """显式提供 head_dim 时应优先使用。"""
        cfg = {
            "model_type": "qwen2",
            "num_hidden_layers": 80,
            "hidden_size": 8192,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 29568,
        }
        path = _write_config(cfg)
        model = parse_hf_config(path)
        assert model.effective_head_dim == 128
        Path(path).unlink()
