"""CLI 参数解析测试。"""

import pytest
import typer

from kvsim.cli import _parse_seq_len_token, _parse_seq_lens


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("4096", 4096),
        ("1k", 1024),
        ("1kb", 1024),
        ("2m", 2 * 1024 * 1024),
        ("2mb", 2 * 1024 * 1024),
        ("1G", 1024**3),
        ("3Gb", 3 * 1024**3),
    ],
)
def test_parse_seq_len_token_with_unit(token: str, expected: int):
    """应支持 k/m/g 及可选 b 后缀，不区分大小写。"""
    assert _parse_seq_len_token(token) == expected


def test_parse_seq_lens_with_comma_and_units():
    """应支持逗号分隔的单位写法。"""
    assert _parse_seq_lens("1k,2mb,4096") == [1024, 2 * 1024 * 1024, 4096]


@pytest.mark.parametrize("token", ["", "1t", "abc", "1kk", "1.5m"])
def test_parse_seq_len_token_invalid(token: str):
    """非法输入应抛出 BadParameter。"""
    with pytest.raises(typer.BadParameter):
        _parse_seq_len_token(token)
