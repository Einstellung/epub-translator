"""Tests for the OCR client's proxy-free environment, step 1 of pdf_to_epub.

httpx (inside PaddleOCR-VL's OpenAI-compatible client) raises at client
construction time on a socks:// proxy URL, before any request is made, and
NO_PROXY does not suppress that. The vLLM server the client talks to is
always on localhost, so `_run_paddle` must launch it with no proxy variables
at all rather than try to make the proxy value the client accepts.
"""

import pdf_to_epub as P


def test_drops_every_proxy_variable():
    base = {
        "ALL_PROXY": "socks://127.0.0.1:7890/",
        "all_proxy": "socks://127.0.0.1:7890/",
        "HTTP_PROXY": "http://127.0.0.1:7890/",
        "http_proxy": "http://127.0.0.1:7890/",
        "HTTPS_PROXY": "socks://127.0.0.1:7890/",
        "https_proxy": "socks://127.0.0.1:7890/",
    }
    env = P._no_proxy_env(base)
    for name in P._PROXY_ENV_VARS:
        assert name not in env


def test_sets_no_proxy_for_localhost():
    env = P._no_proxy_env({})
    assert env["NO_PROXY"] == "localhost,127.0.0.1"
    assert env["no_proxy"] == "localhost,127.0.0.1"


def test_preserves_unrelated_variables():
    base = {"PATH": "/usr/bin", "ALL_PROXY": "socks://127.0.0.1:7890/"}
    env = P._no_proxy_env(base)
    assert env["PATH"] == "/usr/bin"
    assert "ALL_PROXY" not in env


def test_does_not_mutate_input():
    base = {"ALL_PROXY": "socks://127.0.0.1:7890/"}
    P._no_proxy_env(base)
    assert base == {"ALL_PROXY": "socks://127.0.0.1:7890/"}
