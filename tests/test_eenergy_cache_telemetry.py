import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from cache_telemetry import parse_prefix_cache, fetch_prefix_cache  # noqa: E402


# Real sample captured live from a running vLLM 0.23.0 replica's /metrics endpoint
# (rampandroute condition, engine="0" label) -- not synthesized.
SAMPLE_METRICS = """\
# HELP vllm:prefix_cache_queries_total Prefix cache queries, in terms of number of queried tokens.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="/data/pli/models/Qwen2.5-Coder-7B-Instruct"} 371.0
# HELP vllm:prefix_cache_hits_total Prefix cache hits, in terms of number of cached tokens.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="/data/pli/models/Qwen2.5-Coder-7B-Instruct"} 0.0
# HELP vllm:external_prefix_cache_queries_total External prefix cache queries from KV connector cross-instance cache sharing, in terms of number of queried tokens.
# TYPE vllm:external_prefix_cache_queries_total counter
vllm:external_prefix_cache_queries_total{engine="0",model_name="/data/pli/models/Qwen2.5-Coder-7B-Instruct"} 0.0
# HELP vllm:external_prefix_cache_hits_total External prefix cache hits from KV connector cross-instance cache sharing, in terms of number of cached tokens.
# TYPE vllm:external_prefix_cache_hits_total counter
vllm:external_prefix_cache_hits_total{engine="0",model_name="/data/pli/models/Qwen2.5-Coder-7B-Instruct"} 0.0
"""

MULTI_MODEL_METRICS = """\
vllm:prefix_cache_queries_total{engine="0",model_name="/data/pli/models/other-model"} 999.0
vllm:prefix_cache_hits_total{engine="0",model_name="/data/pli/models/other-model"} 999.0
vllm:prefix_cache_queries_total{engine="0",model_name="/data/pli/models/Qwen2.5-Coder-7B-Instruct"} 200.0
vllm:prefix_cache_hits_total{engine="0",model_name="/data/pli/models/Qwen2.5-Coder-7B-Instruct"} 150.0
"""


def test_parse_prefix_cache_extracts_matching_model():
    hits, queries = parse_prefix_cache(SAMPLE_METRICS, "/data/pli/models/Qwen2.5-Coder-7B-Instruct")
    assert (hits, queries) == (0.0, 371.0)


def test_parse_prefix_cache_matches_only_the_requested_model_label():
    hits, queries = parse_prefix_cache(MULTI_MODEL_METRICS, "/data/pli/models/Qwen2.5-Coder-7B-Instruct")
    assert (hits, queries) == (150.0, 200.0)


def test_parse_prefix_cache_defaults_to_zero_when_model_absent():
    assert parse_prefix_cache(SAMPLE_METRICS, "/no/such/model") == (0.0, 0.0)


def test_parse_prefix_cache_defaults_to_zero_on_empty_text():
    assert parse_prefix_cache("", "any-model") == (0.0, 0.0)


def test_parse_prefix_cache_ignores_the_external_prefix_cache_metrics():
    """external_prefix_cache_* is a DIFFERENT metric (cross-instance KV connector sharing,
    not this replica's own prefix cache) -- the regex must not accidentally match its
    trailing "prefix_cache_..._total" substring."""
    text = ('vllm:external_prefix_cache_queries_total{engine="0",model_name="m"} 9999.0\n'
            'vllm:external_prefix_cache_hits_total{engine="0",model_name="m"} 9999.0\n'
            'vllm:prefix_cache_queries_total{engine="0",model_name="m"} 5.0\n'
            'vllm:prefix_cache_hits_total{engine="0",model_name="m"} 2.0\n')
    assert parse_prefix_cache(text, "m") == (2.0, 5.0)


class _FakeResponse:
    def __init__(self, text):
        self._text = text

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, text):
        self._text = text
        self.requested_url = None

    def get(self, url):
        self.requested_url = url
        return _FakeResponse(self._text)


def test_fetch_prefix_cache_hits_the_metrics_endpoint_and_parses_it():
    session = _FakeSession(SAMPLE_METRICS)
    hits, queries = asyncio.run(
        fetch_prefix_cache(session, "127.0.0.1", 8001, "/data/pli/models/Qwen2.5-Coder-7B-Instruct"))
    assert (hits, queries) == (0.0, 371.0)
    assert session.requested_url == "http://127.0.0.1:8001/metrics"
