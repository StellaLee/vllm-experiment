import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from bs_telemetry import parse_running_waiting, fetch_running_waiting  # noqa: E402


SAMPLE_METRICS = """\
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="/data/pli/models/Qwen2.5-Coder-7B-Instruct"} 12.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="/data/pli/models/Qwen2.5-Coder-7B-Instruct"} 3.0
vllm:kv_cache_usage_perc{engine="0",model_name="/data/pli/models/Qwen2.5-Coder-7B-Instruct"} 0.41
"""

MULTI_MODEL_METRICS = """\
vllm:num_requests_running{engine="0",model_name="/data/pli/models/other-model"} 99.0
vllm:num_requests_waiting{engine="0",model_name="/data/pli/models/other-model"} 99.0
vllm:num_requests_running{engine="0",model_name="/data/pli/models/Qwen2.5-Coder-7B-Instruct"} 5.0
vllm:num_requests_waiting{engine="0",model_name="/data/pli/models/Qwen2.5-Coder-7B-Instruct"} 1.0
"""


def test_parse_running_waiting_extracts_matching_model():
    running, waiting = parse_running_waiting(SAMPLE_METRICS, "/data/pli/models/Qwen2.5-Coder-7B-Instruct")
    assert running == 12
    assert waiting == 3


def test_parse_running_waiting_matches_only_the_requested_model_label():
    running, waiting = parse_running_waiting(MULTI_MODEL_METRICS, "/data/pli/models/Qwen2.5-Coder-7B-Instruct")
    assert running == 5
    assert waiting == 1


def test_parse_running_waiting_defaults_to_zero_when_model_absent():
    running, waiting = parse_running_waiting(SAMPLE_METRICS, "/no/such/model")
    assert (running, waiting) == (0, 0)


def test_parse_running_waiting_defaults_to_zero_on_empty_text():
    assert parse_running_waiting("", "any-model") == (0, 0)


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


def test_fetch_running_waiting_hits_the_metrics_endpoint_and_parses_it():
    session = _FakeSession(SAMPLE_METRICS)
    running, waiting = asyncio.run(
        fetch_running_waiting(session, "127.0.0.1", 8001, "/data/pli/models/Qwen2.5-Coder-7B-Instruct"))
    assert (running, waiting) == (12, 3)
    assert session.requested_url == "http://127.0.0.1:8001/metrics"
