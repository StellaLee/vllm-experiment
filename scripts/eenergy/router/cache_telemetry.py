"""Ground-truth prefix-cache hit rate, read from the real vLLM engine's own Prometheus
/metrics endpoint -- unlike cache_mirror.py's router-side estimate (a client-side hash-chain
mirror that assumes infinite cache capacity and never sees real eviction), this reads vLLM's
own internal counters directly. Split the same way bs_telemetry.py/power_nvml.py are: a pure
parser (heavily tested) and a thin async fetch (network-only, exercised live).

vllm:prefix_cache_queries_total / vllm:prefix_cache_hits_total are both counters, in terms of
NUMBER OF TOKENS (not requests) -- directly comparable to cache_mirror's own token-level
1 - new_tokens/raw_tokens estimate. Both counters are cumulative since process start; since
this project's convention already launches a fresh replica process before every trial, a
single end-of-trial snapshot IS that trial's real hit rate -- no polling/delta needed."""
import re

_QUERIES_RE = re.compile(r'vllm:prefix_cache_queries_total\{([^}]*)\}\s+([0-9.eE+-]+)')
_HITS_RE = re.compile(r'vllm:prefix_cache_hits_total\{([^}]*)\}\s+([0-9.eE+-]+)')


def _value_for_model(pattern: re.Pattern, metrics_text: str, model_name: str) -> float:
    needle = f'model_name="{model_name}"'
    for labels, value in pattern.findall(metrics_text):
        if needle in labels:
            return float(value)
    return 0.0


def parse_prefix_cache(metrics_text: str, model_name: str) -> tuple:
    """(hits, queries) token counts for model_name, parsed from a vLLM /metrics text blob.
    Defaults to (0.0, 0.0) if the model's lines aren't present -- a system-boundary parse of
    another service's output shouldn't take anything down on a scrape miss."""
    hits = _value_for_model(_HITS_RE, metrics_text, model_name)
    queries = _value_for_model(_QUERIES_RE, metrics_text, model_name)
    return hits, queries


async def fetch_prefix_cache(session, host: str, port: int, model_name: str) -> tuple:
    """GET the replica's /metrics and parse it. No retry/timeout policy here -- same
    convention as fetch_running_waiting, caller decides how to handle a failed fetch."""
    url = f"http://{host}:{port}/metrics"
    async with session.get(url) as resp:
        text = await resp.text()
    return parse_prefix_cache(text, model_name)
