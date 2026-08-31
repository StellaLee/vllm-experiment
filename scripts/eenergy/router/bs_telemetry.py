"""Optional BS (batch size / in-flight load) source: read it from the real vLLM engine's
own Prometheus /metrics endpoint instead of the router's own dispatch/complete bookkeeping
(load_tracker.py). Ground truth vs. a router-side proxy -- catches things load_tracker can't
see at all, like vLLM-side preemptions. Split the same way power_nvml.py is: a pure parser
(heavily tested) and a thin async fetch (network-only, exercised live)."""
import re

_RUNNING_RE = re.compile(r'vllm:num_requests_running\{([^}]*)\}\s+([0-9.eE+-]+)')
_WAITING_RE = re.compile(r'vllm:num_requests_waiting\{([^}]*)\}\s+([0-9.eE+-]+)')


def _value_for_model(pattern: re.Pattern, metrics_text: str, model_name: str) -> int:
    needle = f'model_name="{model_name}"'
    for labels, value in pattern.findall(metrics_text):
        if needle in labels:
            return int(float(value))
    return 0


def parse_running_waiting(metrics_text: str, model_name: str) -> tuple:
    """(running, waiting) request counts for model_name, parsed from a vLLM /metrics text
    blob. Defaults to (0, 0) if the model's lines aren't present -- a system-boundary parse
    of another service's output shouldn't take the router down on a scrape miss."""
    running = _value_for_model(_RUNNING_RE, metrics_text, model_name)
    waiting = _value_for_model(_WAITING_RE, metrics_text, model_name)
    return running, waiting


async def fetch_running_waiting(session, host: str, port: int, model_name: str) -> tuple:
    """GET the replica's /metrics and parse it. No retry/timeout policy here -- the caller
    (bs_poll_loop in proxy_server.py) decides how to handle a failed fetch (keep the last
    known value rather than snapping a busy replica to 0)."""
    url = f"http://{host}:{port}/metrics"
    async with session.get(url) as resp:
        text = await resp.text()
    return parse_running_waiting(text, model_name)
