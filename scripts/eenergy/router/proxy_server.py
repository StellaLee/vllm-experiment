"""aiohttp reverse proxy: the router's network-facing service. Wires Router (routing
decisions), NvmlPowerReader (power telemetry), and a tokenizer together, and transparently
forwards the OpenAI-compatible streaming /v1/completions request/response so the existing
benchmark harness (src/replay_sharegpt.py) needs no changes -- it just points at this
proxy's host:port instead of a replica's directly (spec S3.1, Global Constraints)."""
import asyncio
import time

from aiohttp import web, ClientSession, ClientTimeout
from transformers import AutoTokenizer

from replica_state import ReplicaConfig, ReplicaState
from router_core import Router, WHALE_TOKEN_THRESHOLD
from power_nvml import NvmlPowerReader
from ramp import update_ramp_state
from bs_telemetry import fetch_running_waiting


def build_replica_states(replica_specs: list) -> list:
    """replica_specs: list of dicts with keys matching ReplicaConfig's fields."""
    return [ReplicaState(config=ReplicaConfig(**spec)) for spec in replica_specs]


def format_assignment_record(ts: float, replica_id: str, gpu_index: int,
                              new_tokens: int, raw_tokens: int) -> str:
    """One CSV line for the per-request replica-assignment log: which physical GPU actually
    served each request (so post-hoc analysis can classify a request's power-pressure window
    by ITS OWN replica instead of the coarser fleet-wide any-GPU fallback), plus the P-token
    the router actually used for that decision alongside the raw prompt length -- the only
    place either number is observable, so this is what auditing the KV$-hit discount for
    honesty against a live cache-hit run has to read."""
    return f"{ts:.6f},{replica_id},{gpu_index},{new_tokens},{raw_tokens}\n"


async def power_poll_loop(states: list, reader: NvmlPowerReader, interval_s: float):
    while True:
        for state in states:
            power_w, ts = reader.read(state.config.gpu_index)
            update_ramp_state(state, power_w, ts)
        await asyncio.sleep(interval_s)


async def bs_poll_loop(states: list, model_name: str, interval_s: float):
    """Background poller for bs_source="telemetry": reads each replica's real
    running+waiting count off its own /metrics endpoint. On a fetch failure, keeps the
    replica's last known telemetry_bs rather than snapping it to 0 -- a transient scrape
    miss shouldn't make a busy replica suddenly look empty and get flooded."""
    async with ClientSession(timeout=ClientTimeout(total=interval_s)) as session:
        while True:
            for state in states:
                try:
                    running, waiting = await fetch_running_waiting(
                        session, state.config.host, state.config.port, model_name)
                    state.telemetry_bs = running + waiting
                except Exception:
                    pass
            await asyncio.sleep(interval_s)


def make_app(states: list, policy: str, model_name: str, assignment_log_path: str = None,
             bs_source: str = "local"):
    router = Router(states, policy, bs_source=bs_source)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    by_id = {s.config.replica_id: s for s in states}
    assignment_log = None
    if assignment_log_path:
        # Truncate, not append: each run_router.py invocation is a fresh process (one per
        # condition run), so a stale log from an earlier attempt at the same policy must
        # not silently accumulate underneath this run's rows -- matches how the harness's
        # --output and power_logger.py's own CSV writer both start clean each invocation.
        assignment_log = open(assignment_log_path, "w")
        assignment_log.write("wall_time,replica_id,gpu_index,new_tokens,raw_tokens\n")
        assignment_log.flush()

    async def handle_completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        token_ids = tokenizer.encode(body["prompt"])
        is_whale = len(token_ids) > WHALE_TOKEN_THRESHOLD
        replica_id = router.route(token_ids)
        target = by_id[replica_id].config
        if assignment_log:
            assignment_log.write(format_assignment_record(
                time.time(), replica_id, target.gpu_index,
                router.last_new_tokens, len(token_ids)))
            assignment_log.flush()

        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        url = f"http://{target.host}:{target.port}/v1/completions"
        try:
            async with ClientSession(timeout=ClientTimeout(total=None)) as session:
                async with session.post(url, json=body) as upstream:
                    async for chunk in upstream.content.iter_any():
                        await resp.write(chunk)
        finally:
            router.complete(replica_id, is_whale)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/completions", handle_completions)
    app["states"] = states
    app["router"] = router
    return app


def run(replica_specs: list, policy: str, model_name: str, host: str, port: int,
        power_interval_s: float = 0.5, assignment_log_path: str = None,
        bs_source: str = "local", bs_poll_interval_s: float = 0.5) -> None:
    states = build_replica_states(replica_specs)
    gpu_indices = [s.config.gpu_index for s in states]
    reader = NvmlPowerReader(gpu_indices)
    app = make_app(states, policy, model_name, assignment_log_path, bs_source=bs_source)

    async def _on_startup(app):
        app["power_task"] = asyncio.create_task(
            power_poll_loop(states, reader, power_interval_s))
        if bs_source == "telemetry":
            app["bs_task"] = asyncio.create_task(
                bs_poll_loop(states, model_name, bs_poll_interval_s))

    app.on_startup.append(_on_startup)
    web.run_app(app, host=host, port=port)
