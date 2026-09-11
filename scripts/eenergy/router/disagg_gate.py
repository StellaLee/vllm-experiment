"""Peak-shaving admission gate shim for 8-GPU P/D-disaggregated serving (see
docs/superpowers/specs/2026-09-11-disagg-peak-shaving-gate-design.md). Sits in front of the
unmodified, vendored nixl_toy_proxy_server.py -- decides WHETHER/WHEN to admit a request
against two independent per-pool power budgets, never touches HOW requests are routed once
admitted (that's entirely the Nixl proxy's job, unchanged). Mirrors proxy_server.py's own
gate-in-front-of-routing separation, generalized from one combined fleet budget to two
independent pool budgets."""
import asyncio
import time

from aiohttp import web, ClientSession, ClientTimeout
from transformers import AutoTokenizer

from power_budget import (PowerBudget, dual_budget_would_exceed, estimate_marginal_energy_j,
                           DecodeByteEstimator)
from power_nvml import NvmlPowerReader


def build_power_budgets(peak_cap_prefill_w: float, peak_cap_decode_w: float,
                         peak_window_s: float = 30.0) -> tuple:
    """Returns (budget_prefill, budget_decode), each a fresh PowerBudget if its own cap is
    set (not None), else None -- independently, so the gate can run with only one pool capped
    (useful for isolating which pool actually drives peak power). Mirrors proxy_server.py's
    build_power_budget, generalized to two independent pools."""
    budget_prefill = PowerBudget(peak_window_s) if peak_cap_prefill_w is not None else None
    budget_decode = PowerBudget(peak_window_s) if peak_cap_decode_w is not None else None
    return budget_prefill, budget_decode


def pool_power_w(reader, gpu_indices: list) -> float:
    """Sum of a live NVML power read across every GPU in one pool -- the aggregate
    instantaneous power that pool's PowerBudget tracks. Unlike proxy_server.py's
    fleet_power_w (which sums a cached state.last_power_w kept current by a separate
    ramp-ceiling poller), this reads NVML directly at poll time: disagg_gate.py has no
    ReplicaState/ramp-ceiling machinery to piggyback on, and doesn't need any -- the gate
    only needs a fleet-aggregate power number, not per-replica ramp tracking."""
    return sum(reader.read(i)[0] for i in gpu_indices)


async def power_poll_loop(reader: NvmlPowerReader, prefill_gpu_indices: list,
                           decode_gpu_indices: list, interval_s: float,
                           budget_prefill=None, budget_decode=None) -> None:
    while True:
        t = time.time()
        if budget_prefill is not None:
            budget_prefill.record(t, pool_power_w(reader, prefill_gpu_indices))
        if budget_decode is not None:
            budget_decode.record(t, pool_power_w(reader, decode_gpu_indices))
        await asyncio.sleep(interval_s)


async def forward_to_nixl_proxy(nixl_proxy_url: str, body: dict, resp: web.StreamResponse) -> int:
    """Streams the ORIGINAL, unmodified client request body to the Nixl proxy's
    /v1/completions and writes each chunk straight through to resp -- same forwarding pattern
    proxy_server.py already uses to forward to a chosen replica (ClientSession + StreamResponse
    + iter_any), pointed at the Nixl proxy's fixed URL instead of a routed replica's. Returns
    the total response byte count, for DecodeByteEstimator.record_completion."""
    url = f"{nixl_proxy_url}/v1/completions"
    response_bytes = 0
    async with ClientSession(timeout=ClientTimeout(total=None)) as session:
        async with session.post(url, json=body) as upstream:
            async for chunk in upstream.content.iter_any():
                response_bytes += len(chunk)
                await resp.write(chunk)
    return response_bytes


def make_app(prefill_gpu_indices: list, decode_gpu_indices: list, nixl_proxy_url: str,
             model_name: str, peak_cap_prefill_w: float = None, peak_cap_decode_w: float = None,
             peak_window_s: float = 30.0, peak_recheck_interval_s: float = 1.0,
             j_per_prefill_token: float = 0.094, j_per_decode_token: float = 0.78,
             bytes_per_token: float = 272.0, reservation_hold_s: float = 1.0):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    budget_prefill, budget_decode = build_power_budgets(
        peak_cap_prefill_w, peak_cap_decode_w, peak_window_s)
    # DecodeByteEstimator only needed if at least one pool is capped -- matches
    # proxy_server.py's own "only build the estimator if the gate can actually act on it"
    # convention (make_app there: `DecodeByteEstimator() if budget is not None else None`).
    decode_estimator = (DecodeByteEstimator()
                         if (budget_prefill is not None or budget_decode is not None)
                         else None)

    async def handle_completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if decode_estimator is not None:
            token_ids = tokenizer.encode(body["prompt"])
            fallback_tokens = body.get("max_tokens", 128)
            expected_decode_tokens = decode_estimator.current_estimate_tokens(
                fallback_tokens, bytes_per_token)
            # estimate_marginal_energy_j is reused UNCHANGED for each leg -- zeroing one
            # token-count argument isolates the other leg's marginal energy (spec Sec 4.1):
            # estimate_marginal_energy_j(p, 0, jp, jd) == p * jp + 0 * jd == p * jp.
            prefill_marginal_j = estimate_marginal_energy_j(
                len(token_ids), 0, j_per_prefill_token, j_per_decode_token)
            decode_marginal_j = estimate_marginal_energy_j(
                0, expected_decode_tokens, j_per_prefill_token, j_per_decode_token)

            while dual_budget_would_exceed(
                    budget_prefill, peak_cap_prefill_w, prefill_marginal_j,
                    budget_decode, peak_cap_decode_w, decode_marginal_j):
                await asyncio.sleep(peak_recheck_interval_s)

            # Reserve on both pools immediately on admission (before any real power sample
            # could reflect this request's draw), release each after a short fixed delay via
            # a decoupled task -- exactly proxy_server.py's reservation-ledger pattern
            # (make_app's docstring there explains why releasing at request COMPLETION
            # instead regressed badly), just duplicated per pool.
            if budget_prefill is not None:
                budget_prefill.reserve(prefill_marginal_j)

                async def _release_prefill_after_delay():
                    await asyncio.sleep(reservation_hold_s)
                    budget_prefill.release(prefill_marginal_j)

                asyncio.create_task(_release_prefill_after_delay())

            if budget_decode is not None:
                budget_decode.reserve(decode_marginal_j)

                async def _release_decode_after_delay():
                    await asyncio.sleep(reservation_hold_s)
                    budget_decode.release(decode_marginal_j)

                asyncio.create_task(_release_decode_after_delay())

        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        try:
            response_bytes = await forward_to_nixl_proxy(nixl_proxy_url, body, resp)
        finally:
            if decode_estimator is not None:
                decode_estimator.record_completion(response_bytes)
        await resp.write_eof()
        return resp

    async def handle_health(request: web.Request) -> web.Response:
        # Bypasses handle_completions entirely -- readiness polling must NOT exercise the
        # gate/estimator (proxy_server.py's make_app docstring documents the real bug this
        # avoids: a completion-shaped readiness probe poisons decode_estimator's cold-start
        # EMA before real traffic begins).
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_post("/v1/completions", handle_completions)
    app.router.add_get("/health", handle_health)
    app["budget_prefill"] = budget_prefill
    app["budget_decode"] = budget_decode
    app["decode_estimator"] = decode_estimator
    return app


def run(prefill_gpu_indices: list, decode_gpu_indices: list, nixl_proxy_url: str,
        model_name: str, host: str, port: int, power_interval_s: float = 0.5,
        peak_cap_prefill_w: float = None, peak_cap_decode_w: float = None,
        peak_window_s: float = 30.0, peak_recheck_interval_s: float = 1.0,
        j_per_prefill_token: float = 0.094, j_per_decode_token: float = 0.78,
        bytes_per_token: float = 272.0, reservation_hold_s: float = 1.0) -> None:
    app = make_app(prefill_gpu_indices, decode_gpu_indices, nixl_proxy_url, model_name,
                    peak_cap_prefill_w, peak_cap_decode_w, peak_window_s,
                    peak_recheck_interval_s, j_per_prefill_token, j_per_decode_token,
                    bytes_per_token, reservation_hold_s)
    budget_prefill = app["budget_prefill"]
    budget_decode = app["budget_decode"]
    reader = NvmlPowerReader(prefill_gpu_indices + decode_gpu_indices)

    async def _on_startup(app):
        app["power_task"] = asyncio.create_task(
            power_poll_loop(reader, prefill_gpu_indices, decode_gpu_indices,
                             power_interval_s, budget_prefill, budget_decode))

    app.on_startup.append(_on_startup)
    web.run_app(app, host=host, port=port)
