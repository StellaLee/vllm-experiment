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
from power_budget import PowerBudget, estimate_marginal_energy_j, DecodeByteEstimator


def build_replica_states(replica_specs: list) -> list:
    """replica_specs: list of dicts with keys matching ReplicaConfig's fields."""
    return [ReplicaState(config=ReplicaConfig(**spec)) for spec in replica_specs]


def format_assignment_record(ts: float, replica_id: str, gpu_index: int,
                              new_tokens: int, raw_tokens: int,
                              share_compute: float = None, share_load: float = None,
                              d_chosen: float = None, min_d_available: float = None,
                              avoidable_threshold_violation: bool = None) -> str:
    """One CSV line for the per-request replica-assignment log: which physical GPU actually
    served each request (so post-hoc analysis can classify a request's power-pressure window
    by ITS OWN replica instead of the coarser fleet-wide any-GPU fallback), plus the P-token
    the router actually used for that decision alongside the raw prompt length -- the only
    place either number is observable, so this is what auditing the KV$-hit discount for
    honesty against a live cache-hit run has to read. share_compute/share_load (appended,
    optional -- None for backward compatibility with callers that don't pass them) are the
    CHOSEN candidate's own two shares at decision time, so post-hoc analysis can tell which
    resource was dominant for the actual pick without needing to reconstruct it from raw
    token/load state after the fact. d_chosen/min_d_available/avoidable_threshold_violation
    (also appended, optional) are Router's Theorem 4/5 diagnostic (router_core.py) -- whether
    THIS decision passed over an available safe candidate (D<=TAU) for an unsafe one
    (D>TAU) -- letting post-hoc analysis measure each policy's actual avoidable-threshold-
    violation rate directly, rather than only the proof-level claim about it."""
    sc = "" if share_compute is None else f"{share_compute:.6f}"
    sl = "" if share_load is None else f"{share_load:.6f}"
    dc = "" if d_chosen is None else f"{d_chosen:.6f}"
    dm = "" if min_d_available is None else f"{min_d_available:.6f}"
    av = "" if avoidable_threshold_violation is None else str(int(avoidable_threshold_violation))
    return f"{ts:.6f},{replica_id},{gpu_index},{new_tokens},{raw_tokens},{sc},{sl},{dc},{dm},{av}\n"


def build_power_budget(peak_cap_w: float = None, peak_window_s: float = 30.0):
    """Returns a fresh PowerBudget if peak-shaving is enabled (peak_cap_w is not None), else
    None. Factored out of make_app so this construction decision is testable without needing
    a real tokenizer/model load (make_app itself is not unit tested -- see this file's other
    tests and the plan's Global Constraints)."""
    return PowerBudget(peak_window_s) if peak_cap_w is not None else None


def fleet_power_w(states: list) -> float:
    """Sum of each replica's most recently observed power draw -- the fleet-aggregate
    instantaneous power the peak-shaving admission gate's PowerBudget tracks. Pure/testable
    separately from power_poll_loop's async NVML polling."""
    return sum(state.last_power_w for state in states)


async def power_poll_loop(states: list, reader: NvmlPowerReader, interval_s: float,
                           budget=None) -> None:
    while True:
        for state in states:
            power_w, ts = reader.read(state.config.gpu_index)
            update_ramp_state(state, power_w, ts)
        if budget is not None:
            budget.record(time.time(), fleet_power_w(states))
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
             bs_source: str = "local", whale_token_threshold: int = WHALE_TOKEN_THRESHOLD,
             peak_cap_w: float = None, peak_window_s: float = 30.0,
             peak_recheck_interval_s: float = 1.0, j_per_prefill_token: float = 0.068,
             j_per_decode_token: float = 2.40, bytes_per_token: float = 272.0,
             reservation_hold_s: float = 1.0):
    # reservation_hold_s=1.0 (2x the 0.5s default power_interval_s, giving power_poll_loop
    # at least one real sample to catch up): first live validation of the reservation ledger
    # released each reservation only at request COMPLETION, not after this short hold -- that
    # held long-running (whale-heavy) requests' reservations for their entire lifetime instead
    # of just the brief gap until the next real power sample reflects their draw. Under
    # Heavy/Matched's bursty, uncapped-concurrency arrivals, many simultaneous long-held
    # reservations piled up, causing a ~10x TTFT regression (mean 38-46s vs. 3.6-3.7s
    # baseline) and requests timing out entirely. The reservation only needs to cover the
    # brief real TOCTOU gap (concurrent admissions within one power_poll_loop interval), not
    # the request's full processing time -- see docs/superpowers/specs/2026-09-11-peak-
    # shaving-admission-design.md Sec 4.2.
    # bytes_per_token=272.0 is REAL WIRE BYTES per token (measured directly against this
    # project's actual vLLM streaming endpoint: 272.2 and 272.1 bytes/event across two
    # independent live samples, n_sse_events used as the token-count proxy) -- NOT
    # src/replay_sharegpt.py's 3.235 plain-text chars-per-token constant, which was wrongly
    # reused here in the first cut of this design. response_bytes (fed to DecodeByteEstimator
    # below) counts raw HTTP/SSE bytes including the JSON event scaffolding
    # ({"id":...,"choices":[{"text":...}],...} per token), which is ~84x larger than the
    # token's plain text alone. Using the plain-text constant caused a self-reinforcing
    # admission-gate lockup on first live validation: the first real completion seeded the
    # EMA with an ~84x-inflated apparent token count, which then made every subsequent
    # marginal-energy estimate blow past any reasonable cap regardless of real measured
    # power, permanently starving new completions (which are the only thing that could have
    # corrected the estimate). See docs/superpowers/specs/2026-09-11-peak-shaving-admission-
    # design.md Sec 4.2.
    router = Router(states, policy, bs_source=bs_source,
                     whale_token_threshold=whale_token_threshold)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    by_id = {s.config.replica_id: s for s in states}
    budget = build_power_budget(peak_cap_w, peak_window_s)
    decode_estimator = DecodeByteEstimator() if budget is not None else None
    assignment_log = None
    if assignment_log_path:
        # Truncate, not append: each run_router.py invocation is a fresh process (one per
        # condition run), so a stale log from an earlier attempt at the same policy must
        # not silently accumulate underneath this run's rows -- matches how the harness's
        # --output and power_logger.py's own CSV writer both start clean each invocation.
        assignment_log = open(assignment_log_path, "w")
        assignment_log.write("wall_time,replica_id,gpu_index,new_tokens,raw_tokens,share_compute,"
                              "share_load,d_chosen,min_d_available,avoidable_threshold_violation\n")
        assignment_log.flush()

    async def handle_completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        token_ids = tokenizer.encode(body["prompt"])
        if budget is not None:
            fallback_tokens = body.get("max_tokens", 128)
            expected_decode_tokens = decode_estimator.current_estimate_tokens(
                fallback_tokens, bytes_per_token)
            marginal_j = estimate_marginal_energy_j(
                len(token_ids), expected_decode_tokens, j_per_prefill_token, j_per_decode_token)
            while budget.would_exceed(peak_cap_w, marginal_j):
                await asyncio.sleep(peak_recheck_interval_s)
            # Reserve immediately on admission, before any real power measurement could
            # reflect this request's draw -- closes a real time-of-check-to-time-of-use gap
            # where several requests arriving close together could each pass against the same
            # stale reading. See power_budget.PowerBudget.reserve's docstring. Released after
            # a short fixed delay (NOT at request completion -- see make_app's docstring
            # comment on reservation_hold_s for why that regressed badly), decoupled from this
            # request's own lifetime via a separate fire-and-forget task.
            budget.reserve(marginal_j)

            async def _release_reservation_after_delay():
                await asyncio.sleep(reservation_hold_s)
                budget.release(marginal_j)

            asyncio.create_task(_release_reservation_after_delay())
        replica_id = router.route(token_ids)
        is_whale = router.last_is_whale
        target = by_id[replica_id].config
        if assignment_log:
            assignment_log.write(format_assignment_record(
                time.time(), replica_id, target.gpu_index,
                router.last_new_tokens, len(token_ids),
                router.last_share_compute, router.last_share_load,
                router.last_D_chosen, router.last_min_D_available,
                router.last_avoidable_threshold_violation))
            assignment_log.flush()

        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        url = f"http://{target.host}:{target.port}/v1/completions"
        response_bytes = 0
        try:
            async with ClientSession(timeout=ClientTimeout(total=None)) as session:
                async with session.post(url, json=body) as upstream:
                    async for chunk in upstream.content.iter_any():
                        response_bytes += len(chunk)
                        await resp.write(chunk)
        finally:
            router.complete(replica_id, is_whale)
            if decode_estimator is not None:
                decode_estimator.record_completion(response_bytes)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/completions", handle_completions)
    app["states"] = states
    app["router"] = router
    app["budget"] = budget
    app["decode_estimator"] = decode_estimator
    return app


def run(replica_specs: list, policy: str, model_name: str, host: str, port: int,
        power_interval_s: float = 0.5, assignment_log_path: str = None,
        bs_source: str = "local", bs_poll_interval_s: float = 0.5,
        whale_token_threshold: int = WHALE_TOKEN_THRESHOLD,
        peak_cap_w: float = None, peak_window_s: float = 30.0,
        peak_recheck_interval_s: float = 1.0, j_per_prefill_token: float = 0.068,
        j_per_decode_token: float = 2.40, bytes_per_token: float = 272.0,
        reservation_hold_s: float = 1.0) -> None:
    states = build_replica_states(replica_specs)
    gpu_indices = [s.config.gpu_index for s in states]
    reader = NvmlPowerReader(gpu_indices)
    app = make_app(states, policy, model_name, assignment_log_path, bs_source=bs_source,
                    whale_token_threshold=whale_token_threshold, peak_cap_w=peak_cap_w,
                    peak_window_s=peak_window_s, peak_recheck_interval_s=peak_recheck_interval_s,
                    j_per_prefill_token=j_per_prefill_token, j_per_decode_token=j_per_decode_token,
                    bytes_per_token=bytes_per_token, reservation_hold_s=reservation_hold_s)
    budget = app["budget"]

    async def _on_startup(app):
        app["power_task"] = asyncio.create_task(
            power_poll_loop(states, reader, power_interval_s, budget))
        if bs_source == "telemetry":
            app["bs_task"] = asyncio.create_task(
                bs_poll_loop(states, model_name, bs_poll_interval_s))

    app.on_startup.append(_on_startup)
    web.run_app(app, host=host, port=port)
