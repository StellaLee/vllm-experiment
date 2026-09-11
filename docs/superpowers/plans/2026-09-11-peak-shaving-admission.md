# Peak-Shaving Admission Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rolling-window energy-budget admission gate in front of the e-Energy router that defers dispatch when it would push the trailing-window average fleet power over a configured cap, leaving the existing routing logic (compute/load balance) completely unchanged for admitted requests.

**Architecture:** A new, hardware-free `power_budget.py` module (`PowerBudget` tracking, `estimate_marginal_energy_j` estimation) is wired into `proxy_server.py`'s existing request-handling coroutine as a pre-check before the existing `router.route()` call, and into the existing NVML power-poll loop as a feed. Everything is opt-in via new env vars (unset = feature fully disabled, zero behavior change).

**Tech Stack:** Python 3.10, aiohttp (async request handling), pytest (hardware-free unit tests), bash orchestration scripts, real 8×4090 hardware for final validation.

**Spec:** `docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md`

## Global Constraints

- `PowerBudget` and `estimate_marginal_energy_j` must be pure/hardware-free and unit-testable without a GPU, NVML, or network access — matches every existing module in `scripts/eenergy/router/` (router_core.py, scoring.py, ramp.py, etc.).
- An empty or single-sample `PowerBudget` window must fail **open** (`would_exceed` returns `False`) — never block all traffic at startup before real samples exist.
- The admission-gate marginal-energy estimate must use **raw prompt token count** (`len(token_ids)`, known before routing), never the cache-discounted P-token from `Router.route()` — calling `route()` speculatively has real side effects (load_tracker, whale_tracker, cache_mirror) that assume the returned replica is actually about to be dispatched to.
- No shedding/max-wait fallback, no explicit deferral queue, no prefill/decode-aware estimator, no offline re-simulation — v1 scope only, per spec §6.
- Every new env var defaults to preserving current behavior exactly when unset (`PEAK_CAP_W` unset → gate fully disabled).
- Do not test `make_app`/`run`/`handle_completions` directly — they require a real `AutoTokenizer.from_pretrained` model load, which is why the existing `test_eenergy_proxy_server.py` only tests pure helper functions (`build_replica_states`, `format_assignment_record`). Follow that precedent: factor any new decision logic into a pure, separately-testable function first.

---

### Task 1: `PowerBudget` rolling-window tracker

**Files:**
- Create: `scripts/eenergy/router/power_budget.py`
- Test: `tests/test_eenergy_power_budget.py`

**Interfaces:**
- Produces: `PowerBudget(window_s: float)` with `.window_s`, `.record(t: float, fleet_power_w: float) -> None`, `.would_exceed(cap_w: float, marginal_j: float) -> bool`. Used by Task 3 (fed by `power_poll_loop`) and Task 4 (read by `handle_completions`).

- [ ] **Step 1: Write the failing tests**

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from power_budget import PowerBudget  # noqa: E402


def test_would_exceed_false_with_fewer_than_two_samples():
    budget = PowerBudget(window_s=10.0)
    assert budget.would_exceed(cap_w=1000.0, marginal_j=100.0) is False
    budget.record(t=0.0, fleet_power_w=5000.0)  # single sample, way over any cap
    assert budget.would_exceed(cap_w=1000.0, marginal_j=0.0) is False


def test_would_exceed_true_when_mean_power_plus_marginal_exceeds_cap():
    budget = PowerBudget(window_s=10.0)
    budget.record(t=0.0, fleet_power_w=900.0)
    budget.record(t=1.0, fleet_power_w=900.0)
    # mean=900W, marginal 2000J over a 10s window = +200W -> projected 1100W > cap 1000W
    assert budget.would_exceed(cap_w=1000.0, marginal_j=2000.0) is True


def test_would_exceed_false_when_projection_stays_under_cap():
    budget = PowerBudget(window_s=10.0)
    budget.record(t=0.0, fleet_power_w=500.0)
    budget.record(t=1.0, fleet_power_w=500.0)
    # mean=500W, marginal 1000J over a 10s window = +100W -> projected 600W <= cap 1000W
    assert budget.would_exceed(cap_w=1000.0, marginal_j=1000.0) is False


def test_record_evicts_samples_older_than_window():
    budget = PowerBudget(window_s=10.0)
    budget.record(t=0.0, fleet_power_w=100.0)
    budget.record(t=5.0, fleet_power_w=100.0)
    budget.record(t=25.0, fleet_power_w=9000.0)  # window is now [15, 25]
    # both earlier samples (t=0, t=5) must be evicted -- only the t=25 sample remains, and
    # would_exceed must fail open (< 2 samples) rather than react to it alone
    assert budget.would_exceed(cap_w=1000.0, marginal_j=0.0) is False


def test_record_keeps_samples_within_window():
    budget = PowerBudget(window_s=10.0)
    budget.record(t=0.0, fleet_power_w=9000.0)
    budget.record(t=5.0, fleet_power_w=9000.0)
    budget.record(t=9.0, fleet_power_w=9000.0)  # all three within [t-10, t] = [-1, 9]
    assert budget.would_exceed(cap_w=1000.0, marginal_j=0.0) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_eenergy_power_budget.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'power_budget'`

- [ ] **Step 3: Write the implementation**

```python
"""Rolling-window fleet-power budget for the peak-shaving admission gate (see
docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md). Pure, hardware-free --
power_poll_loop (proxy_server.py) feeds real samples in; the admission gate in
handle_completions reads it before routing."""
from collections import deque


class PowerBudget:
    """Tracks a rolling window of fleet-aggregate power samples and answers whether admitting
    a request with a given marginal energy cost would push the trailing window's average
    power over a cap. Fails OPEN (would_exceed returns False) when the window has fewer than
    2 samples -- at startup, before power_poll_loop has produced enough history, blocking all
    traffic would be worse than the brief risk of an unenforced cap during warmup."""

    def __init__(self, window_s: float):
        self.window_s = window_s
        self._samples = deque()  # (t, fleet_power_w), oldest first

    def record(self, t: float, fleet_power_w: float) -> None:
        self._samples.append((t, fleet_power_w))
        cutoff = t - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def would_exceed(self, cap_w: float, marginal_j: float) -> bool:
        if len(self._samples) < 2:
            return False
        mean_power_w = sum(p for _, p in self._samples) / len(self._samples)
        projected_avg_w = mean_power_w + marginal_j / self.window_s
        return projected_avg_w > cap_w
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_eenergy_power_budget.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/eenergy/router/power_budget.py tests/test_eenergy_power_budget.py
git commit -m "eenergy: add PowerBudget rolling-window tracker for peak-shaving admission gate"
```

---

### Task 2: Marginal-energy estimator

**Files:**
- Modify: `scripts/eenergy/router/power_budget.py`
- Test: `tests/test_eenergy_power_budget.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `estimate_marginal_energy_j(prompt_tokens: int, j_per_token: float) -> float`. Used by Task 4's admission-gate check in `handle_completions`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eenergy_power_budget.py`:

```python
from power_budget import PowerBudget, estimate_marginal_energy_j  # noqa: E402  (replace prior import line)


def test_estimate_marginal_energy_j_scales_linearly_with_tokens():
    assert estimate_marginal_energy_j(prompt_tokens=100, j_per_token=1.4) == 140.0
    assert estimate_marginal_energy_j(prompt_tokens=0, j_per_token=1.4) == 0.0
```

(Replace the existing `from power_budget import PowerBudget  # noqa: E402` line at the top of
the test file with the two-name import shown above, rather than adding a second import line.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_eenergy_power_budget.py::test_estimate_marginal_energy_j_scales_linearly_with_tokens -v`
Expected: FAIL with `ImportError: cannot import name 'estimate_marginal_energy_j'`

- [ ] **Step 3: Write the implementation**

Append to `scripts/eenergy/router/power_budget.py`:

```python
def estimate_marginal_energy_j(prompt_tokens: int, j_per_token: float) -> float:
    """Marginal energy estimate for an admission-gate check, made BEFORE routing:
    prompt_tokens is the raw, un-cache-discounted prompt length (known immediately from the
    tokenizer), not the P-token new_tokens count Router.route() computes internally -- see
    docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md Sec 4.2 for why calling
    route() speculatively before an admission decision isn't safe (it has real dispatch-
    tracking side effects). Using raw prompt length is a deliberate, safe-direction
    over-estimate: a real cache hit would need less energy than this predicts, so this errs
    toward more deferral, never less, relative to the true cost."""
    return prompt_tokens * j_per_token
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_eenergy_power_budget.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/eenergy/router/power_budget.py tests/test_eenergy_power_budget.py
git commit -m "eenergy: add marginal-energy estimator for peak-shaving admission gate"
```

---

### Task 3: Feed fleet-aggregate power into the budget from the existing poll loop

**Files:**
- Modify: `scripts/eenergy/router/proxy_server.py`
- Test: `tests/test_eenergy_proxy_server.py`

**Interfaces:**
- Consumes: `PowerBudget.record(t, fleet_power_w)` from Task 1.
- Produces: `fleet_power_w(states: list) -> float`, and `power_poll_loop(states, reader, interval_s, budget=None)` (budget param added, default `None` = no-op, current behavior unchanged). Used by Task 4's `run()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eenergy_proxy_server.py`:

```python
from proxy_server import build_replica_states, format_assignment_record, fleet_power_w  # noqa: E402  (replace the existing import line)


def test_fleet_power_w_sums_across_replicas():
    specs = [dict(replica_id="r0", host="h", port=1, gpu_index=0, token_budget=100,
                   max_num_seqs=10, ramp_ceiling_w_per_s=50.0),
             dict(replica_id="r1", host="h", port=2, gpu_index=1, token_budget=100,
                   max_num_seqs=10, ramp_ceiling_w_per_s=50.0)]
    states = build_replica_states(specs)
    states[0].last_power_w = 300.0
    states[1].last_power_w = 250.0
    assert fleet_power_w(states) == 550.0


def test_fleet_power_w_zero_for_freshly_built_states():
    specs = [dict(replica_id="r0", host="h", port=1, gpu_index=0, token_budget=100,
                   max_num_seqs=10, ramp_ceiling_w_per_s=50.0)]
    states = build_replica_states(specs)
    assert fleet_power_w(states) == 0.0
```

(Replace the existing `from proxy_server import build_replica_states, format_assignment_record  # noqa: E402` line with the version above rather than adding a second import line.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_eenergy_proxy_server.py::test_fleet_power_w_sums_across_replicas -v`
Expected: FAIL with `ImportError: cannot import name 'fleet_power_w'`

- [ ] **Step 3: Write the implementation**

In `scripts/eenergy/router/proxy_server.py`, add this function near `format_assignment_record`
(after it, before `power_poll_loop`):

```python
def fleet_power_w(states: list) -> float:
    """Sum of each replica's most recently observed power draw -- the fleet-aggregate
    instantaneous power the peak-shaving admission gate's PowerBudget tracks. Pure/testable
    separately from power_poll_loop's async NVML polling."""
    return sum(state.last_power_w for state in states)
```

Then modify `power_poll_loop` (add the `budget` parameter and the recording call):

```python
async def power_poll_loop(states: list, reader: NvmlPowerReader, interval_s: float,
                           budget=None) -> None:
    while True:
        for state in states:
            power_w, ts = reader.read(state.config.gpu_index)
            update_ramp_state(state, power_w, ts)
        if budget is not None:
            budget.record(time.time(), fleet_power_w(states))
        await asyncio.sleep(interval_s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_eenergy_proxy_server.py -v`
Expected: PASS (all tests, including the 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add scripts/eenergy/router/proxy_server.py tests/test_eenergy_proxy_server.py
git commit -m "eenergy: feed fleet-aggregate power into an optional PowerBudget from the poll loop"
```

---

### Task 4: Admission gate in `handle_completions`

**Files:**
- Modify: `scripts/eenergy/router/proxy_server.py`
- Test: `tests/test_eenergy_proxy_server.py`

**Interfaces:**
- Consumes: `PowerBudget` (Task 1), `estimate_marginal_energy_j` (Task 2), `fleet_power_w`/updated `power_poll_loop` (Task 3).
- Produces: `build_power_budget(peak_cap_w=None, peak_window_s=30.0)`, updated `make_app(...)` and `run(...)` signatures (new keyword-only-by-convention params: `peak_cap_w`, `peak_window_s`, `peak_recheck_interval_s`, `j_per_token`), `app["budget"]`. Used by Task 5 (`run_router.py`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eenergy_proxy_server.py`:

```python
from proxy_server import (build_replica_states, format_assignment_record, fleet_power_w,  # noqa: E402
                           build_power_budget)


def test_build_power_budget_returns_none_when_cap_unset():
    assert build_power_budget(peak_cap_w=None) is None


def test_build_power_budget_returns_configured_budget_when_cap_set():
    budget = build_power_budget(peak_cap_w=2400.0, peak_window_s=15.0)
    assert budget is not None
    assert budget.window_s == 15.0
```

(Replace the import block at the top of the file with the version above rather than adding a
third import line.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_eenergy_proxy_server.py::test_build_power_budget_returns_none_when_cap_unset -v`
Expected: FAIL with `ImportError: cannot import name 'build_power_budget'`

- [ ] **Step 3: Write the implementation**

In `scripts/eenergy/router/proxy_server.py`, add the import and the new function, then update
`make_app` and `run`.

Add to the existing import block at the top:

```python
from power_budget import PowerBudget, estimate_marginal_energy_j
```

Add this function near `fleet_power_w`:

```python
def build_power_budget(peak_cap_w: float = None, peak_window_s: float = 30.0):
    """Returns a fresh PowerBudget if peak-shaving is enabled (peak_cap_w is not None), else
    None. Factored out of make_app so this construction decision is testable without needing
    a real tokenizer/model load (make_app itself is not unit tested -- see this file's other
    tests and the plan's Global Constraints)."""
    return PowerBudget(peak_window_s) if peak_cap_w is not None else None
```

Replace `make_app`'s signature and body (the changed/added lines only — everything else in the
function stays as-is):

```python
def make_app(states: list, policy: str, model_name: str, assignment_log_path: str = None,
             bs_source: str = "local", whale_token_threshold: int = WHALE_TOKEN_THRESHOLD,
             peak_cap_w: float = None, peak_window_s: float = 30.0,
             peak_recheck_interval_s: float = 1.0, j_per_token: float = 1.4):
    router = Router(states, policy, bs_source=bs_source,
                     whale_token_threshold=whale_token_threshold)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    by_id = {s.config.replica_id: s for s in states}
    budget = build_power_budget(peak_cap_w, peak_window_s)
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
            marginal_j = estimate_marginal_energy_j(len(token_ids), j_per_token)
            while budget.would_exceed(peak_cap_w, marginal_j):
                await asyncio.sleep(peak_recheck_interval_s)
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
    app["budget"] = budget
    return app
```

Replace `run`'s signature and body:

```python
def run(replica_specs: list, policy: str, model_name: str, host: str, port: int,
        power_interval_s: float = 0.5, assignment_log_path: str = None,
        bs_source: str = "local", bs_poll_interval_s: float = 0.5,
        whale_token_threshold: int = WHALE_TOKEN_THRESHOLD,
        peak_cap_w: float = None, peak_window_s: float = 30.0,
        peak_recheck_interval_s: float = 1.0, j_per_token: float = 1.4) -> None:
    states = build_replica_states(replica_specs)
    gpu_indices = [s.config.gpu_index for s in states]
    reader = NvmlPowerReader(gpu_indices)
    app = make_app(states, policy, model_name, assignment_log_path, bs_source=bs_source,
                    whale_token_threshold=whale_token_threshold, peak_cap_w=peak_cap_w,
                    peak_window_s=peak_window_s, peak_recheck_interval_s=peak_recheck_interval_s,
                    j_per_token=j_per_token)
    budget = app["budget"]

    async def _on_startup(app):
        app["power_task"] = asyncio.create_task(
            power_poll_loop(states, reader, power_interval_s, budget))
        if bs_source == "telemetry":
            app["bs_task"] = asyncio.create_task(
                bs_poll_loop(states, model_name, bs_poll_interval_s))

    app.on_startup.append(_on_startup)
    web.run_app(app, host=host, port=port)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_eenergy_proxy_server.py -v`
Expected: PASS (all tests, including the 2 new ones)

- [ ] **Step 5: Run the full eenergy test suite to confirm nothing else broke**

Run: `python3 -m pytest tests/test_eenergy_*.py -q`
Expected: PASS, all tests (no regressions in router_core.py/scoring.py, which are untouched)

- [ ] **Step 6: Commit**

```bash
git add scripts/eenergy/router/proxy_server.py tests/test_eenergy_proxy_server.py
git commit -m "eenergy: wire peak-shaving admission gate into handle_completions"
```

---

### Task 5: `run_router.py` env vars

**Files:**
- Modify: `scripts/eenergy/run_router.py`

**Interfaces:**
- Consumes: `run(...)`'s new `peak_cap_w`, `peak_window_s`, `peak_recheck_interval_s`, `j_per_token` params (Task 4).
- Produces: `ROUTER_PEAK_CAP_W`, `ROUTER_PEAK_WINDOW_S`, `ROUTER_PEAK_RECHECK_INTERVAL_S`, `ROUTER_J_PER_TOKEN` env vars. Used by Task 6 (`launch_router_experiment.sh`).

No new test for this task: `main()`'s env-var reads are plain `os.environ.get(...)` calls with
no parsing logic worth unit testing, matching this file's existing precedent — only
`_parse_replicas` (real parsing logic) has a test file (`tests/test_eenergy_run_router.py`);
`main()` itself does not.

- [ ] **Step 1: Update the module docstring's env var list**

In `scripts/eenergy/run_router.py`, add these lines to the docstring's `Env:` block, after the
existing `ROUTER_WHALE_TOKEN_THRESHOLD` entry:

```
  ROUTER_PEAK_CAP_W          default unset (peak-shaving admission gate disabled). Fleet-
                            aggregate power cap in Watts -- see
                            docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md.
  ROUTER_PEAK_WINDOW_S       default 30.0. Rolling window (seconds) the cap is averaged over.
  ROUTER_PEAK_RECHECK_INTERVAL_S  default 1.0. How often a deferred request re-checks the budget.
  ROUTER_J_PER_TOKEN         default 1.4. Marginal-energy constant (J/token) used by the
                            admission gate's pre-routing estimate.
```

- [ ] **Step 2: Update `main()`**

Replace `main()`'s body:

```python
def main() -> int:
    policy = os.environ["ROUTER_POLICY"]
    replica_specs = _parse_replicas(os.environ["ROUTER_REPLICAS"])
    model_name = os.environ["ROUTER_MODEL_NAME"]
    host = os.environ.get("ROUTER_HOST", "0.0.0.0")
    port = int(os.environ.get("ROUTER_PORT", "9000"))
    power_interval_s = float(os.environ.get("ROUTER_POWER_INTERVAL_S", "0.5"))
    assignment_log_path = os.environ.get("ROUTER_ASSIGNMENT_LOG") or None
    bs_source = os.environ.get("ROUTER_BS_SOURCE", "local")
    bs_poll_interval_s = float(os.environ.get("ROUTER_BS_POLL_INTERVAL_S", "0.5"))
    whale_token_threshold = int(os.environ.get("ROUTER_WHALE_TOKEN_THRESHOLD",
                                                 str(WHALE_TOKEN_THRESHOLD)))
    peak_cap_w_raw = os.environ.get("ROUTER_PEAK_CAP_W") or None
    peak_cap_w = float(peak_cap_w_raw) if peak_cap_w_raw is not None else None
    peak_window_s = float(os.environ.get("ROUTER_PEAK_WINDOW_S", "30.0"))
    peak_recheck_interval_s = float(os.environ.get("ROUTER_PEAK_RECHECK_INTERVAL_S", "1.0"))
    j_per_token = float(os.environ.get("ROUTER_J_PER_TOKEN", "1.4"))
    run(replica_specs, policy, model_name, host, port, power_interval_s, assignment_log_path,
        bs_source, bs_poll_interval_s, whale_token_threshold, peak_cap_w, peak_window_s,
        peak_recheck_interval_s, j_per_token)
    return 0
```

- [ ] **Step 3: Run the full eenergy test suite to confirm nothing broke**

Run: `python3 -m pytest tests/test_eenergy_*.py -q`
Expected: PASS, all tests

- [ ] **Step 4: Commit**

```bash
git add scripts/eenergy/run_router.py
git commit -m "eenergy: add ROUTER_PEAK_* env vars for the peak-shaving admission gate"
```

---

### Task 6: Forward peak-shaving env vars through the launch script

**Files:**
- Modify: `orchestrate/eenergy/launch_router_experiment.sh`

**Interfaces:**
- Consumes: `ROUTER_PEAK_CAP_W` etc. (Task 5).
- Produces: `PEAK_CAP_W`, `PEAK_WINDOW_S`, `PEAK_RECHECK_INTERVAL_S`, `J_PER_TOKEN` env vars for orchestration scripts. Used by Task 7.

No automated test for this task (shell orchestration scripts in this project are verified by
live runs, not unit tests — matches every existing `orchestrate/eenergy/*.sh` script).

- [ ] **Step 1: Add the new optional env vars near the existing ones**

In `orchestrate/eenergy/launch_router_experiment.sh`, add these lines right after the existing
`ASSIGNMENT_LOG=${ASSIGNMENT_LOG:-logs/eenergy_assignment_${POLICY}.csv}` line:

```bash
# Optional peak-shaving admission gate (unset PEAK_CAP_W = disabled, matching every other
# optional feature in this script). See
# docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md.
PEAK_CAP_W=${PEAK_CAP_W:-}
PEAK_WINDOW_S=${PEAK_WINDOW_S:-30.0}
PEAK_RECHECK_INTERVAL_S=${PEAK_RECHECK_INTERVAL_S:-1.0}
J_PER_TOKEN=${J_PER_TOKEN:-1.4}
```

- [ ] **Step 2: Forward them to run_router.py**

Replace the router-launch line:

```bash
echo "Starting router (policy=$POLICY) on port $ROUTER_PORT -> assignment log: ${ASSIGNMENT_LOG}"
ROUTER_POLICY="$POLICY" ROUTER_REPLICAS="$REPLICA_SPECS" ROUTER_MODEL_NAME="$MODEL" \
  ROUTER_PORT="$ROUTER_PORT" ROUTER_ASSIGNMENT_LOG="$ASSIGNMENT_LOG" \
  ROUTER_PEAK_CAP_W="$PEAK_CAP_W" ROUTER_PEAK_WINDOW_S="$PEAK_WINDOW_S" \
  ROUTER_PEAK_RECHECK_INTERVAL_S="$PEAK_RECHECK_INTERVAL_S" ROUTER_J_PER_TOKEN="$J_PER_TOKEN" \
  python3 scripts/eenergy/run_router.py &
ROUTER_PID=$!
```

(`ROUTER_PEAK_CAP_W=""` when `PEAK_CAP_W` is unset round-trips correctly through
`run_router.py`'s `os.environ.get("ROUTER_PEAK_CAP_W") or None` from Task 5 — an empty string
is falsy, so it becomes `None`, preserving "unset = disabled" end to end.)

- [ ] **Step 3: Commit**

```bash
git add orchestrate/eenergy/launch_router_experiment.sh
git commit -m "eenergy: forward peak-shaving env vars through launch_router_experiment.sh"
```

---

### Task 7: Live hardware validation battery

**Files:**
- Create: `orchestrate/eenergy/run_pergpu_peak_shaving_validation.sh`

**Interfaces:**
- Consumes: `PEAK_CAP_W`/`PEAK_WINDOW_S` (Task 6), `POLICY=drf_no_power` (existing, unchanged — see spec §2 for why this pairing is recommended).

No automated test — this is the live-hardware confirmation step itself, following the same
structure as `orchestrate/eenergy/run_pergpu_lagged_elevation.sh` and
`run_pergpu_random_power_tiebreak.sh`. n=3 trials per condition (this project's established
"quick check first" convention before deciding whether to expand to n=6), on the two
conditions the spike (`check_peak_shaving_admission_prototype.py`) already characterized, at
the cap/window combination the spike found near-free on both: cap=2400W, with each
condition's own validated window (15s for the sustained condition, 30s for the burstier one,
matching §7 of the spec exactly).

- [ ] **Step 1: Write the orchestration script**

```bash
#!/bin/bash
# First hardware validation of the peak-shaving admission gate
# (docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md). Pairs the gate with
# drf_no_power (per the spec's recommendation -- the admission gate now enforces the power
# constraint, so routing doesn't need to double up on it). cap=2400W is the combination the
# offline spike (check_peak_shaving_admission_prototype.py) found near-free on both
# conditions: 0/901 deferrals on Heavy/CL-long at a 15s window, 3.2% deferral rate (~1.4s mean
# wait) on Heavy/Matched at a 30s window. n=3 trials each (quick-check convention) -- expand to
# n=6 if this looks promising.
set -x
cd /root/pli/vllm-experiment
source /root/pli/venv-vllm023/bin/activate
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5"
POLICY=drf_no_power
OUTNAME=peak_shaving_validation
PEAK_CAP_W=2400

run_common_setup () {
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
  mkdir -p logs
}

wait_for_router () {
  for i in $(seq 1 120); do
    if curl -sf -X POST http://127.0.0.1:9100/v1/completions -H "Content-Type: application/json" \
       -d '{"model":"/data/pli/models/Qwen2.5-Coder-7B-Instruct","prompt":"hi","max_tokens":1}' \
       >/dev/null 2>&1; then
      echo "router ready after ${i} checks"
      return 0
    fi
    sleep 5
  done
}

teardown () {
  pkill -f "vllm.entrypoints" 2>/dev/null
  pkill -f "scripts/eenergy/run_router.py" 2>/dev/null
  pkill -f "power_logger.py" 2>/dev/null
  sleep 3
}

run_trial () {
  local PREFIX=$1
  local PEAK_WINDOW_S=$2
  local TRIAL=$3
  shift 3
  local HARNESS_ARGS=("$@")
  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL cap=${PEAK_CAP_W}W window=${PEAK_WINDOW_S}s ==="
  run_common_setup
  rm -f logs/${PREFIX}_launch_${OUTNAME}_t${TRIAL}.log

  POLICY=$POLICY N_REPLICAS=6 GPU_OFFSET=2 MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
    ROUTER_PORT=9100 RAMP_CEILING_PER_GPU="$RAMP_CEILING_PER_GPU" \
    PEAK_CAP_W="$PEAK_CAP_W" PEAK_WINDOW_S="$PEAK_WINDOW_S" \
    POWER_TRACE=logs/${PREFIX}_power_trace_${OUTNAME}_t${TRIAL}.csv \
    ASSIGNMENT_LOG=logs/${PREFIX}_assignment_${OUTNAME}_t${TRIAL}.csv \
    nohup bash orchestrate/eenergy/launch_router_experiment.sh > logs/${PREFIX}_launch_${OUTNAME}_t${TRIAL}.log 2>&1 &
  echo "launch script pid: $!"
  wait_for_router

  echo "=== running harness (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  timeout 900 python3 src/replay_sharegpt.py \
    --host 127.0.0.1 --port 9100 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
    --dataset data/sharegpt_v3.json --request-timeout 180 \
    --output logs/${PREFIX}_records_${OUTNAME}_t${TRIAL}.jsonl \
    "${HARNESS_ARGS[@]}" \
    2>&1 | tee logs/${PREFIX}_harness_${OUTNAME}_t${TRIAL}.log

  echo "=== tearing down (${PREFIX} policy=$OUTNAME trial=$TRIAL) ==="
  teardown
  echo "=== ${PREFIX} BATCH: policy=$OUTNAME trial=$TRIAL COMPLETE ==="
}

for TRIAL in 1 2 3; do
  run_trial closedloopheavylongpergpu 15.0 "$TRIAL" \
    --min-turns 1 --max-turns 1 --concurrency 32 --num-convs 900 --max-tokens 1024 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000
done

for TRIAL in 1 2 3; do
  run_trial openloopwhalelongoutmatchedpergpu 30.0 "$TRIAL" \
    --min-turns 1 --max-turns 1 --rate 10.7 --num-convs 750 --max-tokens 1024 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000
done

echo "=== PEAK_SHAVING_VALIDATION_BATCH COMPLETE ==="
```

- [ ] **Step 2: Commit the script**

```bash
git add orchestrate/eenergy/run_pergpu_peak_shaving_validation.sh
git commit -m "eenergy: add live hardware validation battery for the peak-shaving admission gate"
```

- [ ] **Step 3: Sync to the remote 8x4090 server and run the full test suite there**

```bash
rsync -avz scripts/eenergy/router/power_budget.py scripts/eenergy/router/proxy_server.py \
           scripts/eenergy/run_router.py \
           183.147.142.123:/root/pli/vllm-experiment/scripts/eenergy/router/ \
           183.147.142.123:/root/pli/vllm-experiment/scripts/eenergy/
rsync -avz orchestrate/eenergy/launch_router_experiment.sh \
           orchestrate/eenergy/run_pergpu_peak_shaving_validation.sh \
           183.147.142.123:/root/pli/vllm-experiment/orchestrate/eenergy/
rsync -avz tests/test_eenergy_power_budget.py tests/test_eenergy_proxy_server.py \
           183.147.142.123:/root/pli/vllm-experiment/tests/
ssh 183.147.142.123 "cd /root/pli/vllm-experiment && source /root/pli/venv-vllm023/bin/activate && python3 -m pytest tests/test_eenergy_*.py -q"
```

Expected: PASS, all tests, on the remote server (confirms the code that will actually run on
hardware matches what was tested locally).

- [ ] **Step 4: Launch the validation battery on the remote server (detached)**

```bash
ssh 183.147.142.123 "cd /root/pli/vllm-experiment && nohup bash orchestrate/eenergy/run_pergpu_peak_shaving_validation.sh > logs/peak_shaving_validation.driver.log 2>&1 & echo LAUNCHED_PID=\$!; disown"
```

Then verify independently via a fresh SSH connection (this project's established discipline —
the launching SSH session can appear to hang or drop without the detached remote process being
affected):

```bash
ssh -o ConnectTimeout=10 183.147.142.123 "pgrep -af 'run_pergpu_peak_shaving_validation'"
```

- [ ] **Step 5: Once complete, verify the cap held and deferral was near-zero as predicted**

For each condition/trial, confirm (a) the power trace's realized values stay close to the
2400W cap (allowing for the gap between the router's own 30s/15s budget window and
`power_logger.py`'s finer 50ms sampling cadence — brief instantaneous excursions above 2400W
are expected and consistent with the design, since the gate enforces a *windowed average*, not
an instantaneous ceiling), and (b) TTFT is not materially worse than the existing
`drf_no_power` baseline on the same conditions (logs/closedloopheavylongpergpu_records_drf_no_power_t*.jsonl and
logs/openloopwhalelongoutmatchedpergpu_records_drf_no_power_t*.jsonl, if present, or a
freshly-run drf_no_power-without-the-gate comparison trial otherwise) — the spike predicted
near-zero deferral at this cap/window combination, so TTFT should be largely unaffected.

---

## Self-review notes

- **Spec coverage:** §3 (architecture/data flow) → Tasks 3-4. §4.1 (PowerBudget) → Task 1.
  §4.2 (estimator) → Task 2. §4.3 (integration/env vars) → Tasks 4-6. §5 (fail-open) → Task 1's
  tests. §6 (non-goals) → deliberately not built anywhere in this plan. §7 (testing plan) →
  Tasks 1-2 unit tests + Task 7 live validation, including the §7-mandated cap/window
  combination.
- **Type consistency:** `PowerBudget.would_exceed(cap_w, marginal_j)` signature is identical
  everywhere it's called (Task 1 tests, Task 4's `handle_completions`). `estimate_marginal_energy_j(prompt_tokens, j_per_token)`
  likewise. `build_power_budget(peak_cap_w, peak_window_s)` matches its use in `make_app`.
- **Placeholder scan:** none found — every step has real, complete code.
