# Peak-shaving admission control (no on-site storage)

*Status: design agreed 2026-09-11. Independent exploratory track, not currently tied to the
e-Energy paper's claims — see "Relationship to the e-Energy project" below. Spike that
motivated this: `scripts/eenergy/check_peak_shaving_regime.py` and
`check_peak_shaving_admission_prototype.py` (trace-driven, throwaway prototypes, kept for
reference, not part of this design).*

---

## 1. Problem motivation

If a utility (or facility contract) imposes a hard cap on peak power draw, and there is no
on-site energy storage (battery, capacitor bank) to smooth demand, the only lever available is
shaping *when and where compute happens*. This is a different problem from the e-Energy
project's ramp-rate/coincidence work: that project targets the *rate of change* of power
(grid-transient/frequency-regulation risk); this targets the *level*, specifically a hard
externally-enforced ceiling on power averaged over a billing window (demand-charge/capacity
risk).

**Key structural fact, confirmed by the spike, that makes a no-storage strategy viable at
all:** utility demand caps are virtually never billed on true instantaneous power — they're
billed on a rolling window average (commonly 15 minutes). That averaging window is a "virtual
battery" a no-storage system can exploit directly: it doesn't need to suppress every
instantaneous spike, only keep the trailing window's *average* under the cap.

**Two regimes, confirmed empirically (spike results, round_robin baseline traces):**

- **Sustained demand comfortably below the cap, bursts push it over instantaneously.**
  Windowing/smoothing alone resolves this with zero requests delayed. Heavy/CL-long
  (sustained, closed-loop): cap=2200-2400W, 15s window → 0 of 901 requests need deferral.
  Heavy/Matched at cap=2400W (21% margin above the trace's 1979W mean): 3.2% deferral rate,
  ~1.4s mean wait.
- **Sustained demand at or near the cap.** No routing/windowing scheme fixes this — some
  requests must wait or be shed. Heavy/Matched at cap=2200W (11% margin): 53.9% deferral rate,
  ~40s mean wait, confirming the wall is real and margin-dependent, not gradual.

This design targets the first regime by construction: it assumes the operator picks a cap with
reasonable margin (the spike suggests roughly 15-20%+ above the workload's own sustained mean
power) such that the admission gate's cost is genuinely small. It does not attempt to make an
infeasible cap tolerable — see §6 (non-goals).

## 2. Relationship to existing routing

**This is a new layer, not a replacement for the router.** The admission gate decides *whether
and when* a request is allowed to proceed; the existing `Router.route()` (any of its existing
policies) continues to decide *which replica* an admitted request goes to, completely
unmodified. This directly satisfies the stated requirement: compute/load balancing among
admitted requests keeps happening exactly as it does today.

**Recommended routing policy to pair this with: `drf_no_power`** (plain 2-resource DRF over
compute+load, no power term). Rationale, drawn from this project's own findings rather than
assumed: this session's ablations (random-tiebreak, lagged-elevation-count) both showed the
existing ramp-based power mechanisms' apparent value is not clearly attributable to correctly
reading or timing live power state, while `drf_no_power` is competitive-to-dominant on several
conditions and carries none of that fragility. Now that the admission gate is the component
actually enforcing the power constraint, there's no remaining reason to also reweight routing
decisions toward power — that would be redundant at best. This is a recommendation, not a hard
dependency: the gate composes with any existing policy unchanged.

## 3. Architecture

```
power_poll_loop (existing, extended)
        |
        v
   PowerBudget.record(t, fleet_power_w)   [new: power_budget.py]
        |
        v
handle_completions (proxy_server.py):
  1. tokenize prompt -> token_ids                          [unchanged]
  2. marginal_j = estimate_marginal_energy_j(len(token_ids))  [new]
  3. while budget.would_exceed(cap_w, marginal_j):          [new: admission gate]
         await asyncio.sleep(recheck_interval_s)
  4. replica_id = router.route(token_ids)                   [unchanged]
  5. dispatch, stream response, router.complete()            [unchanged]
```

The gate sits strictly *before* `router.route()`. `Router` and every existing scoring
function are untouched by this design.

## 4. Components

### 4.1 `power_budget.py` (new module)

```python
class PowerBudget:
    def __init__(self, window_s: float):
        ...
    def record(self, t: float, fleet_power_w: float) -> None:
        """Append a sample, evict samples older than window_s."""
    def would_exceed(self, cap_w: float, marginal_j: float) -> bool:
        """Projected trailing-window average power, including marginal_j spread over
        window_s, compared to cap_w. Empty/sparse window (startup) fails OPEN (returns
        False) -- matches this codebase's existing warmup convention (see the
        lagged-elevation-count ablation's buffer warmup in scoring.py)."""
```

Fed by extending the existing `power_poll_loop` in `proxy_server.py` to additionally call
`budget.record(t, sum(state.last_power_w for state in states))` each cycle — reuses the
already-running NVML polling, no new sampling loop.

### 4.2 Marginal-energy estimator

```python
def estimate_marginal_energy_j(prompt_tokens: int, j_per_token: float) -> float:
    return prompt_tokens * j_per_token
```

Deliberately simple for v1 (your call, per discussion): a single constant `j_per_token`,
defaulting to this project's existing whole-trial calibration (~1.3-1.5 J/token, from
`check_energy_per_token_*.py`), configurable via `J_PER_TOKEN`. Two explicit
approximations, both erring in the *safe* direction (over-estimate, more deferral than
strictly necessary, never under-estimate into a cap violation):

- Uses **raw prompt length** (`len(token_ids)`, known before routing), not the cache-discounted
  P-token `new_tokens` that `router.route()` computes internally. `route()` has real side
  effects (load_tracker, whale_tracker, cache_mirror) that assume its returned replica is
  actually about to be dispatched to — calling it speculatively before an admission check that
  might defer the request would corrupt that state. Raw prompt length avoids touching
  `Router` at all.
- Ignores the prefill-vs-decode cost split and doesn't know output length in advance — a
  single blended constant, not a physically precise model. Documented as the clear upgrade
  path if evaluation shows this estimator is too coarse (see §6).

### 4.3 Integration (`proxy_server.py`)

`handle_completions` gains the four numbered new/changed lines in §3's diagram. New optional
env vars, following this project's existing convention (unset = feature disabled, matching how
`assignment_log_path` and `RAMP_CEILING_PER_GPU` already work):

- `PEAK_CAP_W` — the enforced cap. Unset disables the gate entirely (zero behavior change).
- `PEAK_WINDOW_S` — rolling window duration. Defaults to 30s, matching the window sizes the
  spike actually validated (§1) — **not** the 900s a real utility contract would likely use.
  Widening toward a real demand window is real, unvalidated future work (a longer window
  should only make deferral *easier* per §1's logic, but that's an untested extrapolation,
  not something to assume); shipping an untested default would undercut the whole reason for
  going straight to live validation instead of building an offline simulator first (§6).
- `PEAK_RECHECK_INTERVAL_S` — how often a deferred request re-checks the budget (default ~1-2s,
  matching the spike's implicit resolution).
- `J_PER_TOKEN` — marginal-energy constant (default from existing calibration).

## 5. Error handling / edge cases

- **Startup / cold window:** fails open (admits), per §4.1 — avoids blocking all traffic before
  the fleet has produced enough samples to fill the window.
- **NVML read failures:** inherits whatever staleness/failure behavior `power_poll_loop`
  already has today; this design introduces no new risk here and doesn't attempt to harden it.
- **Fairness among deferred requests:** not strictly FIFO (concurrent `asyncio.sleep` wakeups
  aren't perfectly ordered). Accepted as a minor cost given the spike's wait-time numbers in
  the cap regime this design targets; revisit with an explicit queue (§6) only if evaluation
  shows it matters.

## 6. Explicit non-goals for v1 (YAGNI, not oversight)

- **Shedding / max-wait fallback.** No behavior for "waited too long" — per discussion, cap
  selection (§1's margin guidance) is the intended safety valve, not a runtime fallback.
  Revisit only if live evaluation shows real starvation at a cap worth using.
- **Explicit deferral queue / dispatcher** (Approach 2 from brainstorming). The simpler
  in-coroutine polling gate is what's being built; the queue is a documented fallback if
  fairness turns out to matter in practice.
- **Prefill/decode-aware energy estimator.** The constant-J-per-token estimator is what's being
  built; a more precise model is future work if v1's approximation proves too coarse in
  evaluation.
- **Offline re-simulation of deferral's effect on the power trace.** Per discussion, going
  straight to a live implementation and hardware validation rather than building a more
  rigorous offline simulator first.

## 7. Testing plan

- **Unit tests** (hardware-free, matching this project's existing `router_core.py`/`scoring.py`
  convention): `PowerBudget` against synthetic `(t, power)` sequences with hand-computed
  expected `would_exceed` results at various caps/windows, including the empty-window
  fail-open case. `estimate_marginal_energy_j` (trivial). Both testable without any async
  machinery.
- **Live hardware validation**, once built: same discipline as the rest of this project — n=3-6
  trials, on the same conditions the spike already characterized (Heavy/CL-long, Heavy/Matched)
  at the cap/window combinations the spike suggests should be near-free (cap=2200-2400W,
  window=15-30s per condition — matching §4.3's default), measuring actual deferral rate,
  actual wait-time distribution, and confirming the realized peak (over the configured window)
  stays under cap. This is the first real signal on whether the approximations in §4.2 hold up
  outside the trace-driven estimate. A longer-window (closer to a real 15-min utility contract)
  validation pass is a natural follow-up once the short-window behavior is confirmed.
