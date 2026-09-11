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
  1. tokenize prompt -> token_ids                                    [unchanged]
  2. expected_decode_tokens = decode_estimator.current_estimate_tokens(
         fallback_tokens=body.get("max_tokens"), bytes_per_token)     [new]
  3. marginal_j = estimate_marginal_energy_j(len(token_ids),
         expected_decode_tokens, j_per_prefill_token, j_per_decode_token)  [new]
  4. while budget.would_exceed(cap_w, marginal_j):                    [new: admission gate]
         await asyncio.sleep(recheck_interval_s)
  5. replica_id = router.route(token_ids)                             [unchanged]
  6. dispatch, stream response (counting response_bytes as it streams) [new counting,
         otherwise unchanged]
  7. router.complete(); decode_estimator.record_completion(response_bytes)  [new call,
         otherwise unchanged]
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

**Revised 2026-09-11** after a v1 single-blended-constant design (`prompt_tokens *
j_per_token`, using `max_tokens` as the output-length proxy) was checked against real trace
data and found materially flawed: `max_tokens` is a ceiling, not a typical value — on the two
conditions this design targets, mean realized output was only ~33% of a 1024 cap. Since decode
costs far more energy per token than prefill (see calibration below), that 3x overestimate of
the *expensive* term would have made the gate defer far more than necessary, undermining the
"near-free at the right cap" result from §1. Replaced with a genuine two-term estimator and a
live decode-length estimate instead of the static cap:

```python
def estimate_marginal_energy_j(prompt_tokens: int, expected_decode_tokens: float,
                                j_per_prefill_token: float, j_per_decode_token: float) -> float:
    return prompt_tokens * j_per_prefill_token + expected_decode_tokens * j_per_decode_token
```

**Calibration.** A trial-level aggregate regression (fitting `energy ~= a*prefill_tokens +
b*decode_tokens` across already-collected trials, `check_prefill_decode_energy_regression.py`)
failed outright — idle power and trial duration are confounded with token counts, producing a
physically-impossible negative decode coefficient despite R²=0.99. Replaced with a dedicated
isolation-burst experiment (`orchestrate/eenergy/run_prefill_decode_calibration.sh`, analyzed
by `check_prefill_decode_calibration_result.py`): an idle baseline measured fresh before each
of 3 trials, a pure-prefill burst (`max_tokens=1`, whale-sized prompts), and a decode-heavy
burst (natural-length prompts, `max_tokens=1024`), each trial drawing different conversations.

Results, replicated 3x:
- **`j_per_decode_token`: robust.** 2.31–2.49 J/token across trials, CV≈3-8% depending on
  exactly how idle is subtracted. **Adopted value: 2.40 J/token** (mean).
- **`j_per_prefill_token`: not robust.** 0.011–0.068 J/token across trials, CV≈61-77%.
  Diagnosed cause: the prefill burst itself isn't in a steady state — trial 1 (first burst
  after model load) processes the same token count at ~1.6x the power and half the throughput
  of trial 2, consistent with GPU clocks not yet at steady boost state; even trials 2/3 (both
  presumably past that) still differ by ~26% from each other, so a warm-up burst might not
  fully resolve it either. **Adopted value: 0.068 J/token — the highest observed rate, not the
  mean.** This is a deliberate conservative-upper-bound choice, not a placeholder for a future
  fix: quantified impact shows it's the right tradeoff (see below), and getting
  `j_per_decode_token` right matters far more anyway.

**Why an imprecise prefill constant is acceptable, quantified rather than assumed:** total
contribution per request, not just the per-token rate, is what matters. For a typical
short-prompt request (~300 prefill tokens, ~330 decode tokens), prefill's full observed range
swings the estimate by ~14J against a ~790J decode contribution — under 2%, functionally
irrelevant. For a whale request (~14,000 prefill tokens), the same range swings the estimate by
~650J, comparable to decode's contribution — here it matters. Using the *highest* observed
rate rather than the mean biases specifically toward safety on exactly this whale-heavy case,
consistent with the rest of this design's over-estimate-rather-than-under-estimate philosophy
(under-estimating risks admitting past the cap; over-estimating only costs some avoidable
deferral). Tightening this constant via a warm-up-controlled recalibration is legitimate future
work if evaluation shows it matters in practice (see §6) — not blocking v1.

**Live decode-length estimate**, replacing the flawed `max_tokens` proxy:

```python
class DecodeByteEstimator:
    """Live EMA of realized response byte-length, updated as requests complete. Tracks BYTES,
    not tokens, to avoid a mid-stream tokenizer call on the streamed response -- converts to a
    token-count estimate via a calibrated bytes-per-token constant at the point of use."""

    def __init__(self, smoothing_alpha: float = 0.3):
        self.smoothing_alpha = smoothing_alpha
        self._estimate_bytes = None  # None until warm (no completion observed yet)

    def record_completion(self, response_bytes: int) -> None:
        ...  # first observation seeds the estimate; subsequent ones EMA-blend it

    def current_estimate_tokens(self, fallback_tokens: float, bytes_per_token: float) -> float:
        """Live estimate converted to tokens, or fallback_tokens (cold start, no completion
        observed yet) -- the request's own `max_tokens` is the fallback, matching this
        design's existing safe-direction-over-estimate philosophy for the warmup case
        specifically (once warm, the live EMA takes over and is no longer max_tokens-biased)."""
```

Fed by counting `len(chunk)` in `handle_completions`'s existing streaming loop (`async for
chunk in upstream.content.iter_any()`) — no new dependency, the proxy already sees every byte
of the response. `bytes_per_token` reuses this project's existing calibrated
`--chars-per-token` constant from `src/replay_sharegpt.py` (3.235), not a newly-invented
approximation.

Two explicit approximations remain, both still erring in the *safe* direction:

- Uses **raw prompt length** (`len(token_ids)`, known before routing), not the cache-discounted
  P-token `new_tokens` that `router.route()` computes internally. `route()` has real side
  effects (load_tracker, whale_tracker, cache_mirror) that assume its returned replica is
  actually about to be dispatched to — calling it speculatively before an admission check that
  might defer the request would corrupt that state. Raw prompt length avoids touching
  `Router` at all.
- The bytes-per-token conversion is a fleet-wide constant, not per-request-calibrated — a
  second-order approximation layered on top of an already-conservative prefill constant and a
  well-calibrated decode constant; not the focus of rigor here given the quantified impact
  above.

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
- `J_PER_PREFILL_TOKEN` — default 0.068 (conservative upper bound, §4.2).
- `J_PER_DECODE_TOKEN` — default 2.40 (calibrated mean, §4.2).
- `BYTES_PER_TOKEN` — default 3.235 (reused from `src/replay_sharegpt.py`'s existing
  `--chars-per-token` calibration).

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
- **Warm-up-controlled prefill recalibration.** §4.2's `j_per_prefill_token` is a conservative
  upper bound from an under-powered calibration (CV=61-77%, likely GPU-clock-ramp-confounded),
  not a tight estimate. Revisit only if live evaluation shows the whale-heavy case where this
  matters is common enough to be worth tightening.
- **Per-request-calibrated bytes-per-token conversion.** Reuses one fleet-wide constant
  (§4.2); a more precise, possibly per-model or per-request-shape calibration is future work.
- **Offline re-simulation of deferral's effect on the power trace.** Per discussion, going
  straight to a live implementation and hardware validation rather than building a more
  rigorous offline simulator first.

## 7. Testing plan

- **Unit tests** (hardware-free, matching this project's existing `router_core.py`/`scoring.py`
  convention): `PowerBudget` against synthetic `(t, power)` sequences with hand-computed
  expected `would_exceed` results at various caps/windows, including the empty-window
  fail-open case. `estimate_marginal_energy_j` (trivial, two-term). `DecodeByteEstimator`:
  cold-start fallback, EMA blending after the first observation, conversion to tokens. All
  testable without any async machinery or real hardware.
- **Live hardware validation**, once built: same discipline as the rest of this project — n=3-6
  trials, on the same conditions the spike already characterized (Heavy/CL-long, Heavy/Matched)
  at the cap/window combinations the spike suggests should be near-free (cap=2200-2400W,
  window=15-30s per condition — matching §4.3's default), measuring actual deferral rate,
  actual wait-time distribution, and confirming the realized peak (over the configured window)
  stays under cap. This is the first real signal on whether the approximations in §4.2 hold up
  outside the trace-driven estimate. A longer-window (closer to a real 15-min utility contract)
  validation pass is a natural follow-up once the short-window behavior is confirmed.
