# Peak-shaving admission gate: design, calibration, and hardware validation

*Independent exploratory track, separate from the ramp-rate/coincidence work in
`findings/2026-08-31-eenergy-drf-lmetric-roundrobin-comparison.md`. Motivated by a real
constraint this project's routing work hadn't addressed: a hard utility/facility cap on peak
power draw, with no on-site energy storage available to smooth demand. Spec:
`docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md`. Plan:
`docs/superpowers/plans/2026-09-11-peak-shaving-admission.md`. All work on `main`, committed
incrementally throughout — see `git log --oneline --grep=peak-shaving` /
`--grep=reservation` / `--grep=bytes_per_token` for the exact commit sequence.*

## Part 1: Feasibility spike

**Key structural fact:** utility demand caps are billed on a rolling window average (commonly
15 minutes), not true instantaneous power — that averaging window is a "virtual battery" a
no-storage system can exploit directly.

Two regimes, confirmed with a trace-driven admission-only simulation against real historical
power traces (`scripts/eenergy/check_peak_shaving_regime.py`,
`scripts/eenergy/check_peak_shaving_admission_prototype.py`, both throwaway/reference only):

- Sustained demand comfortably below cap, bursts push it over instantaneously: windowing
  alone resolves it with zero deferral. Heavy/CL-long, cap=2200-2400W, 15s window → 0/901
  requests need deferral.
- Sustained demand at or near the cap: Heavy/Matched at cap=2400W (21% margin above the
  trace's 1979W mean) → 3.2% deferral rate, ~1.4s mean wait. At cap=2200W (11% margin) → 53.9%
  deferral rate, ~40s mean wait — confirming the wall is real and margin-dependent, not
  gradual.

## Part 2: Design — two-term prefill/decode estimator

Architecture: `PowerBudget` (rolling-window fleet-power tracker) gates admission in
`handle_completions`, strictly before `router.route()` — routing among admitted requests is
untouched (paired with `drf_no_power`, chosen because this session's ramp-rate ablations found
it competitive-to-dominant and free of the fragility found in the power-aware ramp mechanisms).

Marginal-energy estimate: `prompt_tokens * j_per_prefill_token + expected_decode_tokens *
j_per_decode_token`. `expected_decode_tokens` comes from `DecodeByteEstimator`, a live EMA
(α=0.3) of realized response byte-length from completed requests, converted to a token count
at the point of use — not a static `max_tokens` assumption (that was checked and rejected:
real output length averages only ~33% of a 1024-token cap on the two conditions this design
targets, an over-estimate that would have caused far more deferral than necessary).

## Part 3: Prefill/decode energy calibration

A trial-level aggregate regression (`scripts/eenergy/check_prefill_decode_energy_regression.py`,
fitting `energy ~= a*prefill_tokens + b*decode_tokens` across ~600 already-collected trials)
failed outright: R²=0.99 but a physically-impossible **negative** decode coefficient — idle
power and trial duration are confounded with token counts across heterogeneous trials.

Replaced with a dedicated isolation-burst experiment
(`orchestrate/eenergy/run_prefill_decode_calibration.sh`, analyzed by
`scripts/eenergy/check_prefill_decode_calibration_result.py`): idle baseline, a pure-prefill
burst (`max_tokens=1`, whale-sized prompts), a decode-heavy burst (natural prompts,
`max_tokens=1024`), replicated 3x with fresh conversations per trial.

- First attempt (idle baseline measured once, pre-experiment, 15s settle windows):
  `j_per_decode_token` tight (2.31-2.49 J/token, CV≈2.4-8%) but `j_per_prefill_token` wildly
  unstable (0.011-0.068 J/token, CV≈61-77%), decreasing monotonically trial-over-trial.
- Diagnosed cause, confirmed by inspecting raw duration/throughput/power per trial: the
  prefill burst itself wasn't in steady state — trial 1 (first burst after model load)
  processed the same ~227k tokens at 1608W/21,913 tok/s vs. trial 2's 871W/35,118 tok/s vs.
  trial 3's 978W/27,379 tok/s — a GPU-clock-ramp artifact, not idle-baseline staleness (fixing
  the idle-window measurement to use only its converged second half only modestly helped,
  CV 76.8%→61.1%).
- **Adopted values, not further chased given diminishing returns and quantified low impact**:
  `j_per_decode_token = 2.40` (mean, robust). `j_per_prefill_token = 0.068` (the *highest*
  observed rate, a deliberate conservative upper bound, not the mean) — justified because
  decode costs 37-115x more per token than prefill in every trial, so prefill's imprecision
  swings a typical short-prompt request's estimate by <2% but a whale request's by an amount
  comparable to its decode cost, where erring high is the safe direction.
- Separately: `bytes_per_token` (converts the live byte EMA to a token count) was initially
  set to `src/replay_sharegpt.py`'s 3.235 plain-text chars-per-token constant — wrong, see
  Part 5. Measured live against a running replica instead: 272.2 and 272.1 bytes/token across
  two independent streamed-response samples (real HTTP/SSE wire bytes, JSON event scaffolding
  included). Adopted `bytes_per_token = 272.0`.

## Part 4: Implementation

7-task TDD plan, all committed: `PowerBudget` (rolling window, fail-open under 2 samples),
`estimate_marginal_energy_j` + `DecodeByteEstimator`, feeding `power_poll_loop` into the
budget, wiring the gate into `handle_completions` (including streaming-response byte
counting), env vars through `run_router.py` and `launch_router_experiment.sh`, and the first
validation battery script. 278 tests pass locally, 275 remotely (the 3-test gap is
`test_eenergy_run_router.py` not being synced, unrelated).

## Part 5: First hardware validation — the `bytes_per_token` lockup

`orchestrate/eenergy/run_pergpu_peak_shaving_validation.sh` (`POLICY=drf_no_power`,
`PEAK_CAP_W=2400`), first attempt: 58 requests admitted then a complete stall for ~810 of 900
trial-seconds, while the power trace showed real fleet power at idle (~120W) the entire
stalled period — the gate was blocking despite the cap not being remotely threatened.

Root cause: `response_bytes` counts raw HTTP/SSE wire bytes (JSON scaffolding per token
included), not plain text — using the 3.235 plain-text constant inflated the apparent decode
token count by ~84x once the first real completion seeded the EMA, which then blocked every
subsequent admission regardless of real power, permanently starving the only thing (new
completions) that could have corrected the estimate. Fixed by measuring the real ratio live
(Part 3) and updating the default everywhere, plus a regression test
(`test_decode_byte_estimator_with_real_wire_bytes_per_token_gives_sane_token_estimate`).

## Part 6: Second validation — costs match predictions, but the cap isn't reliably held

Re-run after the fix, both conditions completed fully (no lockup):

| | Heavy/CL-long TTFT | Heavy/Matched TTFT |
|---|---|---|
| no gate (`drf_no_power`) | 0.433s mean | 3.590s mean |
| gated | 0.422s mean | 3.735s mean |

Matches the spike's prediction closely (near-free on CL-long, modest cost on Matched). TTFT
fully includes any gate hold time — confirmed directly from `src/replay_sharegpt.py`'s
`stream_request()`, where the timer starts at `urlopen()` and stops at the first non-empty
text chunk, so admission delay and replica queueing are indistinguishable to the client, both
counted.

**But checking the actual target metric (max 15s/30s rolling-window-average power, not just
TTFT) told a different story:**

| | Heavy/CL-long (15s window) | Heavy/Matched (30s window) |
|---|---|---|
| no gate | 2158.8/2177.5/2171.6W (0/3 over 2400W) | 2425.6/2436.3/2391.1W (**2/3 over**) |
| gated | 2163.0/2181.6/2199.5W (0/3 over) | 2405.7/2390.5/2408.3W (**2/3 over**) |

CL-long never threatened the cap either way — a clean but uninformative "free" result. Matched
is the real test, and the gate barely helped: worst-case overshoot only dropped from 2436W to
2408W (~1%), still 2/3 trials violating. Diagnosed cause: the gate only checks *real,
already-measured* fleet power (0.5s sampling), so several requests arriving close together
(Matched is open-loop, bursty, uncapped concurrency) can each pass against the same stale
reading before any of their own draw shows up — a real time-of-check-to-time-of-use race, not
an edge case.

## Part 7: Reservation ledger v1 — the fix regresses badly

`PowerBudget.reserve(marginal_j)` (called on admission) / `.release(marginal_j)` (originally
called at request **completion**, in `handle_completions`'s `finally` block) —
`would_exceed` accounts for `self._reserved_j + marginal_j`, not just the live mean, closing
the TOCTOU gap in principle.

Re-run on Heavy/Matched (`orchestrate/eenergy/run_pergpu_peak_shaving_matched_reservation_check.sh`):
severe regression. Mean TTFT 38.44s/46.24s/42.12s (vs. 3.590-3.735s baseline, ~10-12x worse),
and only 642/647/652 of 750 requests completed per trial (900s trial timeout hit). Diagnosed
cause: releasing only at completion held a reservation for a request's **entire processing
lifetime** (whale prefill + up to 1024 decode tokens can run for many seconds), not just the
brief real gap it needs to cover (~0.5-1s until the next power sample reflects it). Under
Matched's bursty, uncapped-concurrency arrivals, many simultaneous long-held reservations
piled up and starved admission almost entirely.

## Part 8: Reservation ledger v2 — time-based release

Fix: `release()` now fires after a short fixed delay (`reservation_hold_s`, default 1.0s = 2x
the 0.5s power-poll interval) via a `asyncio.create_task` spawned at admission time, decoupled
from the request's actual completion — `PowerBudget.reserve()`/`.release()` themselves are
unchanged, only the caller's timing was wrong.

Re-validated on Heavy/Matched, same script: **cap violations resolved** — max windowed-average
power 2349.9/2324.5/2355.5W, **0/3 over the 2400W cap** (vs. 2/3 both without the gate and with
the broken v1 reservation), and all 750/750/750 requests completed (no more drops). But real,
non-trivial cost: mean TTFT 10.04-10.65s (worse than both the 3.590s no-gate baseline and the
3.735s broken-reservation result), though p50 (2.92-3.61s) is actually fine — the mean is
dragged up by a genuine long tail, p95=47.87-50.25s. A correctly-enforcing gate has to make
some requests wait when the fleet is genuinely near capacity; the original under-enforcing gate's
"cheap" TTFT was partly an artifact of not actually enforcing.

## Part 9: Statistical power concern → targeted burst stress test

Raised directly: the vanilla gate holds only ~3% of requests (matching the spike's predicted
deferral rate), and cap violations were themselves small-magnitude — both rare events, making
n=3 trials on Matched's natural Poisson arrivals a thin sample to confidently attribute a
clean result to the fix rather than luck. Addressed by building a fast, deterministic,
directly-targeted test instead of relying on natural bursts to occur by chance:
`orchestrate/eenergy/run_pergpu_peak_shaving_burst_stress_test.sh` — closed-loop "herd"
concurrency (24 all-whale requests fired at t=0, no stagger) at a deliberately tight cap
(2000W, 15s window), two arms (`no_gate`, `gated`), 1 trial each (a mechanism check, not a
statistical claim).

## Part 10: Burst stress test v1 — confounded by a second, distinct estimator-poisoning bug

First run: `gated` (2185.7W max windowed avg) barely differed from `no_gate` (2216.1W), both
over cap, TTFT nearly identical between arms — 23 of 24 whale requests admitted within ~540ms
with no visible throttling at all.

Diagnosed cause: `wait_for_router()`'s readiness probe (`POST /v1/completions`,
`prompt="hi", max_tokens=1`) went through the *same* `decode_estimator` instance as real
traffic. Being the very first completion, its tiny response seeded the EMA with a near-zero
decode-length estimate before the real burst even began, systematically **under**-estimating
every subsequent whale request's marginal energy — the same category of bug as Part 5
(estimator poisoned by an unrepresentative sample) but in the opposite direction, and only
consequential when many admission decisions land within the first second or so of a trial
(Matched's spread-out natural arrivals self-correct within a few seconds; this test's design,
by construction, doesn't get that chance).

Fix: added a real `GET /health` endpoint to the router (`handle_health` in
`proxy_server.py`'s `make_app`) that bypasses `handle_completions` — and therefore the
gate/estimator — entirely. Updated `wait_for_router()` in all three peak-shaving orchestration
scripts to poll it instead of faking a completion.

## Part 11: Burst stress test v2 — clean result, partial success

Re-run after the health-check fix:

| | max windowed avg (15s) | vs. 2000W cap |
|---|---|---|
| no_gate | 2191.6W | +191.6W over |
| gated | 2151.5W | +151.5W over (~21% less overshoot) |

**The mechanism is now visibly, mechanistically working** — assignment-log timing shows
admissions paced into five distinct ~1s-spaced waves (7, 4, 6, 5, 2 requests), a clear
throttling signature, versus the confounded v1's all-23-in-540ms free-for-all. But the cap is
still modestly exceeded. Read as a genuine capacity limit, not a remaining bug: 24 simultaneous
15k-token whale prefills against a tight 2000W/15s budget is more aggregate demand than pure
admission-pacing (without waiting longer between waves or shedding requests outright — shedding
was an explicit non-goal from the design's first draft, spec §6) can fully absorb into one
window. Matches the original spike's own finding (Part 1): "sustained demand at or near the
cap — no routing/windowing scheme fixes this, some requests must wait or be shed." TTFT cost
was negligible here (gated mean 5.07s vs. no_gate 5.23s) — the ~4.5s of admission spacing is
dwarfed by the multi-second prefill/decode time these whale requests need regardless of the
gate.

## Conclusion / current status

- The core admission-gate mechanism, fully debugged (three real, distinct bugs found and
  fixed: `bytes_per_token` units mismatch, reservation held too long, health-check poisoning
  the estimator), demonstrably throttles concurrent admissions and reduces windowed-average
  cap overshoot — confirmed both on natural bursty traffic (Matched, 0/3 violations after the
  fix vs. 2/3 before) and on a deliberately engineered worst-case burst (measurable, visible
  wave-paced throttling).
- It is **not a hard guarantee** under sufficiently extreme, concentrated demand without
  shedding — a structural limit of admission-pacing-only designs, not a bug still to chase.
- There is a **real, non-trivial tail-latency cost** to correct enforcement under bursty
  conditions (Matched: p95 TTFT ~48-50s with the working reservation ledger) that any
  deployment decision needs to weigh explicitly — the original (broken, under-enforcing) gate
  looked cheaper specifically because it wasn't actually holding the line.
- Open, not yet done: n=6 replication of the fixed reservation-ledger result (current evidence
  is n=3 natural + n=1 targeted-burst); tuning `reservation_hold_s` to see if the p95 tail can
  be cut without reopening cap violations; whether shedding (explicitly out of scope for v1)
  is worth adding given the demonstrated hard limit.

## Data and repro

**Scripts** (all on `main`, `/Users/li/Documents/vllm-experiment`, mirrored to
`183.147.142.123:/root/pli/vllm-experiment`):
- Spike (throwaway): `scripts/eenergy/check_peak_shaving_regime.py`,
  `check_peak_shaving_admission_prototype.py`.
- Failed regression check: `scripts/eenergy/check_prefill_decode_energy_regression.py`.
- Calibration: `orchestrate/eenergy/run_prefill_decode_calibration.sh`,
  `scripts/eenergy/check_prefill_decode_calibration_result.py`.
- Core implementation: `scripts/eenergy/router/power_budget.py` (`PowerBudget`,
  `estimate_marginal_energy_j`, `DecodeByteEstimator`), `scripts/eenergy/router/proxy_server.py`
  (`handle_completions`, `handle_health`, `power_poll_loop`), `scripts/eenergy/run_router.py`,
  `orchestrate/eenergy/launch_router_experiment.sh`. Tests:
  `tests/test_eenergy_power_budget.py`, `tests/test_eenergy_proxy_server.py`.
- Validation batteries: `orchestrate/eenergy/run_pergpu_peak_shaving_validation.sh` (Parts
  5-6), `run_pergpu_peak_shaving_matched_reservation_check.sh` (Parts 7-8),
  `run_pergpu_peak_shaving_burst_stress_test.sh` (Parts 9-11).

**Log paths** (remote, `/root/pli/vllm-experiment/logs/`) — note several "before" datasets
were overwritten by later "after" reruns that reused the same `OUTNAME`/output paths (only the
numbers quoted above and the still-distinct driver logs survive for those; the final/fixed run
of each experiment has full raw data on disk):
- Calibration (final, fixed version only — earlier broken attempts were overwritten by
  reruns of the same script): `calibration_power_trace_prefill_decode_calibration.csv`,
  `calibration_phases_prefill_decode_calibration.txt`,
  `calibration_records_{prefill,decode}_burst_t{1,2,3}.jsonl`.
- Admission-gate validation: `{closedloopheavylongpergpu,openloopwhalelongoutmatchedpergpu}_
  {records,assignment,power_trace}_peak_shaving_validation_t{1,2,3}.{jsonl,csv}` (current data
  is the post-`bytes_per_token`-fix, successful run — Part 6's numbers); driver logs
  `peak_shaving_validation.driver.log` (v1, broken) and `_v2.driver.log` (v2, fixed) both
  still present.
- Reservation-ledger check: `openloopwhalelongoutmatchedpergpu_*_peak_shaving_reservation_
  check_t{1,2,3}.*` (current data is v2, time-based-release — Part 8's numbers); driver logs
  `peak_shaving_reservation_check.driver.log` (v1, broken) and `_v2.driver.log` (v2, fixed).
- Burst stress test: `burstallwhalepergpu_*_peak_shaving_burst_stress_{no_gate,gated}.*`
  (current data is v2, post-health-check-fix — Part 11's numbers); driver logs
  `peak_shaving_burst_stress.driver.log` (v1, confounded) and `_v2.driver.log` (v2, clean).
- Baseline for comparison (no gate, pre-existing from earlier this session):
  `{closedloopheavylongpergpu,openloopwhalelongoutmatchedpergpu}_*_drf_no_power_t{1,2,3}.*`.

**Remote access**: `ssh 183.147.142.123`, repo at `/root/pli/vllm-experiment`, venv at
`/root/pli/venv-vllm023`. `RAMP_CEILING_PER_GPU="2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,
7:359.5"` and `GPU_OFFSET=2 N_REPLICAS=6` throughout, matching every other eenergy battery
this project has run.
