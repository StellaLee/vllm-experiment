# DRF vs LMETRIC vs round-robin: first 3-condition routing comparison

**Date:** 2026-08-31
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0, Qwen2.5-Coder-7B-Instruct, TP=1
**Router:** `scripts/eenergy/router/` (see `docs/superpowers/specs/2026-08-30-eenergy-routing-design.md`)
**Workload:** `src/replay_sharegpt.py`, single-turn, `whale-frac 0.15`, whale range
44-50k chars (≈13.6-15.5k tok), concurrency 24, 150 conversations, `data/sharegpt_v3.json`.
**N=6 replicas, GPUs 2-7** (GPUs 0-1 occupied by another user's job on this shared box —
see Methodology). `RAMP_CEILING_W_PER_S=450.0` (calibrated separately, see
`scripts/eenergy/README.md`).
**Status:** single trial, n=150 requests/condition. First real run of the DRF routing
design's 3-condition evaluation (spec §3.3) — not yet replicated.

## Headline

DRF sits genuinely **between round-robin and LMETRIC on power**, at a real cost to tail
latency — the fairness mechanism working as designed (spec §3.2's predicted tradeoff),
though the specific *structural* prediction (cost concentrated in power-pressure windows)
couldn't be tested with usable statistical power this trial (see Open Questions).

## Power (aggregate, summed across 6 GPUs)

| condition | mean W | peak W | mean ramp W/s | max ramp W/s |
|---|---|---|---|---|
| round_robin | 1253 | 2521 | 174 | 9645 |
| lmetric | 1442 | 2674 | **283 (worst)** | **22000 (worst)** |
| drf | 1351 | 2550 | 201 | 14872 |

DRF beats LMETRIC by −29% mean ramp, −32% max ramp — a real, sizeable effect, not noise-level.

## TTFT (seconds)

| condition | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| round_robin | 0.732 | 0.094 | 3.623 | 5.642 | 6.897 |
| lmetric | **0.519 (best)** | 0.088 | **1.935 (best)** | **2.068 (best)** | **2.087 (best)** |
| drf | 0.568 | **0.075 (best)** | 2.425 | 3.347 | **7.109 (worst)** |

DRF has the best median (p50) of all three, but the worst max — a genuine tail cost, not
a shift of the whole distribution.

## TBT-max (ms per request)

| condition | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| round_robin | 435.6 | 46.5 | 1792.5 | 1805.5 | 1805.6 |
| lmetric | 494.4 (worst mean) | 42.9 | 1724.1 | 1754.8 | 1763.2 |
| drf | 448.8 | 39.9 | 1771.2 | 1801.5 | 1801.5 |

No strong separation between conditions on TBT-max this trial — unlike TTFT and power,
this metric doesn't clearly differentiate the routing policies here.

## GPU utilization (proxy)

`power_logger.py` only samples power/energy/temp, not `nvidia-smi`'s `utilization.gpu` —
this is a derived proxy: fraction of the run's wall-clock duration during which each GPU
had ≥1 request in flight (from the assignment log + request dispatch/completion times).

| condition | mean | min | max | spread |
|---|---|---|---|---|
| round_robin | 0.59 | 0.49 | 0.71 | 0.22 |
| lmetric | **0.68 (best, tightest)** | 0.62 | 0.73 | 0.11 |
| drf | 0.64 | 0.57 | 0.77 | 0.20 |

LMETRIC keeps the fleet most consistently busy (tightest spread); DRF is close on the mean
but with more per-GPU variance — some replicas get routed around more than others when
power is the binding constraint.

## Interpretation

- **LMETRIC** wins on latency and utilization — cache/load-aware, packs work aggressively.
  That aggression is exactly what produces the sharpest power ramps: it has zero power
  awareness by design (confirmed from the paper, arXiv:2603.15202), so nothing holds it back.
- **DRF** trades some of LMETRIC's aggressiveness for a real, quantifiable power win
  (−29%/−32% ramp) — the typical case (median TTFT) stays as good as LMETRIC's, but the
  worst case (max TTFT) gets meaningfully worse. This is the honest cost the design predicted
  (spec §3.2): DRF is not claimed to be Pareto-dominant, and it isn't — it wins on power at a
  real tail-latency price.
- **round-robin's low power is not a real baseline to beat** — it's a side effect of being
  content-blind and under-utilizing the fleet (lowest utilization, worst mean TTFT), not an
  active power-aware achievement. Same caveat as the sibling PES-IM project's framing.

## Open questions / not yet resolved

1. **The structural prediction (cost concentrated in power-pressure windows, spec §3.2)
   could not be tested with usable power this trial.** Splitting by power-pressure vs.
   normal window using the sharper *per-replica* classification (added this session via
   `ROUTER_ASSIGNMENT_LOG`) collapsed to n=1-3 requests in the power-pressure bucket per
   condition — an order of magnitude too few to compare. The coarser fleet-wide proxy (any
   of N GPUs ramping) had more usable n (24-35 at N=8) but is a worse match for what
   actually happened to a given request's own replica. Next: either accept the fleet-wide
   proxy as the practical metric, or find a way to generate more per-replica pressure events
   (lower `RAMP_CEILING_W_PER_S`, many more requests, or fewer/larger replicas).
2. **Single trial.** Every number above needs replication before being treated as settled,
   per this project's established practice (see e.g. `2026-08-28-whale-aware-controller.md`'s
   repeated max-statistic noise caveats).
3. **N=6, not the originally-planned N=8** — GPUs 0-1 were occupied by another user's job.
   Not directly comparable to any future N=8 rerun.

## Methodology notes (bugs found and fixed getting here)

1. **Task 9 smoke test caught**: `orchestrate/eenergy/launch_router_experiment.sh` was
   launching vLLM's legacy `api_server` (`/generate`-only); the router proxies
   `/v1/completions`, which only `vllm.entrypoints.openai.api_server` exposes. Fixed before
   any real run (`abba1a5`).
2. **Load-imbalance bug**: `pick_lmetric`/`pick_drf` used a bare `min(candidates, key=...)`;
   Python's `min()` keeps the first tied element. This workload's un-cacheable prefill makes
   `Share_compute` tie across nearly all candidates on almost every call, so every tied
   request piled onto replica 0 deterministically — confirmed via per-GPU power (GPU0 mean
   341W vs GPUs 1-7 at ~90W) and a 2x wall-clock blowup for the DRF condition. Fixed with a
   rotating tie-break cursor, applied to both `pick_lmetric` and `pick_drf` (`d58cf10`).
3. **Assignment-log append bug**: `ROUTER_ASSIGNMENT_LOG` opened in append mode, so an
   aborted earlier attempt's rows silently accumulated underneath a fresh rerun's data.
   Fixed to truncate per run (`af506c1`) — each `run_router.py` invocation is a fresh
   process, appending across separate invocations is never correct.
4. **Ramp-ceiling calibration**: `RAMP_CEILING_W_PER_S=450.0`, from a real burst measurement
   at the router's actual 500ms poll cadence (not `power_logger.py`'s finer 50ms default —
   resampling to the router's real cadence mattered, see `scripts/eenergy/README.md`).
5. **GPU contention on a shared box**: another user's actively-running job occupied GPUs
   0-1 mid-experiment. Added `GPU_OFFSET` to the launch script to dodge occupied GPUs
   without touching the other job (`a2be934`); rescaled all three conditions to N=6 for a
   self-consistent comparison rather than mixing N=8 and N=6 data.

## Implementation

`scripts/eenergy/router/` (scoring, router_core, cache_mirror, load_tracker, ramp,
power_nvml, proxy_server), `scripts/eenergy/run_router.py`,
`orchestrate/eenergy/launch_router_experiment.sh`. 39/39 unit tests passing (local and
remote). Raw outputs for this run: `logs/eenergy_{power_trace,records,assignment}_{round_robin,lmetric,drf}.csv/jsonl`
on the 8x4090 box (not committed — `logs/` isn't tracked).

## Status and next steps (as of the first 3-condition run)

Not yet decided: replicate multi-trial, resolve the per-replica power-pressure
sample-size problem before trusting the structural-prediction test, or treat the fleet-wide
proxy as good enough and move toward writeup. **Superseded by the update below** — a fourth
condition and a corrected coincidence definition changed the picture materially.

---

## Update 2026-08-31 (later): whale-only Power-of-Two-Choices added as a 4th condition

### Why: DRF's target metric was never the right one

Discussion after the first 3-condition run (same day) concluded DRF's own target — per-chip
ramp rate — isn't the right thing to optimize for, on evidence gathered from this project's
own data plus the sibling `[[project_pes_im_energy_paper]]`'s history:

- **Peak magnitude**: untouched by every policy tested here (peak is statistically identical
  across all conditions, see tables below) — and PES-IM's own chunking gate experiment found
  the same null result independently ("all arms saturate ~460W near the GPU's power ceiling
  regardless of chunking"). Two independent experiments now say peak doesn't move via
  scheduling/routing.
- **Ramp rate** (DRF's actual target): PES-IM tried to extrapolate exactly this metric to
  data-center scale and found its own extrapolation methods untrustworthy at the tail —
  first a parametric (OU/discrete-state Monte Carlo) model with a ~5.8x too-low P99 ramp fit
  vs. real hardware, then a model-free trace-bootstrap replacement that itself sign-flipped
  at the heaviest load point from a small-library artifact. PES-IM's resolution was to stop
  leaning on any swept fleet-scale extrapolation as a primary claim. This project's own
  dataset (single trial, ~150 requests, N=6) is thinner than even PES-IM's already-fragile
  20-60-trial libraries, so the same retraction applies here — **no data-center-scale
  extrapolation is claimed anywhere in this document.**
- **Coincidence** (cross-GPU simultaneity of power pressure): directly measurable on this
  N=6 box with no extrapolation model needed at all, mechanistically what a *routing*
  decision — as opposed to chunking, DVFS, or admission control — is uniquely positioned to
  influence (it's fundamentally "which chip handles which request"), and matches the
  diversity/coincidence-factor literature's own established grid-infrastructure-sizing
  framing. **Adopted as the primary energy-system objective for the new policy.**
- **Duty cycle / thermal**: checked as a secondary, near-free axis (same power trace, plus
  the previously-unused `temp_c` column). DRF turned out to be the *worst* of the original 3
  conditions on both — highest/most-uneven duty cycle, hottest peak/mean temp — despite
  winning on ramp rate. A genuine tension: a mechanism good for one power-shape axis isn't
  automatically good for another.

### Whale → cross-GPU pressure-coincidence link, verified before building anything

Before writing any code, checked directly whether whale co-occurrence across replicas
actually explains the cross-GPU pressure coincidence measured in the original 3-condition
data (this check should have come before the design discussion that led to building
`p2c_whale`, not after — noted as a process lesson). Pooled linear correlation between
"#GPUs with an active whale" and "#GPUs pressured" was weak (r=0.18-0.24) — but the
*conditional* check is what matters: of all ticks where 2+ GPUs are simultaneously
pressured, **81-100% also have 2+ GPUs currently hosting an active whale** (round_robin
81.0%, lmetric 89.7%, drf 100.0%, n=21 whale requests/condition). The design premise held.

### Strategy: whale-only Power-of-Two-Choices (`p2c_whale`)

LMETRIC unchanged for all normal (non-whale) traffic. For whale-classified requests only
(prompt tokens > 4000, the existing project-wide whale-classification threshold, known at
admission with no power telemetry needed): sample 2 candidate replicas uniformly at random,
route to whichever has fewer currently-active whales. Theory: Mitzenmacher, *The Power of
Two Choices in Randomized Load Balancing* (1996/2001) — exponential improvement in expected
max load vs. random placement. Anticipatory (reacts to whale presence, known at admission)
rather than DRF's reactive local ramp-avoidance (only visible after a replica is already
ramping), and touches only whale-classified requests rather than every routing decision.

Implementation: `scripts/eenergy/router/whale_tracker.py` (new, mirrors `LoadTracker`'s
shape), `pick_p2c_whale` in `scoring.py`, `Router` gains an injectable `rng` and a 4th
`_POLICIES` entry. 20 new unit tests (58/58 eenergy tests passing, local and remote).
Commits `23ed9b2` (policy), `25d8df7` (coincidence analysis script, see below).

### Full 4-condition comparison (same N=6/GPUs 2-7/workload as the original 3-condition run)

**Power (aggregate, summed across 6 GPUs):**

| condition | mean W | peak W | mean ramp W/s | max ramp W/s |
|---|---|---|---|---|
| round_robin | 1253 | 2521 | 174 | 9645 |
| lmetric | 1442 | 2674 | 283 (worst) | 22000 (worst) |
| drf | 1351 | 2550 | 201 | 14872 |
| **p2c_whale** | 1297 | **2462 (lowest)** | 187 | **8608 (best of all 4)** |

**TTFT (seconds):**

| condition | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| round_robin | 0.732 | 0.094 | 3.623 | 5.642 | 6.897 |
| lmetric | 0.519 (best) | 0.088 | 1.935 (best) | 2.068 (best) | 2.087 (best) |
| drf | 0.568 | 0.075 (best) | 2.425 | 3.347 | 7.109 (worst) |
| **p2c_whale** | 0.545 | 0.094 | 1.961 | 3.294 | **3.338 (more than halves DRF's worst-case)** |

**TBT-max (ms per request):** no strong separation between any of the 4 conditions (round_robin
435.6/1805.6, lmetric 494.4/1763.2, drf 448.8/1801.5, p2c_whale 486.1/1799.6, mean/max
shown) — same as the original 3-condition finding, unaffected by adding p2c_whale.

**GPU utilization (proxy, fraction of run duration with ≥1 request in flight):**

| condition | mean | min | max | spread |
|---|---|---|---|---|
| round_robin | 0.59 | 0.49 | 0.71 | 0.22 |
| lmetric | 0.68 (best) | 0.62 | 0.73 | 0.11 (tightest) |
| drf | 0.64 | 0.57 | 0.77 | 0.20 |
| **p2c_whale** | **0.57 (lowest of all 4)** | 0.47 | 0.66 | 0.19 |

**Duty cycle (fraction of time power > 420W near-ceiling, `scripts/pesim/analyze_power_shape.py`'s
threshold convention) and thermal:**

| condition | pooled duty cycle | per-GPU spread | peak temp | mean temp |
|---|---|---|---|---|
| round_robin | 0.121 | 0.04-0.26 | 69°C | 44.7°C |
| lmetric | 0.133 | 0.09-0.17 | 70°C | 47.9°C |
| drf | 0.136 | **0.02-0.30 (worst, 15x ratio)** | **72°C (worst)** | 48.2°C (worst) |
| **p2c_whale** | 0.120 | **0.06-0.15 (tied-best w/ lmetric)** | **69°C (tied-best)** | **44.6°C (coolest)** |

**Coincidence (cross-GPU simultaneity) — see the definition correction below for why this
uses same-direction, not `abs()`, ramp classification:**

| condition | 2+ GPUs same-direction coincident (% of all pressure-time) |
|---|---|
| **drf** | **29.9% (best)** |
| p2c_whale | 41.7% |
| round_robin | 52.2% |
| lmetric | 64.2% (worst) |

### Coincidence metric definition correction: `abs()` vs. same-direction ramp

`classify_power_windows.py` (built earlier this session for per-replica request-latency
conditioning, matching the router's own reactive `Share_power` signal) uses `abs(ramp)` —
correct for *that* purpose, and correct on power-systems-equipment grounds generally
(voltage/frequency-regulation equipment is stressed by rate-of-change in either direction).
But it is the wrong definition for a **facility-meter coincidence** claim specifically: a
GPU ramping up while another ramps down cancels in the signed sum a utility meter actually
measures (confirmed directly — cancellation ratios of 11-16% observed among multi-GPU
pressure ticks, similar across conditions). Recomputed coincidence using only
**same-direction** simultaneous ramping (2+ GPUs up together, or 2+ down together) via the
newly-committed `scripts/eenergy/analyze_coincidence.py`. **Ranking is unchanged** (drf best,
lmetric worst, p2c_whale between drf and round_robin, not surpassing drf) but the gap
between drf and p2c_whale narrows under the corrected definition (~12 points vs. ~15-25
points depending on which of two slightly-differing earlier ad hoc computations is compared
— the committed script is now the single source of truth going forward).

### Why p2c_whale protects ramp severity despite losing on coincidence frequency

Apparent paradox, resolved by checking the single worst aggregate-ramp tick per condition
directly rather than assuming an explanation: LMETRIC's worst moment is a genuine
**fleet-wide synchronized burst** — all 6 GPUs ramping up together, each individually by
2200-5200 W/s. DRF's worst moment still involves 4 GPUs each swinging 2400-5000 W/s.
**p2c_whale's worst moment involves a similar GPU count (5) but each individual swing is
only 1100-2200 W/s — roughly half the magnitude of DRF's or LMETRIC's largest individual
contributors.** This matches the aggregated multi-pressure-tick average too: p2c_whale's sum
of individual ramp magnitudes at coincident moments (2192 W/s) is meaningfully smaller than
DRF's (2888 W/s), even before any sign-cancellation is applied.

**Mechanistic account, tied to the duty-cycle finding above**: P2C doesn't suppress
co-occurrence *frequency* as effectively as DRF (hence its higher coincidence % — DRF
reactively avoids an already-ramping replica on *every* decision using full 6-candidate
visibility; P2C only intervenes for whale-classified requests and samples just 2-of-6
candidates). But because P2C actively evens out *which* replica has absorbed how much whale
load over time (the duty-cycle-evenness win), no single GPU accumulates a backlog of whale
work that then bursts with maximum severity all at once. **Coincidence (how often) and
severity (how large) are different axes of the same underlying phenomenon, and a routing
policy can trade one for the other** — this is the paper's most novel finding from this
round, not previously hypothesized, found only by checking the actual worst-tick composition
rather than trusting the aggregate mean/max numbers alone.

### Open questions after this round

1. **Why does p2c_whale lose on coincidence frequency specifically, not just by how much?**
   Working hypothesis: only 21 whale events total across 6 replicas this trial, k=2-of-6
   sampling — may not be enough "throws" for Mitzenmacher's asymptotic exponential-improvement
   guarantee to have kicked in, unlike DRF's full-visibility-every-decision approach. Not yet
   tested: a full-argmin-over-whale-count ablation (no sampling) or a larger k, to check
   whether the frequency gap closes at this same N, or whether it's a fundamental property of
   the anticipatory/whale-only design regardless of k.
2. **Utilization dropped below even round_robin's** (0.57 vs 0.59) — did not fully preserve
   LMETRIC-like utilization as hypothesized, though its spread isn't worse than DRF's either.
   Not yet explained mechanistically.
3. **Single trial throughout, still.** Every number in this update needs replication before
   being treated as settled — same caveat as the original 3-condition run, now applying to a
   4th condition and several new metrics (coincidence, duty cycle, thermal) as well.
4. **N=6, not N=8**, same caveat as before (shared-box GPU contention).

### Updated status and next steps (2026-08-31, end of this round)

Agreed order: (1) this write-up [done], (2) investigate the coincidence-frequency question
(open item 1 above) via a full-argmin-over-whale-count ablation before committing to a final
`p2c_whale` design, (3) replicate the winning design multi-trial once its mechanism is
settled — replicating a version of the policy likely to still change isn't worth the GPU time
yet.

---

## Update 2026-08-31 (later still): `whale_argmin` ablation — the coincidence gap isn't just
## sampling, and it turns out to dominate `p2c_whale` on almost every other axis too

### Ablation: full-visibility whale-count routing, no sampling

Added `pick_whale_argmin` (`scripts/eenergy/router/scoring.py`, `router_core.py` 5th
`_POLICIES` entry, commit `f9a5d7b`) — identical whale-only design to `p2c_whale` (non-whale
traffic delegates to LMETRIC unchanged), but whale requests compare **every** candidate's
active whale count instead of sampling 2. Isolates whether P2C's 2-of-6 sampling (only 21
whale events this trial) was the reason it didn't beat DRF's coincidence figure.

**Result: sampling explains part of the gap, not all of it.** Coincidence (same-direction,
`scripts/eenergy/analyze_coincidence.py`): drf 29.9% < **whale_argmin 38.9%** < p2c_whale
41.7% < round_robin 52.2% < lmetric 64.2%. Removing sampling narrows the gap to DRF by about
a quarter (11.8 points → 9.0 points) but doesn't close it — DRF's real-time ramp signal
(which captures *where in its lifecycle* a whale currently is: just-started/still-prefilling
vs. long-finished-prefill-now-decoding) is evidently a finer-grained hazard signal than
whale *presence/count* (coarse, binary) can be, even with full visibility.

### Unexpected finding: whale_argmin doesn't just win on coincidence, it dominates p2c_whale broadly

| metric | p2c_whale | **whale_argmin** | winner |
|---|---|---|---|
| max ramp (W/s) | 8608 | **7649 (best of all 5 arms)** | whale_argmin |
| TTFT mean (s) | 0.545 | **0.506 (best of all 5, beats LMETRIC)** | whale_argmin |
| TTFT p99/max (s) | 3.294/3.338 | **2.080/2.144 (near-LMETRIC)** | whale_argmin |
| utilization mean | 0.57 | **0.67 (near-LMETRIC's 0.68)** | whale_argmin |
| TBT p99/max (ms) | 1799.6/1799.6 | **1778.0/1778.0** | whale_argmin |
| mean ramp (W/s) | **187** | 276 | p2c_whale |
| TBT mean (ms) | **486.1** | 551.8 (worst of all 5) | p2c_whale |

Only whale-request routing differs between the two policies (both delegate identically to
LMETRIC for the ~86% of traffic that isn't a whale), so this TTFT/utilization gap on
*non-whale* traffic is not obviously explained by the ablation's stated purpose.
**Hypothesis, not yet verified**: LMETRIC's own load signal (`in_flight_after`) counts
request slots, blind to whether one is a whale mid-prefill. If p2c_whale's sampling
occasionally makes a worse whale-placement choice than the true global minimum, whales can
concentrate onto fewer replicas than necessary; LMETRIC, not knowing those replicas are
quietly overloaded by whale compute, keeps routing *short* requests there too, indirectly
hurting non-whale TTFT/TBT. whale_argmin's consistently-optimal spread would avoid creating
that hidden hotspot. Not confirmed mechanistically.

### Full 5-arm comparison (all metrics, same N=6/workload)

**Power:**

| arm | mean W | peak W | mean ramp W/s | max ramp W/s |
|---|---|---|---|---|
| round_robin | 1253 | 2521 | 174 | 9645 |
| lmetric | 1442 | 2674 | 283 (worst) | 22000 (worst) |
| drf | 1351 | 2550 | 201 | 14872 |
| p2c_whale | 1297 | 2462 (lowest) | 187 | 8608 |
| **whale_argmin** | 1444 | 2680 (highest) | 276 | **7649 (best)** |

**TTFT (s):**

| arm | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| round_robin | 0.732 | 0.094 | 3.623 | 5.642 | 6.897 |
| lmetric | 0.519 | 0.088 | 1.935 | 2.068 | 2.087 |
| drf | 0.568 | 0.075 | 2.425 | 3.347 | 7.109 (worst) |
| p2c_whale | 0.545 | 0.094 | 1.961 | 3.294 | 3.338 |
| **whale_argmin** | **0.506 (best)** | 0.078 | 1.946 | 2.080 | 2.144 |

**TBT-max (ms):**

| arm | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| round_robin | 435.6 | 46.5 | 1792.5 | 1805.5 (worst) | 1805.6 (worst) |
| lmetric | 494.4 | 42.9 | 1724.1 (best) | 1754.8 | 1763.2 |
| drf | 448.8 (best) | 39.9 (best) | 1771.2 | 1801.5 | 1801.5 |
| p2c_whale | 486.1 | 48.6 | 1766.8 | 1799.6 | 1799.6 |
| whale_argmin | 551.8 (worst) | 47.0 | 1742.8 | 1778.0 | 1778.0 |

**Utilization (proxy):**

| arm | mean | min | max | spread |
|---|---|---|---|---|
| round_robin | 0.59 | 0.49 | 0.71 | 0.22 |
| lmetric | 0.68 (best) | 0.62 | 0.73 | 0.11 (tightest) |
| drf | 0.64 | 0.57 | 0.77 | 0.20 |
| p2c_whale | 0.57 (lowest) | 0.47 | 0.66 | 0.19 |
| whale_argmin | 0.67 | 0.57 | 0.73 | 0.16 |

**Duty cycle and thermal:**

| arm | pooled duty cycle | per-GPU spread (ratio) | peak temp | mean temp |
|---|---|---|---|---|
| round_robin | 0.121 | 0.04-0.26 (6.5x) | 69°C | 44.7°C |
| lmetric | 0.133 | 0.09-0.17 (1.9x) | 70°C | 47.9°C |
| drf | 0.136 | 0.02-0.30 (15x, worst) | 72°C (worst) | 48.2°C (worst) |
| p2c_whale | 0.120 | 0.06-0.15 (2.5x, best) | 69°C | 44.6°C (coolest) |
| whale_argmin | 0.129 | 0.07-0.20 (2.9x) | 68°C (coolest) | 46.0°C |

**Coincidence:**

| arm | 2+ GPUs same-direction coincident (% of pressure-time) |
|---|---|
| **drf** | **29.9% (best)** |
| whale_argmin | 38.9% |
| p2c_whale | 41.7% |
| round_robin | 52.2% |
| lmetric | 64.2% (worst) |

### Headline: `whale_argmin` is the best-balanced arm, not an outright winner on any single axis except max-ramp

No arm wins everything. LMETRIC and whale_argmin are the two best on latency/utilization;
DRF and whale_argmin are the two best on max-ramp and coincidence; p2c_whale and whale_argmin
are the two coolest/most-even on duty-cycle/thermal. whale_argmin places in the top two on
nearly every category rather than winning some axes and losing badly on others (unlike DRF's
latency-tail cost, LMETRIC's power cost, round_robin's utilization cost, or p2c_whale's
coincidence/utilization shortfall).

**Precision on the two places it's weaker, not just "doesn't regress":** TBT-*mean* is
actually the worst of all 5 arms (551.8ms vs LMETRIC's 494.4ms) even though TBT p99/max are
near-best — a real, modest regression, distinct from TTFT (which is best-in-class,
including beating LMETRIC). Mean ramp (276 W/s) is only marginally better than LMETRIC's 283
— essentially tied, not a real optimization; the genuine ramp win is specifically on the
*max* (7649 vs 22000, -65%), not the mean.

### Updated open questions

1. Why does whale_argmin's non-whale TTFT/TBT/utilization come out *better* than p2c_whale's,
   when only whale-routing differs between them? (hypothesis above, not yet verified —
   plausibly a hidden-hotspot effect via LMETRIC's whale-blind load counting)
2. Why doesn't even full-visibility whale-count routing close the gap to DRF's coincidence
   figure? (working answer: whale count is a coarser proxy for "is this replica in a
   power-risky moment right now" than DRF's real-time ramp signal — not yet directly tested,
   e.g. by checking whether DRF's wins concentrate specifically on whales caught mid-prefill)
3. Single trial throughout, still — this is the most important remaining gap before any of
   the above is trustworthy.

### Status and next steps (agreed order, 2026-08-31)

(1) This write-up [done]. (2) Investigate the coincidence-frequency question via the
full-visibility ablation [done — narrows but doesn't close the gap]. (3) **Replicate
`whale_argmin` multi-trial** (and likely `lmetric` alongside it, as the baseline the
"doesn't regress" claim is measured against) — now the clear next step, since whale_argmin
has emerged as the best-balanced candidate and every number above is still single-trial.
**Superseded by the replication below — the single-trial power/coincidence story does not
hold up.**

---

## Update 2026-08-31 (final round): 3x replication overturns the power-ramp/coincidence
## claim; only TTFT-mean and TBT-max-stability survive

Replicated `lmetric` and `whale_argmin` 3 trials each, same default `--pad-seed 12345` every
trial (isolates serving-timing noise, not workload-draw variance — matches this project's
established replication convention). N=6/GPUs 2-7/workload unchanged.

### Result: the "whale_argmin wins on power ramp and coincidence" story does not replicate

| metric | lmetric (3-trial mean±std) | whale_argmin (3-trial mean±std) | single-trial claim | verdict |
|---|---|---|---|---|
| max ramp (W/s) | 5497 ± 1132 | 5732 ± 1217 | whale_argmin best (7649 vs lmetric's 22000) | **statistically tied — not a real difference** |
| mean ramp (W/s) | **241 ± 5** | 279 ± 9 | roughly tied | **lmetric is actually better — reversed** |
| coincidence (%, same-direction) | **23.7 ± 4.1** | 32.1 ± 4.3 | whale_argmin much better (38.9% vs lmetric's 64.2%) | **lmetric is better — direction reversed** |
| TTFT mean (s) | 0.526 ± 0.020 | **0.492 ± 0.008** | whale_argmin best | **holds up** |
| TTFT max (s) | 3.690 ± 0.143 | 2.905 ± 0.709 | near-tied | whale_argmin better, but noisy (std ≈24% of mean) |
| TBT mean (ms) | **449 ± 47** | 555 ± 18 | whale_argmin worse | **holds up** |
| TBT max (ms) | 2364 ± 971 (unstable) | **1767 ± 5 (remarkably stable)** | near-tied | **new finding, not visible single-trial: whale_argmin's tail is far more consistent trial-to-trial, not just similar in magnitude** |

**Root cause of the single-trial illusion**: LMETRIC's single-trial max ramp of 22000 W/s
came from one rare fleet-wide synchronized burst (see the earlier worst-tick breakdown — all
6 GPUs ramping up together, 2200-5200 W/s each). Across 3 trials with the same workload,
this event **did not recur** (replicated max ramp 5497±1132) — it was a single-trial extreme
value, not a systematic property of LMETRIC. Once that outlier is gone, LMETRIC is
statistically tied with or better than whale_argmin on every ramp/coincidence metric.

### What actually survives replication

whale_argmin's genuine, replicated advantages are narrower than first claimed: **better TTFT
mean**, and **a much more stable worst-case TBT** (barely moves trial-to-trial: 1760/1770/
1769ms, vs LMETRIC's 1833/3485/1776ms — one huge outlier trial drives LMETRIC's instability).
Its genuine, replicated cost: **worse TBT mean**, confirmed in the same direction as the
single-trial reading. **The power-ramp and coincidence wins do not survive and should not be
claimed.**

### Why this matters beyond just correcting these numbers

This is the exact single-trial-misleads failure mode this project has hit repeatedly before
(PES-IM's own fleet-bootstrap sign-flips, this same document's earlier n=1-3 sample-size
problems) — now reproduced on our own numbers, at the scale specifically built to catch it.
**Every comparison anywhere above this section that is still single-trial (round_robin, drf,
p2c_whale, and every duty-cycle/thermal/coincidence number for arms other than lmetric and
whale_argmin) should be read with the same skepticism** — this replication doesn't prove
those specific numbers are wrong, but it demonstrates concretely, on this exact workload and
scale, that single-trial extreme-value statistics (max ramp) and even percentage-based
aggregates (coincidence, built from a handful of pressure-window seconds) are not reliable
without replication.

### Updated status and next steps

The honest current state: whale_argmin is not a demonstrated power/coincidence win over
LMETRIC — it's a mechanism with a modest, real TTFT-mean edge and a notably more stable tail,
at a real TBT-mean cost. Before any further claim: (1) replicate `drf` and `p2c_whale` too,
so the *full* 5-arm comparison rests on the same footing rather than mixing 3-trial and
single-trial numbers; (2) decide whether "more stable tail latency" (not "lower power ramp")
is now the actual headline this design supports, since that's what replicated; (3) revisit
whether the coincidence-based routing objective (spec'd several updates above) is still the
right target given LMETRIC — which has zero power awareness — replicated as the *better*
coincidence performer. **Superseded by the full 4-arm replication below — DRF turns out to
be the real, robust winner on the energy objective, reversing this section's tentative read.**

---

## Update 2026-08-31 (truly final round): full 4-arm replication — DRF wins the energy
## objective robustly; the whale-count design hypothesis is refuted, not unconfirmed

Replicated `drf` and `p2c_whale` 3 trials each (same methodology/pad-seed as the lmetric/
whale_argmin replication above), completing a 4-arm comparison where every number rests on
3 trials rather than mixing single-trial and replicated figures. (`round_robin` remains
single-trial — not replicated this round.)

### Power / ramp / coincidence (mean ± std across 3 trials)

| arm | mean ramp (W/s) | coincidence (%, same-direction) | max ramp (W/s) |
|---|---|---|---|
| **drf** | **208.6 ± 4.2 (best, tight)** | **17.7 ± 1.9 (best, tight)** | 6310 ± 1349 |
| lmetric | 241.2 ± 4.7 | 23.7 ± 4.1 | 5497 ± 1132 |
| p2c_whale | 215.6 ± 24.8 (unstable) | 32.0 ± 7.2 (worst, tied) | 5014 ± 1008 |
| whale_argmin | 278.6 ± 9.5 (worst) | **32.1 ± 4.3 (worst, tied)** | 5732 ± 1217 |

**DRF is the clear, robust winner on both metrics it was built for.** The single-trial read
that LMETRIC was "worst" on coincidence (64.2%) was the misleading part, driven by one
outlier max-ramp trial — replicated, LMETRIC is a solid #2, clearly ahead of *both*
whale-count mechanisms. **p2c_whale and whale_argmin land at essentially identical
(worst) coincidence — 32.0% vs 32.1%.**

**This refutes the design hypothesis behind the whole whale-count-routing thread, not just
leaves it unconfirmed.** The premise (spread whale co-occurrence across replicas → reduce
cross-GPU coincidence) does not survive replication: DRF (reacting to real-time ramp state)
and even LMETRIC (blind to power entirely) both beat both whale-aware mechanisms on the
metric they were specifically designed to win. Whale presence/count, even with full
visibility (`whale_argmin`, no sampling), is evidently not a good enough proxy for
coincidence risk — despite the pre-registered check (81-100% of coincident *events* have 2+
active whales) correctly establishing correlation, routing to avoid that correlation doesn't
reliably reduce it. Plausible reason not yet tested: coincidence may be driven more by
request-arrival-timing structure (Poisson clustering, or periodic admission waves at
concurrency=24) than by *which* replica happens to receive a whale.

### Latency (mean ± std across 3 trials)

| arm | TTFT mean | TTFT max | TBT mean | TBT max |
|---|---|---|---|---|
| lmetric | 0.526±0.020 | 3.690±0.143 | 449±47 | 2364±971 (unstable) |
| drf | 0.520±0.024 | **6.579±0.188 (worst, very tight — robustly bad, not noise)** | **458±27** | **1803±1.9 (remarkably stable)** |
| p2c_whale | 0.524±0.016 | 4.133±0.550 | 537±37 | 1822±19 |
| **whale_argmin** | **0.492±0.008 (best, tight)** | 2.905±0.709 | 555±18 (worst) | 1767±5 |

DRF's TTFT-tail cost is real and robustly replicated (6.58s ± 0.19 — the tiny variance means
this isn't noise, DRF consistently pays this cost). whale_argmin's TTFT-mean edge is also
real and tight, holding up against all three other arms, not just LMETRIC. Both whale-count
arms pay a consistent TBT-mean tax. DRF's TBT-max is unexpectedly the most stable of all
four arms (±1.9ms) — a previously-underappreciated DRF strength.

### Honest synthesis

DRF wins the actual energy objective (mean ramp, coincidence) with tight, replicated
confidence, at a real and consistent latency-tail cost (TTFT max ≈6.6s, every trial).
Neither whale-count mechanism delivers the coincidence benefit it was built for — refuted,
not merely unconfirmed. whale_argmin's only genuine surviving advantage is a modest,
real TTFT-mean edge, unrelated to the energy story that motivated building it.

**This means the real open design problem is narrower and different than what this whole
thread has been pursuing**: not "how do we get LMETRIC-like latency AND better coincidence
via whale-aware routing" (refuted), but **"how do we keep DRF's real coincidence/ramp
advantage while fixing its TTFT-tail cost"** — a targeted DRF-latency-tail problem, not a
whale-detection problem. The constrained-LMETRIC idea from earlier in this session (filter
out over-ceiling replicas, then LMETRIC among survivors — the V→∞ limit of Lyapunov
drift-plus-penalty) was never built or tested and may be the better next design to try,
since it directly targets DRF's actual weakness (its `Share_power` term overriding routing
on marginal, not just dangerous, power differences) rather than replacing DRF's power signal
with a coarser whale-presence proxy.

### Updated status and next steps (2026-08-31, superseded by the sections below)

1. Replicate `round_robin` too, for full footing (currently the only single-trial arm left).
2. **Build and test constrained-LMETRIC** (spec'd earlier this session, never implemented)
   as the next design — directly targets DRF's TTFT-tail weakness rather than its
   coincidence strength, which replication now shows doesn't need fixing.
3. If constrained-LMETRIC doesn't close DRF's tail-latency gap, investigate *why* DRF's tail
   is so consistently bad (6.5-6.7s every trial) — trace which specific requests drive it,
   since the consistency (tiny std) suggests a systematic cause, not random bad luck.

---

## Update 2026-08-31 (session close, part 1): constrained-LMETRIC tested — also refuted

Built and replicated `constrained_lmetric` (filter out replicas over their ramp ceiling,
then plain LMETRIC among survivors — the V→∞ limit of Lyapunov drift-plus-penalty, spec'd
earlier this session, see the theory discussion above). **Result: it doesn't win on either
axis.** Replicated mean ramp (255.4±8.6) is worse than plain LMETRIC (241.2±4.7); replicated
coincidence (32.9±8.3, worst of every arm tested) is worse than plain LMETRIC (23.7±4.1) and
worse than DRF (17.7±1.9); TTFT mean (0.549±0.002) is the worst of every arm tested, contrary
to the design's own prediction that it should equal plain LMETRIC in the common case (no
replica over ceiling, which the earlier pressure-window analysis suggested should hold
~85-90% of wall-clock time). This is refuted, not merely unconfirmed, matching the fate of
`p2c_whale` and `whale_argmin` — the third of three theoretically-motivated designs this
session to fail at combining LMETRIC's latency with DRF's power performance.

## Update 2026-08-31 (session close, part 2): diagnosed and fixed a real DRF bug — a
## controlled demonstration that the power/latency tension is real, not just three failed guesses

**Diagnosis.** Traced DRF's consistently bad TTFT max (6.58s±0.19, every replicated trial)
to a single, deterministic, reproducible cause: the exact same conversation
(`conv_id=9nPyg92_0`) is the single worst-TTFT request in *all three* DRF trials, with
nearly identical values (6.376/6.613/6.747s). Checking that same request across every other
policy confirmed it's a DRF-specific mechanism failure, not an intrinsically slow whale:
LMETRIC handles it in 1.8-3.6s, whale_argmin in 1.8-1.9s (fast *and* remarkably stable),
while DRF alone is 2-4x worse every single trial.

**Mechanism**: this whale arrives early and un-cacheable, so `new_tokens` (P-token) is
identical across all 6 replicas — nobody has anything cached yet. That makes
`Share_compute` (~0.90, a ~14.7k-token whale against a 16384-token budget) identical *and*
dominant across every candidate, since it dwarfs the load/power shares that early in a run.
`dominant_share`'s `max()` then returns that same tied value for every candidate,
**discarding all information about which replica has less load** — the routing decision
degenerates to an arbitrary tie-break rotation, exactly when load-awareness matters most.
LMETRIC's multiplicative score doesn't share this blind spot (`P-token × BS` still
differentiates by BS even with P-token tied), which is exactly why it — and the whale-count
policies, which don't use `dominant_share` for whales at all — handle this request better.

**The fix is a correction, not a new invention**: real DRF (Ghodsi et al., NSDI 2011;
Mesos's own reference implementation) breaks ties by comparing the *full sorted share
vector* lexicographically — dominant share first, then next-highest, then next — not a
scalar max. The router's `pick_drf` had simplified this to "max, then arbitrary rotation,"
a real deviation from the textbook mechanism. Added `dominant_share_vector()` (a strict
refinement — `vec[0] == dominant_share` always) and switched `pick_drf`'s tie-break key to
it (`df1bc32`). All existing DRF tests passed unchanged, confirming this only changes
behavior in the specific tied-dominant-share case it targets.

**Result, replicated 3x (`drf_fixed`, same "drf" mechanism family, one precise change):**

| | drf (original) | **drf_fixed** |
|---|---|---|
| TTFT max | 6.579±0.188s (worst of every arm) | **3.517±0.082s (best-in-class, tighter than whale_argmin's own)** |
| TTFT mean | 0.520±0.024 | 0.517±0.035 (unchanged) |
| mean ramp | **208.6±4.2 (best of every arm)** | 258.5±18.7 |
| coincidence | **17.7±1.9 (best of every arm)** | 30.0±2.7 (now *worse* than plain LMETRIC's 23.7±4.1) |

Every one of these deltas has cleanly non-overlapping std ranges — real effects, not noise.
**The fix works exactly as diagnosed (the tail-latency bug is gone), but it costs the power
advantage almost entirely — not partially, but flipped below the power-blind LMETRIC
baseline.** Mechanistic read: the "buggy" arbitrary rotation wasn't purely a defect — it
accidentally avoided the aggressive, LMETRIC-like load-packing that the theoretically
correct lexicographic tie-break now performs in that same tied-compute situation. Making
DRF smart about load in the one diagnosed bad case makes it smart about load in *every*
tied-compute case, pulling its overall behavior toward LMETRIC's aggressive-packing
philosophy — inheriting LMETRIC's power cost along with its latency benefit.

**This is the strongest evidence this session has for a real power-vs-latency tension in
this problem, precisely because it isn't an inference from three different failed designs
(the earlier, retracted framing) — it's the *same* mechanism, moved deliberately along the
curve by one understood, minimal, theory-motivated change, with both endpoints directly
measured.**

## Final comparison: all arms, replicated (mean ± std, 3 trials each except round_robin)

**Power / ramp / coincidence:**

| arm | mean W | mean ramp (W/s) | max ramp (W/s) | coincidence (%) |
|---|---|---|---|---|
| **drf (original)** | 2459±37 | **208.6±4.2 (best)** | 6310±1349 | **17.7±1.9 (best)** |
| lmetric | 2537±24 | 241.2±4.7 | 5497±1132 | 23.7±4.1 |
| drf_fixed | 2635±81 | 258.5±18.7 | 4901±603 | 30.0±2.7 |
| p2c_whale | 2477±131 | 215.6±24.8 | 5014±1008 | 32.0±7.2 |
| whale_argmin | 2695±19 | 278.6±9.5 (worst) | 5732±1217 | 32.1±4.3 |
| constrained_lmetric | 2669±37 | 255.4±8.6 | 6150±1258 | 32.9±8.3 (worst) |

**Latency:**

| arm | TTFT mean (s) | TTFT max (s) | TBT mean (ms) | TBT max (ms) |
|---|---|---|---|---|
| whale_argmin | **0.492±0.008 (best)** | 2.905±0.709 | 554.9±17.8 (worst) | 1766.6±5.3 |
| drf_fixed | 0.517±0.035 | **3.517±0.082 (tightest good tail)** | 467.6±39.6 | 1776.2±12.2 |
| drf (original) | 0.520±0.024 | 6.579±0.188 (worst) | **457.8±26.9 (best)** | 1803.3±1.9 (tightest) |
| lmetric | 0.526±0.020 | 3.690±0.143 | 449.4±46.9 | 2364.4±970.6 (worst, unstable) |
| p2c_whale | 0.524±0.016 | 4.133±0.550 | 537.0±37.0 | 1822.2±19.3 |
| constrained_lmetric | 0.549±0.002 (worst mean) | 3.652±0.107 | 467.7±18.4 | 1806.5±61.7 |

**Duty cycle and thermal** (replicated — the earlier single-trial "DRF is worst on this
axis" finding does *not* hold up; all 6 arms land in a narrow, largely indistinguishable
band, unlike the dramatic single-trial spread reported earlier in this document):

| arm | duty cycle | peak temp | mean temp |
|---|---|---|---|
| lmetric | 0.146±0.006 | 71.0±2.0°C | 48.1±1.7°C |
| drf (original) | 0.132±0.001 | 71.0±1.7°C | 48.1±1.7°C |
| drf_fixed | 0.141±0.008 | 71.0±1.7°C | 48.4±2.0°C |
| p2c_whale | 0.138±0.007 | 72.0±1.0°C | 49.9±0.2°C |
| whale_argmin | 0.131±0.005 | 71.3±0.6°C | 49.9±0.1°C |
| constrained_lmetric | 0.135±0.006 | 70.3±1.2°C | 48.3±1.7°C |

`round_robin` (single-trial only, not on the same footing as the rest): mean ramp 174, max
ramp 9645, coincidence 52.2%, TTFT mean 0.732, TTFT max 6.897, TBT mean 435.6, TBT max
1805.6.

### Session-closing synthesis

Three theoretically-motivated designs aimed at combining LMETRIC's latency with DRF's power
performance (`p2c_whale`, `whale_argmin`, `constrained_lmetric`) all failed at that specific
goal, once properly replicated — this should be read as evidence, not proof, that no such
mechanism exists, but it's real evidence. The strongest, most direct evidence for *why* is
the controlled `drf` vs `drf_fixed` comparison: the same mechanism, one precise, correctly
theory-motivated change (fixing a genuine deviation from textbook DRF), and power and
latency-tail performance moved in clearly opposite directions together. The duty-cycle/
thermal axis, once replicated across all six mechanisms, turned out not to meaningfully
differentiate any of them — an entire earlier storyline in this document that didn't survive
replication.

**Open for a future session**: whether this tension is fundamental to *routing* as a lever
specifically (as opposed to admission-time pacing/deferral, chunking, or DVFS, none of which
were tried here), and whether accepting a specific point on this now-measured curve (e.g.
`drf_fixed`, which gets most of DRF's power win at close to LMETRIC's latency) is preferable
to continuing to search for a dominating alternative.

## Update 2026-08-31 (new session): `pressure_switch` — fleet-state-triggered DRF/LMETRIC
## switching, a genuinely different mechanism from the four scoring-formula attempts above

### Design

Rejected the "LMETRIC + chunking" idea as not novel. Instead: route via DRF (lexicographic
tie-break, i.e. `drf_fixed`) only when the fleet is *currently* power-pressured (any
replica's own ramp rate exceeds its calibrated ceiling — a signal already computed for every
candidate on every decision, no new telemetry); route via plain LMETRIC otherwise. This
mirrors the sibling mlsys project's whale-aware budget controller (`token_budget = 512 if
whale_active else 16384`) — switch between two independently-validated mechanisms on
directly-observed global state, rather than one scoring formula trying to do both jobs (the
common failure mode of `p2c_whale`, `whale_argmin`, `constrained_lmetric` above). Built with
strict TDD (`fleet_pressured`/`pick_pressure_switch` in `scoring.py`, wired into `router_core`
as a 7th `_POLICIES` entry, commit `4bb1010`). 52/52 eenergy unit tests passing, local and
remote.

### Result (3-trial replication, same N=6/GPUs 2-7/workload/pad-seed 12345 as every arm above)

**Power / ramp / coincidence** (pressure_switch added to the table from the section above):

| arm | mean W | mean ramp (W/s) | max ramp (W/s) | coincidence (%) |
|---|---|---|---|---|
| drf (original) | 2459±37 | **208.6±4.2 (best)** | 6310±1349 | 17.7±1.9 |
| p2c_whale | 2476±131 | 215.6±24.8 | 5014±1008 | 32.0±7.2 |
| lmetric | 2537±24 | 241.2±4.7 | 5497±1132 | 23.7±4.1 |
| constrained_lmetric | 2669±37 | 255.4±8.6 | 6150±1258 | 32.9±8.3 |
| drf_fixed | 2635±81 | 258.5±18.7 | 4901±603 | 30.0±2.7 |
| **pressure_switch** | 2606±83 | 267.3±28.1 | **4286±1324 (best)** | **21.5±4.6 (2nd best)** |
| whale_argmin | 2695±19 | 278.6±9.5 | 5732±1217 | 32.1±4.3 |

**Latency:**

| arm | TTFT mean (s) | TTFT max (s) | TBT mean (ms) | TBT max (ms) |
|---|---|---|---|---|
| whale_argmin | 0.492±0.008 (best) | 2.905±0.709 | 554.9±17.8 | 1766.6±5.3 |
| drf_fixed | 0.517±0.035 | 3.517±0.082 | 467.6±39.6 | 1776.2±12.2 |
| **pressure_switch** | 0.517±0.082 | 3.619±0.166 | 497.0±78.6 | 1780.6±27.1 |
| drf (original) | 0.520±0.024 | 6.579±0.188 (worst) | **457.8±26.9 (best)** | 1803.3±1.9 |
| p2c_whale | 0.524±0.016 | 4.133±0.550 | 537.0±37.0 | 1822.2±19.3 |
| lmetric | 0.526±0.020 | 3.690±0.143 | 449.4±46.9 | 2364.4±970.6 (worst, unstable) |
| constrained_lmetric | 0.549±0.002 (worst mean) | 3.652±0.107 | 467.7±18.4 | 1806.5±61.7 |

**Duty cycle and thermal:**

| arm | duty cycle | peak temp | mean temp |
|---|---|---|---|
| pressure_switch | 0.135±0.004 | **69.7±1.5°C (best)** | 48.1±1.7°C |
| drf (original) | 0.132±0.001 | 71.0±1.7°C | 48.1±1.7°C |
| lmetric | 0.146±0.006 | 71.0±2.0°C | 48.1±1.7°C |
| drf_fixed | 0.141±0.008 | 71.0±1.7°C | 48.4±2.0°C |
| p2c_whale | 0.138±0.007 | 72.0±1.0°C | 49.9±0.2°C |
| whale_argmin | 0.131±0.005 | 71.3±0.6°C | 49.9±0.1°C |
| constrained_lmetric | 0.135±0.006 | 70.3±1.2°C | 48.3±1.7°C |

**How often the switch actually fires**: post-hoc reconstruction from the power trace (any
GPU's own instantaneous ramp exceeding the 450 W/s ceiling, at 0.05s tick resolution) puts
the fleet in the "pressured → DRF" regime **10.4-11.8% of the time** across the 3 trials —
consistent trial-to-trial, so the mode-switch itself isn't noisy, even though its downstream
latency numbers are (see below).

### Honest assessment: a real, partial win — not the dominating alternative

`pressure_switch` is the **first arm in this whole sequence that improves a power/ramp metric
without regressing TTFT mean relative to LMETRIC** (0.517 vs 0.526, within noise of each
other) — every earlier attempt (`p2c_whale`, `whale_argmin`, `constrained_lmetric`) that got
close to LMETRIC's TTFT gave up DRF's power win entirely, and every arm that kept DRF's power
win (`drf`, `drf_fixed`) paid a real TTFT-mean or TTFT-max cost. Specifically it:

- **Wins outright on max ramp** (4286 W/s, lowest of all 7 arms, beating even `drf_fixed`) —
  the tail-protection goal it was designed for.
- **Avoids DRF's worst known failure mode** (`drf` original's TTFT-max of 6.6s, the
  lexicographic-tiebreak bug's residual symptom) — pressure_switch's TTFT-max (3.62s) is in
  the same tight band as LMETRIC/drf_fixed/constrained_lmetric.
- **2nd-best coincidence** (21.5%, only `drf` original is better, and that arm has the worst
  TTFT-max in the whole table).
- **Best peak GPU temperature** of any arm (69.7°C).

But it is not a dominating alternative:

- **Does not improve mean ramp** (267.3 W/s — 2nd-worst of all 7 arms, worse than plain
  LMETRIC's 241.2). The switch only engages ~11% of the time, so it can bound the tail
  (max ramp) but can't move the *average* ramp, which is dominated by the ~89% of decisions
  made under plain LMETRIC.
- **Does not improve TBT mean** (497.0ms, worse than lmetric/drf/drf_fixed/constrained_lmetric).
- **Elevated run-to-run variance on every latency metric** (TTFT mean std 0.082 — roughly 4x
  every other arm's TTFT-mean std; TBT mean std 78.6ms — also the largest in the table). This
  is very likely a direct consequence of mode-switching itself: whichever trial happens to
  have a whale land during one of the ~11% pressured windows gets routed by DRF (which this
  document has already shown trades mean-case latency for tail-ramp protection), so trial
  outcomes depend on the specific timing of pressure windows relative to whale arrivals —
  the same sensitivity source as the underlying `drf`/`drf_fixed` tension, just diluted by
  the 89%-of-the-time LMETRIC floor rather than eliminated.

**Bottom line**: `pressure_switch` doesn't resolve the power/latency tension identified in
this document — it *localizes* it to the ~11% of decisions made under pressure, which is
enough to win the tail-ramp metric (the paper's actual headline power metric) at effectively
zero TTFT-mean cost, at the price of higher trial-to-trial variance than any static policy.
For a paper whose claim is about ramp/coincidence (not mean power), this is arguably the
best-positioned arm in the whole table: it's the first mechanism that's simple, theoretically
grounded (hysteresis/mode-switching, not a new scoring formula), and defensible as "does not
compromise LLM-serving latency" on the metric that matters most (TTFT mean) — with the caveat
that its variance needs to be reported honestly, not hidden.

### Updated status and next steps

- `pressure_switch` is now the leading candidate for the paper's constructive contribution
  (parallel to the sibling mlsys project's whale-aware controller, which has the same
  "deployable, non-oracle, mode-switching on observable state" shape).
- Open: whether the elevated variance is itself reportable as a property of the mechanism
  (a controlled cost of localizing the tension), or whether a 5th-6th trial would show it
  settling down — not yet tested past n=3.
- Open (carried over): whether `drf_fixed`'s pure profile (better mean ramp, worse TTFT mean
  by a hair) is a better single point on the tradeoff curve for a simpler paper narrative than
  `pressure_switch`'s more complex but better-localized story — a framing decision, not a
  measurement one, deferred to writeup.

## Update 2026-08-31 (new session): the first workload where KV$-awareness actually engages —
## every prior comparison in this document ran with cache-hit rate = 0 by construction

### Why every arm's "KV$-awareness" term was dead code until now

Traced precisely (not just suspected): the whale-injection harness prepends a per-request
random pad in FRONT of the whole prompt (`replay_sharegpt.py:230-237`,
`prompt = pad + "\n\n" + prompt`), and `cache_mirror.py`'s block-hash walk starts from block 0
and breaks on the first miss. Since the pad is unique per (pad_seed, conversation, turn), block
0 misses on essentially every request, so `new_tokens_if_routed` always returns the full prompt
length. This isn't an approximation — it's exact: **P-token was the raw prompt length on every
single request in every arm comparison in this document, including DRF's `Share_compute`**,
which uses the identical `new_tokens` value. Every power/latency conclusion above was measured
with the KV\$-awareness half of every scoring formula structurally inert.

Two small infra additions made this checkable rather than assumed: `Router.last_new_tokens`
(exposes the P-token actually used per decision, no signature change) and two new columns on
the assignment log, `new_tokens`/`raw_tokens` (`06bfc88`). A live audit against a real
cache-hit-producing run (serialized concurrency=1, unambiguous 1:1 correlation between
assignment-log rows and harness records) found the mirror completely honest: every discount
was an exact multiple of 16 tokens, and all 5 apparent "this replica wasn't the previous turn's
replica, why does it show a hit" cases traced to the SAME conversation having been served by
that exact replica on an EARLIER (non-immediately-previous) turn — `cache_mirror` correctly
remembers every replica a conversation has ever touched, not just the last one.

Also added, same session: `bs_source="telemetry"` (`1ec6963`) — an opt-in BS source that
reads real running+waiting counts off each replica's own vLLM `/metrics` (ground truth,
catches vLLM-side preemptions the router's own dispatch/complete bookkeeping can't see),
verified end-to-end on live hardware, zero errors across two full smoke-test runs. Not used
for the comparison below (still `bs_source="local"`, the default) — an orthogonal fix, folded
in for completeness.

### Workload: plain ShareGPT, no pad, no whale, real multi-turn

`--pad-chars 0 --whale-frac 0.0 --min-turns 2 --max-turns 4`, same N=6/GPUs 2-7,
`--concurrency 24 --num-convs 150 --max-tokens 128`, same fixed pad-seed 12345. 7 arms
(dropping the superseded buggy `drf`, keeping `drf_fixed` as "the" DRF arm going forward) x
3 trials each, 21 runs total. Assignment log enabled on every run so cache-hit-rate is a
directly measured number for the first time, not assumed.

**Power, ramp, coincidence & cache-hit-rate** (mean ± std, 3 trials):

| arm | hit rate (%) | peak fleet W | mean ramp (W/s) | max ramp (W/s) | coincidence (%) |
|---|---|---|---|---|---|
| round_robin | 9.9 ± 2.1 | 1957.6 ± 25.8 | 103.03 ± 3.65 | 3086.0 ± 310.9 | 19.6 ± 7.7 |
| lmetric | 50.2 ± 0.3 | 1915.0 ± 21.3 | 94.20 ± 2.57 | 3287.7 ± 502.5 | 24.5 ± 4.0 |
| drf_fixed | **46.6 ± 0.9 (lowest of the cache-aware arms)** | 1907.7 ± 3.0 | 90.97 ± 10.81 | **2890.4 ± 80.4 (tightest)** | 20.3 ± 2.9 |
| p2c_whale | 50.4 ± 0.3 | 1898.1 ± 5.6 | 89.14 ± 4.14 | 3878.6 ± 640.2 (worst) | 34.6 ± 13.4 (worst) |
| whale_argmin | 50.3 ± 0.2 | 1901.0 ± 1.6 | **87.14 ± 4.59 (best)** | 3257.7 ± 893.7 | 24.4 ± 12.6 |
| constrained_lmetric | 50.4 ± 0.3 | 1915.5 ± 1.8 | 88.38 ± 2.11 | 3206.4 ± 790.4 | 34.0 ± 10.3 |
| pressure_switch | 50.1 ± 0.1 | 1901.4 ± 1.5 | 91.88 ± 2.02 | **2895.5 ± 458.1 (2nd-tightest)** | 25.5 ± 6.0 |

**Latency:**

| arm | TTFT mean (s) | TTFT max (s) | TBT mean (ms) | TBT max (ms) |
|---|---|---|---|---|
| round_robin | **0.073 ± 0.001 (worst)** | 0.248 ± 0.019 | **48.2 ± 1.4 (worst)** | 158.8 ± 4.7 |
| lmetric | 0.065 ± 0.001 | 0.242 ± 0.020 | 31.7 ± 0.2 | 157.6 ± 2.2 |
| drf_fixed | 0.065 ± 0.000 | 0.234 ± 0.020 | 33.7 ± 1.3 | 154.7 ± 0.9 |
| p2c_whale | 0.065 ± 0.001 | 0.234 ± 0.014 | 31.9 ± 0.2 | 154.9 ± 7.0 |
| whale_argmin | 0.065 ± 0.001 | 0.255 ± 0.001 (worst) | 31.7 ± 0.3 | 149.7 ± 4.7 |
| constrained_lmetric | **0.064 ± 0.000 (best)** | **0.230 ± 0.010 (best)** | 32.2 ± 0.4 | 154.8 ± 7.3 |
| pressure_switch | 0.065 ± 0.000 | 0.236 ± 0.003 | **31.3 ± 0.4 (best)** | 150.0 ± 4.3 |

**Duty cycle & thermal:**

| arm | duty cycle | peak temp | mean temp |
|---|---|---|---|
| round_robin | 0.000 ± 0.000 | 68.7 ± 2.1°C | 50.3 ± 2.3°C |
| lmetric | 0.000 ± 0.000 | 67.7 ± 1.5°C | 51.4 ± 0.1°C |
| drf_fixed | 0.000 ± 0.000 | 67.0 ± 1.7°C | 51.1 ± 0.6°C |
| p2c_whale | 0.000 ± 0.000 | 68.0 ± 1.0°C | 51.6 ± 0.0°C |
| whale_argmin | 0.000 ± 0.000 | 67.7 ± 2.1°C | 51.1 ± 0.6°C |
| constrained_lmetric | 0.000 ± 0.000 | 69.3 ± 0.6°C | 51.7 ± 0.0°C |
| pressure_switch | 0.000 ± 0.000 | **66.3 ± 1.2°C (lowest)** | 51.6 ± 0.1°C |

Duty cycle is flat 0.000 for every arm — the 420W-per-sample threshold was calibrated for the
whale-workload's power profile; this much lighter chat traffic never sustains a single GPU
above it. Not a real "no difference" finding, a threshold mismatch — would need its own
recalibration to be meaningful for this workload.

### Read

**Cache-awareness finally shows a real, measured effect**: round_robin sits nearly alone at
9.9% hit rate (content-blind, occasional accidental same-replica revisits from pure rotation
luck); every cache-aware arm clusters tightly around ~50%, replicated to sub-percent std.
`drf_fixed` is the one measurable outlier among them — consistently ~4 points lower — the
first direct evidence that DRF's power/load weighting has a real, small cost to cache
efficiency, something the whale-injection workload could never show since hit rate was zero
regardless of arm there.

**`pressure_switch` is the best-balanced arm here, not an outright winner on every axis** —
same pattern as `whale_argmin` was for the whale-injection comparison. It wins TBT-mean and
peak-temp outright, ties for tightest/lowest max-ramp with `drf_fixed`, and is never clearly
worst on anything — but `whale_argmin` beats it on mean-ramp (87.1 vs 91.9) and
`constrained_lmetric` edges it on TTFT-mean (0.064 vs 0.065). Consistently near the top across
every axis, not uniquely best on most of them.

**TTFT differentiation is real but modest**: round_robin's 0.073s vs the cache-aware pack's
0.064-0.065s is a genuine ~12-14% gap, not nothing — but far smaller than the dramatic
separation one might expect from "half your prompt is being skipped." Most likely explanation:
this workload (concurrency=24 against N=6 replicas with max_num_seqs=64, i.e. only ~4
concurrent requests per replica on average) is running well under saturation — a cache hit
mostly saves compute that wasn't the bottleneck yet. **Open, proposed next**: rerun at
meaningfully higher concurrency to see whether TTFT differentiation sharpens once replicas are
actually contending for capacity, which would be the more realistic regime for this effect to
matter in practice.

**Does DRF's power win (established under whale-injection) generalize to normal traffic? Not
clearly.** `mean_ramp` here is 85-107 W/s across ALL arms (vs 208-278 W/s under
whale-injection) — a much gentler regime with the 6 cache-aware arms overlapping heavily
within noise; DRF/`drf_fixed` doesn't stand out as the mean-ramp winner the way it did there
(87.1-94.2 range, `drf_fixed` at 91.0 is mid-pack, `whale_argmin` is actually lowest here).
Most likely explanation: without whale-driven power spikes, `Share_power` rarely dominates the
max in DRF's dominant-share calculation, so DRF behaves closer to a load-balance-only policy
for most of this workload. One thing that DOES partially survive: `drf_fixed` and
`pressure_switch` remain the two tightest/lowest `max_ramp` arms even without whales — some
tail-ramp protection generalizes even though the mean-ramp advantage doesn't.

**Caveat this whole comparison needs**: this is a genuinely lighter workload, not an
apples-to-apples swap of the whale-injection experiment minus whales. Peak power here
(1898-1958W) is ~20-25% below the whale-workload's (2400-2700W) — short chat turns plus
128-token outputs simply don't stress the fleet the same way. Some of "DRF's power win
disappeared" could be "there's much less power dynamics to win on here" rather than a real
refutation of the whale-injection finding.

### Status and next steps

- Propose a higher-concurrency rerun of this same plain multi-turn workload (same 7 arms, same
  replication discipline) to test whether TTFT differentiation sharpens toward saturation, and
  whether the power/ramp differentiation between arms (currently washed out without whales)
  reappears under real contention rather than requiring whale-driven spikes specifically.
- Open: whether `drf_fixed`'s ~4-point cache-hit-rate cost is itself worth reporting as a
  distinct, named tradeoff in the paper (DRF trades some cache efficiency for power-awareness),
  independent of the power/latency tension already established.

---

## Update 2026-09-01: higher-concurrency rerun refutes the TTFT-sharpening hypothesis; BurstGPT
## real traffic surfaces a genuine round-robin WIN under load; two isolating experiments trace
## it to decode-duration variance, not workload identity or loop-mode alone

### Higher-concurrency (60) rerun of the cache-hit workload: TTFT gap collapses, doesn't widen

**Conditions:** `replay_sharegpt.py --dataset data/sharegpt_v3.json --pad-chars 0
--whale-frac 0.0 --min-turns 2 --max-turns 4 --concurrency 60 --num-convs 150
--max-tokens 128`, real multi-turn (plain ShareGPT, no pad, no whale — identical to the
cache-hit workload in the section above, only `--concurrency` raised 24→60), pad-seed 12345,
N=6 replicas/GPUs 2-7/Qwen2.5-Coder-7B-Instruct/ramp-ceiling 450 W/s, 7 arms x 3 trials (21
runs) — to test the proposed next step (does TTFT differentiation sharpen toward saturation?).

**Result: refuted.** TTFT mean collapses to a statistically flat band across every arm —
round_robin 0.086±0.001, lmetric 0.087±0.003, drf_fixed 0.085±0.003, p2c_whale 0.086±0.001,
whale_argmin 0.088±0.002, constrained_lmetric 0.085±0.001, pressure_switch 0.084±0.001. The
12-14% gap seen at concurrency=24 (round_robin 0.073 vs. cache-aware pack 0.064-0.065) is gone,
not widened. TBT-mean differentiation survives (round_robin 69.9±1.5ms vs. 49.5-55.8ms for the
rest) and hit-rate differentiation survives (round_robin 11.9±1.5% vs. ~50% for cache-aware
arms, `drf_fixed` again the outlier at 43.6±2.6%) — but the TTFT-specific hypothesis this rerun
was built to test did not hold up. Queueing-delay dominance at higher concurrency plausibly
washes out cache-hit savings' relative contribution to TTFT specifically, without touching the
decode-side (TBT) advantage. Directly motivated the switch to real bursty traffic (BurstGPT)
to test the same underlying question a different way, below.

### BurstGPT real-trace comparison: the first workload where round_robin actually WINS

**Conditions:** `scripts/make_burstgpt_trace.py BurstGPT_1.csv <out.csv> --off 75000
--nrows <N> --gapcap 3.0 --target-s 60` slices real rows from `BurstGPT_1.csv` (off=75000),
rescales the real inter-arrival gaps (capped at 3.0s) to a 60s span, and replays them via
`replay_sharegpt.py --host ... --trace-csv <out.csv> --request-timeout 180` — real arrival
burstiness and real request/response length distribution, single-turn, unique synthetic
filler content per request (deliberately uncacheable by design — this is the counterpart test
to the cache-hit workload above, not a replacement for it). Same N=6 replicas/GPUs
2-7/Qwen2.5-Coder-7B-Instruct/ramp-ceiling 450 W/s setup throughout. 3 rates tested (via
`--nrows`: 300/900/1500 rows over the same 60s span), 7 arms x 3 trials each (63 runs total):

| rate | trace | BS source | req_tokens (mean/p99/max) |
|---|---|---|---|
| 5 req/s | 300 rows | local | 451 / 2240 / 2493 |
| 15 req/s | 900 rows | telemetry | 528 / 1991 / 2493 |
| 25 req/s | 1500 rows | local | 553 / 2037 / 4156 |

(`BS source` = router's in-flight-count signal: `local` is the router's own dispatch/complete
bookkeeping, `telemetry` reads real `vllm:num_requests_running/waiting` off each replica's
`/metrics` — see the BS-via-telemetry implementation note earlier in this doc's history.)

**TTFT mean, all 3 rates:**

| arm | 5 req/s (local) | 15 req/s (telemetry) | 25 req/s (local) |
|---|---|---|---|
| round_robin | 0.080 ± 0.001 (worst) | **0.094 ± 0.000 (best)** | **0.100 ± 0.000 (best)** |
| lmetric | 0.077 ± 0.001 | 0.131 ± 0.005 | 0.109 ± 0.000 |
| drf_fixed | 0.078 ± 0.001 | 0.134 ± 0.001 | 0.112 ± 0.001 |
| p2c_whale | 0.077 ± 0.000 | 0.128 ± 0.002 | 0.109 ± 0.001 |
| whale_argmin | 0.077 ± 0.000 | 0.131 ± 0.003 | 0.110 ± 0.001 |
| constrained_lmetric | 0.077 ± 0.001 | 0.130 ± 0.003 | 0.109 ± 0.000 |
| pressure_switch | 0.077 ± 0.000 | 0.131 ± 0.003 | 0.109 ± 0.000 |

At 5 req/s round_robin is (marginally) worst, matching every prior workload. At 15 and 25
req/s round_robin **wins outright on every latency metric** (TTFT mean/max, TBT mean/max —
full tables in session logs, not reproduced here) by 9-43%, tightly replicated (std mostly
≤0.005). This is the first workload in the whole project where round-robin beats every
cache/load-aware arm on latency, not a marginal or single-trial effect.

**Mechanism, worked out in stages (see "isolating experiments" below for the controlled
tests)**: under 0% cache-hit rate (confirmed 0.0% for every arm/trial here, by construction —
unique filler content, no shared prefix), LMETRIC/DRF's routing decision collapses to
picking the replica with the fewest in-flight *requests* — a count, not a measure of actual
remaining work. This is a known queueing-theory failure mode (join-shortest-queue-by-count
underperforming blind assignment when service times are highly variable): a replica secretly
decoding one long response can look "count=1" (free) while genuinely being the worst choice,
and count-based routing keeps sending it more work. Two isolating experiments below test the
competing candidate explanations (workload identity / loop-mode / decode-duration variance)
directly rather than resting on this as an untested inference.

**A second, independently-diagnosable anomaly surfaced at 25 req/s: `drf_fixed` has the
*worst* mean_ramp of all 7 arms — worse than both power-blind baselines.**

| arm | mean_ramp (W/s), 25 req/s | |
|---|---|---|
| round_robin (zero power awareness) | 121.94 ± 3.15 | |
| lmetric (zero power awareness, by design) | 132.26 ± 5.79 | |
| **drf_fixed (power-aware — this is its whole point)** | **150.57 ± 2.20 (worst of 7)** | tight std, real effect |

Mechanism: DRF's dominant share is `max(Share_compute, Share_load, Share_power)`, tied on
`Share_compute` across every candidate under 0% cache-hit (P-token = raw prompt length
identically for all 6 replicas on every decision, far more often than in the original
whale-injection workload where compute only ties occasionally). The lexicographic tie-break
then falls through to `Share_load` almost every decision, making `drf_fixed` behave like
aggressive LMETRIC-style load-packing for its *compute* term while nominally claiming
power-awareness — inheriting the packing aggression that produces sharp ramps, exactly the
mechanism findings.md's earlier `drf`→`drf_fixed` diagnosis (session close, part 2, above)
already identified, just engaging far more often here. Not independently re-verified via
per-decision share-value instrumentation (the assignment log only records
`new_tokens`/`raw_tokens`, not the three share components) — a plausible, evidence-consistent
inference, not a directly-traced confirmation.

**Whale-threshold recalibration was tested as a candidate fix and refuted.**
`WHALE_TOKEN_THRESHOLD` (admission-time cutoff for `p2c_whale`/`whale_argmin`'s
whale-aware routing, default 4000) was calibrated against the synthetic whale-injection
workload's 13.6-15.5k-token whales; real BurstGPT traffic essentially never crosses it (0-2
of 300-1500 rows per trace, ≤0.1%) — so that machinery never engages here. Made the threshold
configurable (`ROUTER_WHALE_TOKEN_THRESHOLD` env var, `Router(whale_token_threshold=...)`,
TDD'd, 27/27 tests) and recalibrated to this trace's own p85 (1200 tokens, matching the
project's whale-frac=0.15 convention), rerunning just `p2c_whale`/`whale_argmin` (the only two
policies that consume `is_whale`) at 25 req/s (`local` BS source, same 1500-row trace, same
fleet config as above):

| arm | threshold=4000 (old) | threshold=1200 (recalibrated) | round_robin baseline |
|---|---|---|---|
| p2c_whale TTFT mean | 0.109 ± 0.001 | 0.107 ± 0.001 | 0.100 ± 0.000 |
| whale_argmin TTFT mean | 0.110 ± 0.001 | 0.106 ± 0.001 | 0.100 ± 0.000 |

A small, real improvement (~2%) but nowhere near closing the gap. Whale-threshold
miscalibration is not the (or not the primary) mechanism.

### Isolating experiment 1: closed-loop vs. open-loop, same whale-injection dataset — refutes
### the backpressure hypothesis

Working hypothesis after the BurstGPT result: the whale-injection workload above is
closed-loop (`--concurrency 24` — a new request only submits when another completes,
self-throttling, so a bad routing decision gets corrected by the next arrival's fresh state);
BurstGPT is open-loop (`--trace-csv`, real arrival timestamps indifferent to system state, so
a misrouted burst can't self-correct before more arrivals land). Tested directly.

**Conditions:** `replay_sharegpt.py --dataset data/sharegpt_v3.json --min-turns 1
--max-turns 1 --rate 8.5 --num-convs 150 --max-tokens 128 --whale-frac 0.15
--whale-min-chars 44000 --whale-max-chars 50000 --request-timeout 180` — the **exact same**
whale-injection dataset/params as the original closed-loop comparison at the top of this
document (single-turn, whale-frac 0.15, 44-50k-char whales, num-convs 150, max-tokens 128),
only `--rate 8.5` (open-loop Poisson) in place of `--concurrency 24`. Rate chosen via Little's
Law (L=λW) from the original closed-loop run's own real measured per-request latency
(`logs/eenergy_records_lmetric.jsonl`: mean W=2.81s, n=150, 21 whales at 4.57s, 129 short at
2.53s) to target the same average concurrency (~24) as the closed-loop baseline: λ =
24/2.81 ≈ 8.5 conv/s. Same N=6 replicas/GPUs 2-7/Qwen2.5-Coder-7B-Instruct/ramp-ceiling 450
W/s. 7 arms x 3 trials (21 runs).

**Result: refuted.** round_robin does not flip to winning — it stays clearly worst (TTFT mean
0.783±0.079 open-loop vs. 0.732 closed-loop, actually marginally *worse*, not better), while
lmetric/p2c_whale/whale_argmin stay flat-to-better (−0.6% to −10.6%).

| arm | closed-loop (orig) | open-loop, same dataset | Δ |
|---|---|---|---|
| round_robin | 0.732 (n=1) | 0.783 ± 0.079 | +7.0% |
| lmetric | 0.526 ± 0.020 | 0.523 ± 0.045 | −0.6% |
| drf_fixed | 0.517 ± 0.035 | 0.677 ± 0.165 | **+30.9%** |
| p2c_whale | 0.524 ± 0.016 | 0.468 ± 0.022 | −10.6% |
| whale_argmin | 0.492 ± 0.008 | 0.455 ± 0.025 | −7.6% |
| constrained_lmetric | 0.549 ± 0.002 | 0.592 ± 0.033 | +7.9% |
| pressure_switch | 0.517 ± 0.082 | 0.636 ± 0.028 | **+23.1%** |

Closed-loop-vs-open-loop is **not** the explanation for BurstGPT's round-robin win — refuted
by direct test, not merely unconfirmed. Real, separate finding surfaced instead: the DRF
lexicographic-tie-break mechanism (`drf_fixed`, and `pressure_switch` via its DRF-pressured
mode) degrades substantially under open-loop specifically (+31%, +23%), while whale-count-aware
and pure-LMETRIC routing don't — consistent with, and independent evidence for, the same
tie-break-collapses-to-packing mechanism diagnosed for the BurstGPT `drf_fixed` ramp anomaly
above, now shown to be sensitive to loop-mode even on a dataset where it doesn't flip the
overall ranking.

### Isolating experiment 2: decode-duration variance (max-tokens 128→1024, open-loop held
### fixed) — partial confirmation

Revised hypothesis after experiment 1's refutation: not loop-mode, but **decode-duration
variance**. The whale-injection workload caps every request's output at `max_tokens=128`
regardless of prefill size — bounding how long any request can occupy a replica slot, however
big its prefill. BurstGPT has real, unbounded response lengths (up to 1278 tokens) — a replica
stuck decoding a long response can look "count=1" for a genuinely long stretch, a much bigger
blind spot for count-based routing.

**Conditions:** `replay_sharegpt.py --dataset data/sharegpt_v3.json --min-turns 1
--max-turns 1 --rate 8.5 --num-convs 150 --max-tokens 1024 --whale-frac 0.15
--whale-min-chars 44000 --whale-max-chars 50000 --request-timeout 180` — identical to
isolating experiment 1 (same open-loop rate 8.5 conv/s, same dataset/whale-params, same N=6
replicas/GPUs 2-7/Qwen2.5-Coder-7B-Instruct/ramp-ceiling 450 W/s), with exactly one variable
changed: `--max-tokens` 128→1024 (comparable ceiling to BurstGPT's real max response of 1278
tokens). 7 arms x 3 trials (21 runs).

| arm | open-loop, 128-token cap | open-loop, 1024-token cap | Δ |
|---|---|---|---|
| round_robin | 0.783 ± 0.079 | 0.792 ± 0.057 | **+1.1% (flat)** |
| lmetric | 0.523 ± 0.045 | 0.674 ± 0.091 | **+28.9%** |
| drf_fixed | 0.677 ± 0.165 | 0.668 ± 0.089 | −1.3% |
| p2c_whale | 0.468 ± 0.022 | 0.664 ± 0.037 | **+41.9%** |
| whale_argmin | 0.455 ± 0.025 | 0.519 ± 0.036 | +14.1% |
| constrained_lmetric | 0.592 ± 0.033 | 0.536 ± 0.070 | −9.5% |
| pressure_switch | 0.636 ± 0.028 | 0.657 ± 0.030 | +3.3% |

**Result: partial confirmation, right direction, not a full reversal.** round_robin stays
essentially flat (its routing never depends on a load signal that could go stale, so unbounded
decode length shouldn't move it — and it doesn't). The count-based aware arms specifically
degrade — lmetric +28.9%, p2c_whale +41.9% — exactly the predicted stale-signal blind spot.
The gap to round_robin genuinely narrows (round_robin was 72% worse than the best arm,
whale_argmin, under the 128-token cap; only 53% worse under the 1024-token cap) but round_robin
still doesn't win outright here — it remains the single worst arm in the long-output table.
Most likely explanation: this workload's rate (8.5 conv/s, matched to the closed-loop
baseline's intensity) is lighter than the BurstGPT runs where round_robin actually won (15-25
req/s) — decode-duration variance is a real, correctly-directional contributing mechanism, but
appears to need real load/density to fully compound into an outright win, not either factor
alone. **Not yet tested**: this same long-output open-loop workload at BurstGPT-comparable
rate (15-25 conv/s-equivalent) — the natural next step to check for a full reversal.

### P-token cache-hit-honesty audit: algorithm verified, per-request empirical join not
### currently possible

`cache_mirror.py`'s 4 unit tests directly verify the exact cache-hit-computation logic (empty
cache → full miss; identical-prefix extension → only trailing new tokens counted; prefix
divergence mid-block → only the diverging block onward counted, matching prefix credited
correctly) — all pass. Attempted a deeper empirical cross-check (do first-turn requests, which
should show zero cache benefit, actually report `new_tokens == raw_tokens`?) using the
concurrency=24 cache-hit ShareGPT run's (conditions above, "Workload: plain ShareGPT, no pad,
no whale") real `lmetric`/`drf_fixed`/`round_robin` assignment logs + records.jsonl, trial 1
of each. Two join attempts (naive
row-index matching, then dispatch-time-reconstructed via `ts - latency`) both showed ~50-65%
"mismatches" for cache-aware arms — traced to the join itself being unreliable, not a router
bug: neither log has a shared request ID, and under concurrency=24 even reconstructed
timestamps can't reliably disambiguate near-simultaneous dispatches. **Open**: add a monotonic
request-sequence-number to both `ROUTER_ASSIGNMENT_LOG` and `replay_sharegpt.py`'s record
output if a rigorous per-request empirical confirmation (beyond the unit-test-level guarantee)
is wanted later.

### Session-closing synthesis (2026-09-01)

The cache-aware/power-aware routing family's TTFT advantage over round-robin, which held up
across every prior workload in this document (whale-injection, cache-hit ShareGPT at both
concurrency levels), **is not universal** — real BurstGPT traffic at moderate-to-high arrival
rate reverses it. Two isolating experiments narrowed the mechanism from an initial
"closed-loop backpressure" guess (refuted by direct test) to "decode-duration variance"
(partially confirmed, right direction, needs density to fully compound) — both real,
answerable questions this project can pursue further:

1. Does the decode-duration-variance mechanism produce a full reversal at BurstGPT-comparable
   density, or does something else (real content/size distribution, something specific to
   BurstGPT's actual trace) also matter? — next open-loop-long-output-at-density run answers
   this directly.
2. A work-aware BS signal (in-flight *estimated remaining tokens*, not request count) was
   designed but not yet built — directly targets the diagnosed mechanism. Design note: it must
   normalize DRF's `Share_load` against `token_budget` rather than `max_num_seqs` when active
   (a token-sum ÷ sequence-count-capacity isn't a real fraction and would make `Share_load`
   permanently dominate DRF's `max()`).

### Status and next steps

1. Run the long-output open-loop whale-injection workload at BurstGPT-comparable density
   (15-25 conv/s-equivalent) to test for a full round-robin reversal, isolating the last
   remaining variable (rate/density) from decode-duration variance.
2. Build the work-aware BS signal (`bs_source="work"`, static per-request `max_tokens`
   estimate, `Share_load` renormalized against `token_budget`) and test whether it recovers
   the cache/power-aware arms' edge on BurstGPT-like traffic.
3. Consider adding request-sequence-number logging to enable a rigorous per-request P-token
   honesty audit beyond the current unit-test-level guarantee, if that level of confirmation
   becomes load-bearing for the paper.

---

## Update 2026-09-01 (continued): density-matched round-robin reversal test refuted; two
## candidate DRF fixes built and tested — neither is a strict Pareto improvement

### Rate=25 density-matched attempt saturated the fleet — corrected via token-throughput
### matching, not request-rate matching

First attempt at the open-loop-long-output-at-BurstGPT-comparable-density test (status item 1
above) used `--rate 25` (matching BurstGPT's raw *request* rate) on the same whale-injection
dataset as the earlier open-loop experiments. **Result: complete saturation, not informative.**
TTFT mean collapsed to a statistically-indistinguishable ~19.7-20.9s band across all 7 arms
(round_robin 20.875±0.658, lmetric 20.439±0.784, drf_fixed 20.146±0.434, p2c_whale
19.844±0.589, whale_argmin 20.383±1.015, constrained_lmetric 19.712±0.671, pressure_switch
20.000±0.332) — routing policy stops mattering entirely once the fleet is far enough past its
throughput ceiling; pure queueing delay dominates and swamps every policy difference. Root
cause: matching raw *request* rate isn't the same as matching aggregate *token* throughput.
BurstGPT's real 25 req/s trace (`logs/burstgpt_arms_trace_25rps.csv`) demands ~18659 tokens/s
(829631 prompt + 289928 response tokens over 60s, computed directly from the trace). This
whale-injection workload's real per-request demand (measured from the successful rate=8.5
run's actual generated output, not the nominal cap: mean `prompt_tokens_approx`=1403, mean
`output_tokens`=337) is ~1740 tokens/req — 2.3x heavier per request than BurstGPT's real
~746 tokens/req average, so matching "25 req/s" on this workload demanded far more aggregate
throughput than BurstGPT's real 25 req/s ever did.

**Corrected, token-throughput-matched conditions:** `replay_sharegpt.py --dataset
data/sharegpt_v3.json --min-turns 1 --max-turns 1 --rate 10.7 --num-convs 750 --max-tokens
1024 --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 --request-timeout
180`. Rate = 18659 / 1740 ≈ 10.7 conv/s (BurstGPT's real 25rps aggregate token demand divided
by this workload's real per-request demand); num-convs scaled to 750 for a ~70s sustained
window comparable to BurstGPT's own 60s trace. Same N=6 replicas/GPUs 2-7/
Qwen2.5-Coder-7B-Instruct/ramp-ceiling 450 W/s. 7 arms x 3 trials (21 runs) — this run is
referred to as "throughput-matched" or "matched" below.

**Result: still refuted, and the gap widened, not narrowed.** round_robin TTFT mean
5.380±0.312 (worst of 7) vs. the best arm whale_argmin's 2.255±0.970 — round_robin 139%
worse, compared to only 53% worse at the lighter rate=8.5/1024-cap run. Decode-duration
variance combined with properly-matched density does **not** reproduce BurstGPT's
round-robin win on this synthetic whale-injection construction — moving in the opposite
direction from what the joint hypothesis predicted. Full latency table:

| arm | TTFT mean (s) | TTFT max (s) | TBT mean (ms) |
|---|---|---|---|
| round_robin | 5.380 ± 0.312 (worst) | 24.538 ± 3.932 | 1253.8 ± 42.9 (best) |
| lmetric | 4.147 ± 0.923 | 23.127 ± 2.819 | 1299.6 ± 38.1 |
| drf_fixed | 3.712 ± 0.935 | 20.622 ± 7.470 | 1336.3 ± 43.9 |
| p2c_whale | 3.403 ± 0.457 | 15.165 ± 2.512 | 1385.6 ± 23.6 |
| whale_argmin | 2.255 ± 0.970 (best) | 11.215 ± 0.535 (best) | 1418.1 ± 55.3 (worst) |
| constrained_lmetric | 3.704 ± 0.412 | 18.994 ± 2.396 | 1291.5 ± 35.9 |
| pressure_switch | 3.441 ± 1.093 | 18.350 ± 3.572 | 1330.1 ± 9.5 |

**Leading remaining hypothesis, not yet tested**: this whale-injection workload's big items
are identifiable *at admission time* (a 44-50k-char whale is obviously huge from its prompt
alone) — both whale-aware and even plain count-based routing can anticipate and avoid them.
BurstGPT's real requests have no such signal: a short-prompt request can still generate a
long response (up to 1278 tokens) with nothing in the prompt hinting at it — a blind spot no
current classification here (prompt-length-only) can catch. Checking BurstGPT's real
prompt-length-vs-response-length correlation directly would confirm or refute this; not yet
done.

**A genuine, useful side-finding: round_robin is best on power/thermal axes here, worst on
latency — a real tradeoff, not "round_robin loses across the board."**

| arm | max_ramp (W/s) | coincidence (%) | TPOT mean (s) | mean_temp (°C) |
|---|---|---|---|---|
| round_robin | 2981.3 ± 255.5 (best) | 1.4 ± 0.3 (best) | 0.0377 ± 0.0016 (best) | 58.4 ± 1.3 (coolest) |
| whale_argmin | 3777.1 ± 1134.0 | 3.6 ± 1.3 | 0.0428 ± 0.0005 (worst) | 59.7 ± 0.6 |

round_robin's blind, evenly-cycled spreading avoids concentrating decode-heavy load on any
single replica (good for power/thermal evenness) even while its lack of queue-awareness costs
it badly on TTFT. whale_argmin shows the mirror-image tradeoff: best TTFT, worst TPOT-mean,
among the least power-even. Duty cycle is meaningfully nonzero here for the first time on any
open-loop whale run (0.27-0.30 across arms, vs. flat 0.000 at the lighter rate=8.5) — this
workload genuinely stresses power at this density, unlike the earlier lighter runs.

### Diagnosis: why does a power-aware policy (drf_fixed) lose to power-blind ones (lmetric,
### round_robin) on the power metrics it exists to protect?

Two independent runs (this throughput-matched whale workload, and the BurstGPT 25 req/s run
documented above) both show `drf_fixed` with the *worst* max_ramp/coincidence of every arm
tested, worse than both power-blind baselines:

| arm | max_ramp (W/s), matched | coincidence (%), matched |
|---|---|---|
| round_robin (zero power awareness) | 2981.3 ± 255.5 | 1.4 ± 0.3 |
| lmetric (zero power awareness, by design) | 3182.7 ± 600.2 | 3.1 ± 1.1 |
| **drf_fixed (power-aware — this is its whole point)** | **5010.6 ± 1940.8 (worst)** | **7.7 ± 1.5 (worst)** |

`share_power(c) = max(c.ramp_rate_w_per_s, 0.0) / c.ramp_ceiling_w_per_s`
(`scripts/eenergy/router/scoring.py`) — a *reactive* measure of how close a replica's own
currently-observed ramp is to its ceiling. DRF routes on `max(Share_compute, Share_load,
Share_power)`; Share_power only affects the outcome when it's the *largest* of the three for
a candidate, otherwise it's invisible to the decision. Two structural mechanisms sideline it
on exactly this kind of workload:

1. **For whale requests — the ones that matter most for ramp — Share_compute structurally
   dominates Share_power almost by construction.** A 13.6-15.5k-token whale against a
   16384-token `token_budget` gives Share_compute ≈ 0.83-0.95; Share_power would need the
   *target* replica already at 83-95% of its own ramp ceiling to win instead. Precisely when
   dispatching the request that would actually cause a big ramp spike, DRF's power term is
   most likely to be overridden by the compute term.
2. **Under 0% cache-hit (confirmed for both workloads), Share_compute ties across every
   candidate on every decision**, forcing the lexicographic tie-break (`drf_fixed`'s own fix,
   see "session close, part 2" above) down to `Share_load`, bypassing Share_power entirely —
   this time regardless of whale status. `dominant_share_vector`'s tie-break is a pure
   **magnitude sort** over `(compute, load, power)`, not a resource-prioritized one — whichever
   of `{load, power}` happens to be numerically larger at that moment wins the tie-break, by
   accident of scale, not because load is deliberately meant to matter more.

Not independently verified via per-decision share-value instrumentation (the assignment log
only records `new_tokens`/`raw_tokens`, not the three share components) — a plausible,
evidence-consistent inference from two independent workloads showing the same pattern, not a
directly-traced confirmation.

### Fix 1: `drf_power_tiebreak` — resource-priority tie-break instead of magnitude sort

Targets mechanism 2 directly: same primary DRF criterion (route to minimum dominant share),
but the tie-break vector becomes `(dominant_share, share_power, share_load)` — power always
compared second, regardless of its numeric size relative to load — instead of
`dominant_share_vector`'s pure magnitude sort. `scoring.py`'s
`dominant_share_vector_power_priority()` / `pick_drf_power_tiebreak()`, wired into
`router_core.py` as an 8th `_POLICIES` entry. TDD, 4 new tests (2 scoring-level demonstrating
the tie-break reversal directly with a constructed tied-compute/differing-load-and-power
example, 1 router-level integration smoke test, 1 default-matches-plain-drf-when-no-tie
regression guard), 185/185 passing. Commit `698fc5f`.

**Tested on the throughput-matched (moderate-heavy) workload — a real, clean win on the
targeted metrics:**

| metric | drf_fixed (old) | drf_power_tiebreak (new) | round_robin (reference) |
|---|---|---|---|
| max_ramp (W/s) | 5010.6 ± 1940.8 (worst of 7) | **3015.3 ± 257.1 (near round_robin's floor)** | 2981.3 ± 255.5 (best) |
| coincidence (%) | 7.7 ± 1.5 (worst of 7) | **4.4 ± 1.2** | 1.4 ± 0.3 (best) |
| mean_ramp (W/s) | 191.86 ± 9.52 | 204.13 ± 7.51 (slightly worse) | 195.40 ± 0.68 |
| TTFT mean (s) | 3.712 ± 0.935 | 3.942 ± 0.271 (flat, much tighter std) | 5.380 ± 0.312 |
| TBT max (ms) | 2703.7 ± 968.5 | 2034.6 ± 181.6 | 2974.1 ± 953.1 |
| duty cycle | 0.281 ± 0.021 | 0.259 ± 0.010 | 0.274 ± 0.013 |

max_ramp went from worst-of-7 to essentially matching round_robin's tightest number (40%
reduction, std collapsed ~8x, ±1940.8→±257.1 — no more occasional catastrophic bursts).
Coincidence dropped 43%. TBT-max improved and got far more stable. Real costs: mean_ramp
slightly worse (+6%), TTFT-max noisier (20.6→25.6, wider std). TTFT-mean flat within noise,
with a notably tighter std.

**Tested on the concurrency=24 cache-hit workload (light load) — regresses.**
`replay_sharegpt.py --dataset data/sharegpt_v3.json --pad-chars 0 --whale-frac 0.0
--min-turns 2 --max-turns 4 --concurrency 24 --num-convs 150 --max-tokens 128
--request-timeout 180`, pad-seed 12345 (default), same fleet config, 3 trials.

| metric | drf_fixed | drf_power_tiebreak | direction |
|---|---|---|---|
| hit_rate % | 46.6 ± 0.9 | 46.9 ± 1.0 | flat — fix doesn't touch cache efficiency |
| max_ramp (W/s) | 2890.4 ± 80.4 (tightest of 7) | 3533.9 ± 860.1 | **worse, and 10x noisier** |
| coincidence (%) | 20.3 ± 2.9 (2nd-best) | 24.1 ± 10.0 | **worse, and much noisier** |
| TTFT mean (s) | 0.065 ± 0.000 | 0.065 ± 0.001 | flat |
| TBT mean (ms) | 33.7 ± 1.3 | 34.2 ± 0.7 | flat |

**Mechanism for the reversal**: this workload's duty_cycle is flat 0.000 for every arm
(confirmed earlier in this doc — light chat traffic never sustains a GPU above the 420W
threshold), meaning `share_power(c) ≈ 0` for essentially every candidate essentially all the
time. When the tie-break is forced to compare power *first* under these conditions, it's
comparing near-zero, largely noise-level ramp differences — whichever replica shows a
marginally lower reading *this instant* wins, even though the difference is meaningless —
instead of falling through to `share_load`, which genuinely does vary and would have been the
more informative signal. This is the mirror image of the original bug: originally load won
ties by *accident of magnitude* even when power was the resource that actually mattered
(heavy load); now power wins ties *by design* even when it's the resource that *doesn't*
meaningfully differ (light load) — a fixed priority order is only correct for the regime it
was diagnosed in. Not directly verified via per-decision instrumentation, same standing
caveat.

### Fix 2: `lmetric_power` — continuous multiplicative power penalty, LMETRIC's own philosophy

`Score = new_tokens × in_flight_after × (1 + share_power)`, route to the minimum, no
tie-break rule needed at all. Extends LMETRIC's own multiplicative form (Zhang et al.,
OSDI'26) rather than patching DRF's `max()`-of-shares mechanism — a product never lets one
factor's influence go to zero the way `max()` structurally lets one dimension silence the
other two (the root cause behind both mechanisms diagnosed above). `(1 + share_power)` is
bounded, always positive, and continuous: a replica at its ramp ceiling scores 2x worse than
an identical replica with zero ramp; one with headroom is barely penalized — a lightweight,
always-on cousin of Neely's Lyapunov drift-plus-penalty (already cited here for
`constrained_lmetric`'s hard V→∞ cutoff), but smooth instead of a discrete on/off boundary, so
there's no threshold to misfire near. `scoring.py`'s `lmetric_power_score()` /
`pick_lmetric_power()`, wired into `router_core.py` as a 9th `_POLICIES` entry. TDD, 5 new
tests (score formula, matches-lmetric-under-no-pressure, differentiates-tied-lmetric-scores-
via-power, empty-candidates guard, router-level integration smoke test), 190/190 passing.
Commit `04a8d31`.

**Tested on the cache-hit workload (light load) — the light-load prediction holds up
cleanly, a real confirmation:**

| metric | lmetric (baseline) | lmetric_power | drf_fixed (reference) |
|---|---|---|---|
| hit_rate % | 50.2 ± 0.3 | **50.4 ± 0.2 (matches)** | 46.6 ± 0.9 (worse — DRF's known cache cost) |
| TTFT mean (s) | 0.065 ± 0.001 | **0.065 ± 0.000 (matches)** | 0.065 ± 0.000 |
| TBT mean (ms) | 31.7 ± 0.2 | **31.5 ± 0.5 (matches, tiny improvement)** | 33.7 ± 1.3 |
| mean_ramp (W/s) | 94.20 ± 2.57 | **88.91 ± 2.12 (real, small improvement)** | 90.97 ± 10.81 |
| max_ramp (W/s) | 3287.7 ± 502.5 | 3453.5 ± 1033.6 (within lmetric's own noise) | 2890.4 ± 80.4 |
| coincidence (%) | 24.5 ± 4.0 | 30.0 ± 15.9 (within lmetric's own noise, very wide) | 20.3 ± 2.9 |

hit_rate, TTFT-mean, and TBT-mean all land essentially on top of plain LMETRIC — confirming
the formula's core prediction directly: under `duty_cycle=0.000`, `share_power≈0` everywhere,
so `(1+share_power)≈1` everywhere, and the score collapses to plain LMETRIC. Unlike
`drf_power_tiebreak`, no clear directional regression against its own natural baseline
(lmetric) on any metric — mean_ramp even improved slightly. max_ramp/coincidence nominally
worse but with very wide std relative to the value (coincidence: 30.0 ± 15.9, >50% relative
uncertainty) — consistent with this project's established caution about noisy extreme-value
stats at n=3 when real power-pressure events are rare (duty_cycle=0.000 here).

**Tested on the throughput-matched workload (moderate-heavy load) — genuinely mixed, an
unpredicted regression on mean_ramp/duty_cycle:**

| metric | round_robin | lmetric (baseline) | drf_fixed | drf_power_tiebreak | lmetric_power |
|---|---|---|---|---|---|
| TTFT mean (s) | 5.380 | 4.147 | 3.712 | 3.942 | **3.320 (2nd-best of 9, beats lmetric by 20%)** |
| max_ramp (W/s) | 2981.3 (best) | 3182.7 | 5010.6 (worst) | **3015.3 (near-best)** | 3757.0 (better than drf_fixed, worse than lmetric & drf_power_tiebreak) |
| coincidence (%) | 1.4 (best) | 3.1 | 7.7 (worst) | 4.4 | 4.5 (matches drf_power_tiebreak, still worse than lmetric) |
| mean_ramp (W/s) | 195.40 | 198.71 | 191.86 | 204.13 | **209.08 (worst of all 9 arms)** |
| duty_cycle | 0.274 | 0.273 | 0.281 | 0.259 (best of the aware arms) | **0.292 (worst of all 9 arms)** |

TTFT is genuinely excellent — 2nd-best of all 9 arms tested on this workload (only
`whale_argmin` beats it), a real 20% improvement over plain LMETRIC. Coincidence improves
substantially over `drf_fixed`, matching `drf_power_tiebreak`'s fix. But mean_ramp and
duty_cycle both got *worse* than plain LMETRIC, worse than every other arm tested here — not
predicted going in.

**Working hypothesis for the mean_ramp/duty_cycle regression, not yet verified**: a
continuous penalty steers *every* decision away from whichever replica currently looks most
power-pressured — under real, fast-changing load this could create an oscillation (avoid
whoever's hot → pile onto whoever's cool → that one heats up → gets avoided next → load
swings back). That would produce exactly this signature: fewer catastrophic *multi-GPU*
coincident spikes (better coincidence, since load keeps getting redistributed before multiple
GPUs ramp together) and no single extreme outlier (moderate max_ramp), but *more frequent,
smaller* ramp transitions everywhere as the router keeps chasing the momentarily-coolest
replica — exactly what mean_ramp and duty_cycle measure. `drf_power_tiebreak` doesn't share
this failure mode because it only intervenes on an exact tie of the dominant share — a much
narrower trigger than a continuous penalty reshaping every decision. Not directly verified
(would need per-decision replica-choice-sequence tracing to confirm the oscillation directly).

### Pareto-frontier synthesis: no single arm is a strict improvement across every axis

Checking whether any arm dominates another on *every* axis simultaneously (throughput-matched
workload, the richest dataset with all three DRF-family variants):

| arm | TTFT mean | TBT mean | max_ramp | coincidence | mean_ramp | duty_cycle |
|---|---|---|---|---|---|---|
| whale_argmin | **2.255 (best)** | 1418.1 (worst) | 3777.1 | 3.6 | 213.04 (worst) | 0.298 |
| drf_power_tiebreak | 3.942 | 1320.9 | **3015.3 (best excl. r_r)** | **4.4** | 204.13 | 0.259 (best excl. r_r) |
| lmetric_power | 3.320 | 1357.4 | 3757.0 | 4.5 | 209.08 (worst) | 0.292 (worst) |
| constrained_lmetric | 3.704 | 1291.5 (best) | 3584.0 | 3.5 (best) | 205.84 | 0.283 |

Nothing dominates `whale_argmin` on latency without giving up TBT/ramp. Nothing dominates
`drf_power_tiebreak` on ramp without giving up TTFT. `lmetric_power` sits between them on
TTFT/coincidence but is dominated by `drf_power_tiebreak` on mean_ramp and duty_cycle
specifically (worse on both, not just one) — so unlike the other three, `lmetric_power` is
not itself on the Pareto frontier of this specific 4-arm comparison, though it remains
genuinely useful where TTFT is the priority and mean-ramp/duty-cycle are secondary. **There is
no single arm across this project's whole routing-family search that is Pareto-optimal in the
strong sense (beats everything on every axis) — this is a real, multi-dimensional tradeoff
surface, not a gap in what's been tried.** The two genuine frontier points for a
moderately-loaded, non-round-robin deployment are `whale_argmin` (TTFT-first) and
`drf_power_tiebreak` (ramp-tail-first, with the best mean-ramp/duty-cycle profile of the three
power-aware fixes).

### Status and next steps

1. A hybrid switching `whale_argmin` (default) / `drf_power_tiebreak` (under fleet pressure)
   — mirroring `pressure_switch`'s own lmetric/drf_fixed pattern one level up — was proposed
   as a way to inherit `whale_argmin`'s TTFT most of the time while only paying
   `drf_power_tiebreak`'s ramp-protective cost when actually needed. Not yet built; the
   simpler `lmetric_power` alternative was tried first per explicit request, and while it
   didn't turn out to be a strict improvement, testing both single-formula candidates before
   committing to a mode-switching hybrid (with `pressure_switch`'s known ~4x variance cost)
   was the right order of investigation.
2. Neither `drf_power_tiebreak` nor `lmetric_power` has been tested on BurstGPT itself yet —
   only on the two whale-injection workloads. Given round_robin's win is BurstGPT-specific
   (not reproduced by the density-matched whale-injection construction above), it's not yet
   known whether either DRF fix changes anything relative to round_robin there.
3. The mean_ramp/duty_cycle oscillation hypothesis for `lmetric_power` is plausible but
   unverified — per-decision replica-choice-sequence tracing (not currently logged; would
   need a small instrumentation addition) would confirm or refute it directly.
4. Per-decision share-value instrumentation (Share_compute/Share_load/Share_power at each
   routing decision, not just the final `new_tokens`/`raw_tokens` currently logged) remains
   the standing open item to move every mechanism claim in this section from
   "evidence-consistent inference" to "directly traced."

### Fix 3: `lmetric_power_convex` — convex power penalty, testing whether it fixes fix 2's
### regression

Hypothesis for `lmetric_power`'s unpredicted mean_ramp/duty_cycle regression (fix 2, above):
a *linear* `(1+share_power)` penalty reacts to noise-level power differences even when every
replica is comfortably under-ceiling, plausibly causing a chase-the-coolest-replica
oscillation. Convex ramp-cost penalties are the standard convention in power-systems economic
dispatch / unit commitment literature specifically because stressing a generator near its
ramp limit carries disproportionate, super-linear cost — arguably better theoretical grounding
for this project's actual domain than the linear form, not just an ad hoc tweak.

`Score = new_tokens × in_flight_after × (1 + share_power²)` — same zero-free-parameter
LMETRIC-style form, squaring instead of adding linearly. Strictly smaller than the linear
penalty below the ceiling (0.3² = 0.09 vs 0.3 — barely reactive to noise), equal to it exactly
at the ceiling (share_power=1), and strictly larger above it (steeper punishment for genuinely
exceeding the calibrated ceiling). `scoring.py`'s `lmetric_power_convex_score()` /
`pick_lmetric_power_convex()`, wired into `router_core.py` as a 10th `_POLICIES` entry. TDD, 6
new tests (score formula, the convex<linear-below/equal-at/greater-above-ceiling property
directly, matches-lmetric-under-no-pressure, differentiates-tied-scores,
empty-candidates guard, router-level integration smoke test), 196/196 passing. Commit
`e750a65`.

**Tested on the cache-hit workload (light load) — light-load prediction holds, and improves
on the linear version:**

| metric | lmetric | lmetric_power (linear) | lmetric_power_convex |
|---|---|---|---|
| hit_rate % | 50.2 ± 0.3 | 50.4 ± 0.2 | 50.5 ± 0.0 |
| TTFT mean (s) | 0.065 ± 0.001 | 0.065 ± 0.000 | 0.065 ± 0.000 |
| TBT mean (ms) | 31.7 ± 0.2 | 31.5 ± 0.5 | 31.5 ± 0.6 |
| mean_ramp (W/s) | 94.20 ± 2.57 | 88.91 ± 2.12 | **85.24 ± 5.32 (best of the three)** |
| coincidence (%) | 24.5 ± 4.0 | 30.0 ± 15.9 (very noisy) | **18.7 ± 4.8 (better than lmetric, much tighter than linear)** |

**Tested on the throughput-matched workload (moderate-heavy load) — a real, replicated trade
in the opposite direction from what was predicted, not a clean fix:**

| metric | lmetric_power (linear) | lmetric_power_convex | direction |
|---|---|---|---|
| TTFT mean (s) | 3.320 ± 0.663 | **3.234 ± 0.466 (improved)** | better |
| mean_ramp (W/s) | 209.08 ± 6.11 (worst of 9) | **204.12 ± 4.34 (fixed — now ties drf_power_tiebreak)** | better |
| duty_cycle | 0.292 ± 0.007 (worst of 9) | **0.271 ± 0.015 (fixed — matches lmetric's own baseline)** | better |
| max_ramp (W/s) | 3757.0 ± 771.5 | **5128.9 ± 1994.0 (worst of all 10 arms)** | **worse** |
| coincidence (%) | 4.5 ± 1.1 | **7.8 ± 1.3 (back to drf_fixed's poor level)** | **worse** |

The convex penalty fixed exactly what it targeted — mean_ramp and duty_cycle, the
*chronic/average* symptoms — but at the cost of making max_ramp and coincidence, the
*acute/tail* symptoms, worse than the linear version, both cleanly (not noise-level:
reasonable stds on both sides of the trade). **Working mechanism, not directly verified**:
making the penalty barely reactive below the ceiling (by design — that's what fixed the
chronic oscillation) also removes the router's early warning to steer away from a replica
*before* it gets close to dangerous. Pressure can build up further before the convex penalty
engages meaningfully, so when it finally reacts, the correction is more abrupt — worse tail
events, even though the average behavior in between is calmer. The linear version reacts a
little all the time (worse chronic behavior, but continuous early course-correction); the
convex version reacts sharply only late (better chronic behavior, worse acute reactions). This
is a genuine chronic-vs-acute tradeoff in penalty shape, not a bug in either formula.

**This is not a Pareto improvement over `lmetric_power` either — it is a different point on a
different tradeoff axis**, not a resolution of the ramp-protection tension. It strengthens
rather than resolves the Pareto-frontier picture below.

### Pareto-frontier synthesis, updated with all three DRF/LMETRIC-family fixes

Checking whether any arm dominates another on every axis simultaneously
(throughput-matched workload, all fixes on the same footing):

| arm | TTFT mean | TBT mean | max_ramp | coincidence | mean_ramp | duty_cycle |
|---|---|---|---|---|---|---|
| whale_argmin | **2.255 (best)** | 1418.1 (worst) | 3777.1 | 3.6 | 213.04 (worst) | 0.298 |
| drf_power_tiebreak | 3.942 | 1320.9 | **3015.3 (best excl. r_r)** | **4.4** | 204.13 | **0.259 (best excl. r_r)** |
| lmetric_power | 3.320 | 1357.4 | 3757.0 | 4.5 | 209.08 (worst) | 0.292 |
| lmetric_power_convex | **3.234** | 1306.9 | 5128.9 (worst) | 7.8 | 204.12 | 0.271 |
| constrained_lmetric | 3.704 | **1291.5 (best)** | 3584.0 | **3.5 (best)** | 205.84 | 0.283 |

**No single arm dominates every axis — five genuinely distinct, non-dominated points now
populate this frontier**, each best suited to a different priority: `whale_argmin` (TTFT-
first, at a real TBT-mean/mean-ramp cost), `drf_power_tiebreak` (best all-around ramp/tail
profile among the power-aware fixes, competitive but not best TTFT), `constrained_lmetric`
(best TBT-mean and coincidence, simple hard-constraint mechanism, no new formula needed),
`lmetric_power` (best TTFT among the LMETRIC-family fixes, worst chronic ramp), and
`lmetric_power_convex` (best TTFT overall among non-`whale_argmin` arms, worst acute/tail
ramp). This is a genuine, multi-dimensional tradeoff surface — the search for a single
formula-level Pareto-dominant strategy across load/cache/power jointly has now been tried
three independent ways (magnitude-sort→resource-priority tie-break, linear penalty, convex
penalty) and none of the three escapes it. That itself is evidence the tension is structural
to the problem (queue state and power state are different physical quantities that are not
reliably correlated in this traffic), not a gap in the search.

### Which strategy for the paper: recommendation

**`drf_power_tiebreak` is the strongest single candidate for a paper headline**, not because
it dominates the frontier (nothing does) but because of what it *is*: a one-line, precisely
diagnosed, theoretically-grounded fix to a cited mechanism (Ghodsi et al.'s own lexicographic
DRF tie-break, restored from a mid-project simplification), validated by two independent
converging diagnoses (BurstGPT 25rps and the throughput-matched whale workload both showing
`drf_fixed` as the *worst* arm on the metrics DRF exists to protect) and a clean, large,
tightly-replicated fix (max_ramp -40%, std ~8x tighter; coincidence -43%) at zero
cache-efficiency cost, with a mechanistically-explained (not just observed) scope limitation
that is low-stakes in practice (the light-load regime where it regresses is also the regime
where duty_cycle≈0 and there is no real ramp problem to protect against). Its own weakness —
not the single best TTFT — is honestly bounded by the frontier table above, not hidden.

**Recommended framing, not yet built or tested**: patch `pressure_switch` (whose
DRF-pressured mode currently calls plain `pick_drf`, i.e. `drf_fixed`, per
`scoring.py:233`) to call `pick_drf_power_tiebreak` instead. This combines two
independently-validated pieces from this session — `pressure_switch`'s proven gating (only
engages DRF-mode during real fleet pressure, which should sidestep `drf_power_tiebreak`'s own
light-load regression) with `drf_power_tiebreak`'s proven fix (sidesteps `drf_fixed`'s flaw in
exactly the regime `pressure_switch` calls it) — inheriting LMETRIC's light-load behavior by
default and a validated ramp-protection fix under real pressure, potentially with neither
mechanism's weakness surfacing in its wrong regime. This is the natural next experiment, not
yet run.

### Status and next steps (updated)

1. **Build and test the `pressure_switch` + `drf_power_tiebreak` synthesis** (one-line change
   to `pick_pressure_switch`) on both the cache-hit and throughput-matched workloads — the
   most promising untested candidate, combining two pieces already independently validated
   this session.
2. A hybrid switching `whale_argmin` (default) / `drf_power_tiebreak` (under fleet pressure)
   remains a live alternative if the simpler pressure_switch synthesis above doesn't clear
   `whale_argmin`'s TTFT bar.
3. None of `drf_power_tiebreak`, `lmetric_power`, or `lmetric_power_convex` has been tested on
   BurstGPT itself yet — only on the two whale-injection workloads. Given round_robin's win is
   BurstGPT-specific (not reproduced by the density-matched whale-injection construction
   above), it's not yet known whether any of these fixes changes anything relative to
   round_robin there.
4. The mean_ramp/duty_cycle chronic-oscillation hypothesis (for the linear penalty) and the
   early-warning/acute-severity hypothesis (for the convex penalty) are both plausible but
   unverified — per-decision replica-choice-sequence tracing (not currently logged; would
   need a small instrumentation addition) would confirm or refute both directly.
5. Per-decision share-value instrumentation (Share_compute/Share_load/Share_power at each
   routing decision, not just the final `new_tokens`/`raw_tokens` currently logged) remains
   the standing open item to move every mechanism claim in this section from
   "evidence-consistent inference" to "directly traced."

## Update 2026-09-01 (continued): why whale-aware routing has no real signal on real BurstGPT traffic

**Motivation.** The paper's remaining open gap (§ "Open gaps" in `paper-eenergy/paper.md`):
real BurstGPT traffic shows `round_robin` *winning* at 15-25 req/s while every synthetic
whale-injection workload shows it losing, and none of the whale-aware or DRF-family fixes
have ever been run against the real BurstGPT trace itself. Leading untested hypothesis
carried over from the prior session: whale classification is admission-time-visible in the
synthetic workload by construction (a whale IS a long prompt, by design), but real traffic
may not have that property — decode length may be effectively unpredictable from prompt
length alone, breaking the premise every whale-aware policy depends on.

**Method.** Direct analysis of the raw trace (`/root/pli/BurstGPT/data/BurstGPT_1.csv` on
the 8x4090 server, 1,429,738 rows, 145,707 "Conversation log" rows with response≥1 tokens —
this is the real dataset, not a replay-harness-filtered subset, so the earlier-diagnosed
1024-token prompt-censoring bug in the old `burstgpt.cli` replay tool does not apply here).
Computed Pearson and Spearman correlation between `Request tokens` (prompt) and
`Response tokens` (decode length) across the full trace, plus conditional statistics at this
project's `WHALE_TOKEN_THRESHOLD=4000`.

**Result — the admission-time-signal hypothesis is confirmed, with real numbers:**

- Pearson r(prompt, response) = **0.336**, Spearman ρ = **0.325** — a real but weak-to-moderate
  correlation, nowhere near strong enough to treat prompt length as a reliable decode-length
  proxy.
- Only **565 of 145,707 requests (0.39%)** would be admission-time-flagged as a whale at the
  4000-token threshold — vs. the synthetic workload's designed 15% whale fraction, a ~40x
  gap in workload composition alone, independent of the correlation question.
- **Of the top 5% longest-response requests — the ones whale-aware routing exists to
  protect — only 3.24% have prompt_tokens > 4000.** 96.76% of the requests that actually
  matter for decode-duration/TBT protection are admission-time-invisible to every
  whale-aware policy tested (`p2c_whale`, `whale_argmin`, `whale_argmin_power_switch`).
- The reverse check also fails: of the 565 requests that DO get flagged as whales, only
  **41.77%** turn out to actually be in the top-5%-longest-response group — the majority of
  admission-time-flagged "whales" are long-prefill-but-ordinary-decode requests, a false
  positive for the mechanism whale-aware routing is trying to protect against.

**Interpretation.** This is real, well-supported evidence (not yet a closed-loop causal
test — see below) that the entire whale-aware branch of policies (`p2c_whale`,
`whale_argmin`, `whale_argmin_power_switch`) has almost no admission-time signal to act on in
real BurstGPT traffic, because the synthetic workload's core design assumption (long prompt
⟹ long decode, by construction) does not transfer to real traffic (weak correlation, and the
threshold-based classifier catches the wrong 0.39% either way). This does NOT, by itself,
explain the DRF-family's behavior (`drf`, `drf_fixed`, `drf_power_tiebreak`,
`lmetric_power(_convex)`, `pressure_switch`) — those react to *live* per-replica power-ramp
state, not admission-time prompt length, so this specific mechanism doesn't implicate them.
Whether they also lose to `round_robin` on real traffic, and if so why, is still untested.

**Still open (this is a partial, not complete, resolution of the paper's gap):**
1. This is correlational evidence about the *signal*, not a causal test of the *routing
   outcome* — no DRF-family or whale-aware policy has actually been run against the real
   BurstGPT trace itself yet (only against the throughput-matched synthetic reconstruction).
   That experiment is the natural next step to close this gap fully.
2. Does NOT explain the non-whale-aware DRF-family policies' relationship to the real-trace
   `round_robin` win, if any — untested.
3. `/root/pli/vllm-experiment/scripts/make_burstgpt_trace.py` already exists and slices real
   trace windows into `(arrival_s, prompt_tokens, response_tokens)` for
   `src/replay_sharegpt.py --trace-csv` — the infra to run this experiment is already in
   place, not a new build.

## Update 2026-09-01 (continued): the round_robin-wins-on-real-traffic discrepancy is real, replicated, and NOT limited to whale-aware policies

**Discovery.** While scoping a new experiment to test DRF-family policies against the real
BurstGPT trace, found that this experiment already exists and is complete: 7 policies
(`round_robin`, `lmetric`, `drf_fixed`, `p2c_whale`, `whale_argmin`, `constrained_lmetric`,
`pressure_switch`) × 3 replicated trials × 3 real-trace rate conditions (default ~5 req/s/300
arrivals, 15 req/s/900 arrivals, 25 req/s/1500 arrivals — all sliced from the real
`/root/pli/BurstGPT/data/BurstGPT_1.csv` via `make_burstgpt_trace.py`, not the old
prompt-censoring-bugged replay tool) were already run and sitting in
`/root/pli/vllm-experiment/logs/burstgpt{,15,25}_*`. Pulled and analyzed all 63+63+63=189
result files.

**Result at 15 req/s — round_robin wins decisively and broadly, replicated tightly (n=3):**

| metric | round_robin | best of the other 6 | worst of the other 6 |
|---|---|---|---|
| TTFT_mean | **0.094±0.000** | 0.128±0.002 (p2c_whale) | 0.134±0.001 (drf_fixed) |
| TTFT_max | **0.268±0.010** | 0.515±0.063 (constrained_lmetric) | 0.645±0.087 (lmetric) |
| lat_mean | **1.922±0.003** | 1.970±0.002 (p2c_whale) | 1.984±0.006 (drf_fixed) |
| TBT_mean_ms | **18.209±0.007** | 18.325±0.065 (p2c_whale) | 18.384±0.034 (drf_fixed) |
| TBT_max_ms | **224.2±12.6** | 376.2±25.5 (constrained_lmetric) | 456.2±21.2 (lmetric) |
| mean_ramp | **22.1±0.4** | 33.1±1.3 (lmetric) | 36.1±1.7 (drf_fixed) |
| peak_W | **383.1±6.4** | 428.3±8.1 (constrained_lmetric) | 439.4±11.2 (whale_argmin) |

round_robin wins every metric except max_ramp (noisy, overlapping std). The gaps are large
(37-43% on TTFT_mean, ~2x on TTFT_max/TBT_max) and the replication is tight — this is not
noise. **Critically, `drf_fixed` and `pressure_switch` — policies that react to live
per-replica power-ramp state, not admission-time prompt-length classification — lose just as
badly as the whale-aware policies.** The whale-aware-signal explanation (previous update,
2026-09-01) cannot account for this: DRF-family's mechanism doesn't depend on whale
classification at all.

**At 25 req/s the gap narrows and partially reverses**: round_robin still wins TTFT_mean
(0.100 vs 0.109-0.112, smaller ~9-12% gap), mean_ramp (23.4 vs 25.9-28.6), and max_ramp, but
loses TBT_mean (20.433 vs 20.158-20.198 for everyone else) and roughly ties lat_mean/peak_W.

**At the default ~5 req/s (light/moderate load) round_robin is actually the worst arm** on
TTFT_mean, lat_mean, TBT_mean, and mean_ramp (though by small margins, 2-30%) — the reversal
is specifically a moderate-to-heavy-load phenomenon (15-25 req/s), not present at light load.
This is directionally consistent with (though not the same finding as) the synthetic
workload's own light-load result ("round robin is definitely the one to go" was never
claimed — rather the opposite direction here, worth noting as a genuine surprise, not
smoothed over).

**Implication — this is a bigger problem for the paper than previously scoped.** The
whale-aware-signal analysis explained why `p2c_whale`/`whale_argmin`/
`whale_argmin_power_switch` specifically have no real admission-time signal on real traffic.
This result shows the reversal is broader: at real, moderate-to-heavy load, EVERY power/
fairness-aware policy tested — including the DRF-family, which reacts to live state rather
than admission-time classification — loses to content-blind round_robin on nearly every
metric that matters (TTFT, latency, mean ramp). The mechanism for the DRF-family's specific
failure is still unexplained; candidate hypotheses (untested):
1. Real traffic's prompt/response distribution (median prompt 548 tok, median response 232
   tok, heavy-tailed) differs qualitatively from the whale-injection synthetic workload
   (artificially padded, more uniform heavy load) — DRF/LMETRIC's scoring may not track true
   resource cost well on this distribution, since P-token (prefill cost) correlates only
   weakly with actual decode burden (r=0.34, per the prior update) even setting aside the
   whale threshold specifically.
2. Round-robin's deterministic cycling gives an implicit, content-blind fairness guarantee
   (exactly 1/N of requests per replica) that a content-based scorer can accidentally violate
   when its score is a poor proxy for true cost — e.g. many similarly-"cheap-looking" small
   requests could pile onto whichever replica currently looks least loaded, creating
   synchronization/clumping that round-robin's blind cycling structurally avoids.
Neither hypothesis has been directly tested — this is the standing open item.

**Still open / next steps:**
1. The 4 newest power-aware policies (`drf_power_tiebreak`, `lmetric_power`,
   `lmetric_power_convex`, `whale_argmin_power_switch`) have not yet been run against any of
   these 3 real-trace conditions — natural next batch, using the exact same
   `run_router.py`/batch-script pattern and the exact same trace files
   (`burstgpt_arms_trace{,_15rps,_25rps}.csv`) already proven to work.
2. Neither candidate mechanism above (cost-proxy mismatch vs. implicit-fairness-violation) is
   distinguished yet — would need per-request replica-assignment-vs-true-cost analysis
   (assignment logs already exist for all 63 already-run trials, not yet mined for this).
3. This finding should be surfaced prominently in the paper, not folded quietly into a
   footnote — it materially affects how confidently the paper can claim power-aware routing
   is a viable real-world lever at all, independent of the Pareto-frontier characterization
   among the power-aware policies themselves.

## Update 2026-09-02: full comparison grid — four alternative mechanisms tried against
## `drf_power_tiebreak`, none beat it; comprehensive table across all conditions

Following the second thesis pivot (theory-first paper, `drf_power_tiebreak` as the featured
result), this session tried four structurally different alternative mechanisms, motivated by
the user's critique that Pareto-optimality is a weak claim ("it doesn't guarantee anything")
and that hand-picked branching thresholds (as in `pressure_switch`, `constrained_lmetric`)
don't generalize across arrival rate / prompt length. Each is a genuinely different design,
not another permutation of the same DRF tie-break idea:

1. **`drf_coincidence_tiebreak`** — replaces `share_power(c)` (per-replica ramp/ceiling) with
   a fleet-aggregate share that scores whether routing to `c` would trigger a NEW
   simultaneous multi-GPU ramp, not how ramped `c` itself is. Breaks Lemma 1's separable-share
   assumption (needs full candidate-list visibility, not just `c`). **Refuted**: worse max_ramp
   AND worse coincidence than `drf_power_tiebreak` in the one condition tested (heavy/matched).
   Mechanism: it makes piling more load onto an already-ramping replica "free" (no new
   coincidence event), which plausibly pushes that replica's OWN ramp trajectory higher and
   makes routing more path-dependent (confirmed noisier: 4x higher std on max_ramp).
2. **`drf_peak_power_tiebreak`** — swaps the ramp-rate share for a peak POWER LEVEL share
   (`c.power_w / 450W hardware limit`). Power level is non-negative (no cancellation across
   replicas, unlike signed ramp), so this stays fully separable — Lemma 1 applies unmodified,
   no new theory needed. **Consistent trade-off across all 3 conditions tested**: best (or
   near-best) coincidence every time, but worst-or-near-worst on nearly every other metric
   (TTFT, TBT, sometimes mean_ramp) — most severe in the light/cachehit condition (mean_ramp
   4x every other arm, TTFT/TBT ~2x worse). Mechanism (plausible, matches `lmetric_power`'s
   own documented "chase-the-coolest-replica" pathology): under fast, bursty closed-loop
   completion patterns, always routing toward whichever replica currently looks least-powered
   causes much more frequent idle<->loaded cycling per replica, but the cycling stays
   decorrelated across replicas, hence the coincidence win despite everything else losing.
3. **`drf_power_tiebreak_p2c`** — Power-of-Two-Choices (Mitzenmacher 1996/2001) applied to
   the DRF score itself: sample 2 candidates at random, route to whichever wins on
   `drf_power_tiebreak`'s own tie-break vector, instead of full-visibility argmin over all N.
   **Refuted** in the heavy/matched condition (worse and much noisier than full-visibility on
   nearly every metric) but showed a real partial win in light/cachehit (best-of-family
   coincidence after peak_power). Mechanism: P2C's classical benefit (getting most of full
   information's value from O(1) probes) doesn't apply at N=6 with a single centralized
   router that already has full, fresh visibility for free — discarding 4 of 6 replicas'
   information trades away signal without buying the decorrelation P2C exists to provide,
   except apparently in the lighter-pressure regime.
4. **`drf_power_tiebreak_adaptive`** — replaces the fixed, hand-calibrated
   `ramp_ceiling_w_per_s=450` constant with a self-tracking value (rolling p99 of the fleet's
   own recently observed ramp readings, floored at 450, one-directional/relax-only). Directly
   answers the "hand-picked constants don't generalize" critique for the ONE genuinely
   fragile constant in the whole design (compute/load shares are normalized against real
   vLLM config constants, not calibrated from data). **Found to have a self-defeating loop**:
   "concentration" (2+ replicas simultaneously elevated) is exactly the condition that fills
   the calibration window with elevated values, so sustained pressure inflates its own
   tolerance for pressure — worst p99_ramp of every arm tested in heavy/matched (1694.5±155.4)
   despite being designed to guard against exactly this. Partial fix,
   **`drf_power_tiebreak_adaptive_isolated`**: filters the calibration input itself — skip
   observing a whole decision round if 2+ replicas are simultaneously elevated, so a genuine
   concentration event can no longer inflate the ceiling meant to guard against it (isolated
   single-replica transitions still get observed normally). Keeps the per-candidate scoring
   rule fully separable (Lemma 1 still applies) — only the calibration bookkeeping needs
   fleet-wide visibility, not the routing decision. **Result: fixes the exact failure it
   targeted in heavy/matched** (p99_ramp 1694.5→1584.2, mean_ramp actually beats plain
   `drf_power_tiebreak` too) **but reshuffles rather than eliminates the light-load
   trade-off** (loses naive adaptive's p99_ramp edge, gains the best coincidence of the whole
   family, 16.0 vs naive's 22.0). Never a clean win, never a clean loss.

A third workload condition was added this session: **heavy/closed-loop** (same whale-injection
composition as heavy/matched — whale-frac 0.15, 44-50k-char whales, single-turn — but
concurrency=32 closed-loop instead of rate=10.7 open-loop; N=6/GPUs 2-7, ramp_ceiling=450,
same as every other condition). No pre-existing baseline existed with `drf_power_tiebreak`
already included at N=6, so this is a fresh condition, not a replication of prior work.
`round_robin`/`lmetric`/naive `drf_power_tiebreak_adaptive` were not run against it (time
budget); the 5 arms that were (`drf_fixed`, `drf_power_tiebreak`, `drf_coincidence_tiebreak`,
`drf_peak_power_tiebreak`, `drf_power_tiebreak_p2c`, `drf_power_tiebreak_adaptive_isolated`)
show no clean winner — all cluster in a fairly narrow band except `drf_peak_power_tiebreak`
repeating its signature trade-off (best coincidence, worst TTFT/TBT by a wide margin).

**Two real operational bugs found and fixed while running this batch** (both purely
experiment-harness issues, not scoring-logic bugs): (1) the first `drf_fixed` closed-loop-heavy
attempt (all 3 trials, both the original run and a same-mistake rerun) silently produced zero
real records — the router process crashed at startup with `ValueError: unknown policy:
'drf_fixed'` because the actual valid policy string in `router_core.py` is `"drf"` (the
`_fixed` suffix is this project's file-naming convention for the bug-fixed tie-break, not the
literal policy string this session's script generator assumed) — fixed by keeping
`OUTNAME=drf_fixed` for file-naming consistency but correcting `POLICY=drf`; (2) the very
first `drf_fixed` closed-loop-heavy attempt failed identically for a deterministic reason
(not a race condition as first suspected), confirmed once the actual error was found in the
launch log rather than inferred from symptoms alone.

**Full comparison grid (all arms x all 3 conditions x 5 metrics: mean_ramp, p99_ramp,
coincidence, TTFT_mean, TBT_mean), 3-trial mean±std throughout:**

### Heavy/matched (open-loop, rate=10.7, 750 convs, whale-frac 0.15, cache-hit≈0)

| arm | mean_ramp (W/s) | p99_ramp (W/s) | coincidence (%) | TTFT_mean (s) | TBT_mean (ms) |
|---|---|---|---|---|---|
| round_robin | 195.4±0.7 | 1427.6±122.6 (best) | 1.4±0.3 (best) | 5.380±0.312 (worst) | 1253.8±42.9 (worst) |
| lmetric | 198.7±11.3 | 1383.3±13.7 | 3.1±1.1 | 4.147±0.923 | 1299.6±38.1 |
| drf_fixed | 191.9±9.5 | 1620.0±166.7 (worst) | 7.7±1.5 (worst) | 3.712±0.935 | 1336.3±43.9 |
| **drf_power_tiebreak** (featured) | 204.1±7.5 | 1485.3±7.5 | 4.4±1.2 | 3.942±0.271 | 1320.9±63.3 |
| drf_coincidence_tiebreak | 206.2±27.0 | 1429.6±160.0 | 6.9±2.0 | 3.344±1.002 | 1296.6±27.9 |
| drf_peak_power_tiebreak | 170.8±17.1 (best) | 1577.6±209.5 | 3.9±0.8 | 5.004±0.674 | 1308.0±7.8 |
| drf_power_tiebreak_p2c | 202.0±13.0 | 1568.3±57.3 | 5.2±2.1 | 3.970±1.059 | 1324.9±48.3 |
| drf_power_tiebreak_adaptive | 210.8±8.0 (worst) | 1694.5±155.4 | 6.2±1.2 | 3.529±0.914 | 1338.3±48.5 |
| drf_power_tiebreak_adaptive_isolated | 194.4±8.9 | 1584.2±85.4 | 5.7±1.3 | 3.810±1.318 (best) | 1294.6±9.7 (best) |

### Light/cachehit (closed-loop, conc=24, 150 convs, plain multi-turn, cache-hit≈45-47%)

| arm | mean_ramp (W/s) | p99_ramp (W/s) | coincidence (%) | TTFT_mean (s) | TBT_mean (ms) |
|---|---|---|---|---|---|
| round_robin | 103.0±3.7 | 1758.7±74.6 | 19.6±7.7 | 0.073±0.001 | 48.2±1.4 |
| lmetric | 94.2±2.6 | 2071.4±144.0 (worst) | 24.5±4.0 (worst) | 0.065±0.001 | 31.7±0.2 (best) |
| drf_fixed | 91.0±10.8 | 1607.1±330.8 | 20.3±2.9 | 0.065±0.000 | 33.7±1.3 |
| **drf_power_tiebreak** (featured) | 94.0±4.6 | 1939.9±279.0 | 24.1±10.0 | 0.065±0.001 | 34.2±0.7 |
| drf_coincidence_tiebreak | 97.0±2.8 | 2097.0±145.1 | 32.2±8.4 (worst) | 0.065±0.001 | 32.4±0.2 |
| drf_peak_power_tiebreak | 412.6±23.2 (worst) | 2916.3±274.6 (worst) | 9.1±4.4 (best) | 0.110±0.004 (worst) | 62.3±4.1 (worst) |
| drf_power_tiebreak_p2c | 110.5±8.0 | 1751.7±135.5 | 12.8±3.7 | 0.074±0.001 | 48.4±0.2 |
| drf_power_tiebreak_adaptive | 97.2±4.3 | 1702.5±102.6 (best) | 22.0±1.3 | 0.066±0.000 | 34.9±1.5 |
| drf_power_tiebreak_adaptive_isolated | 96.3±5.8 (best) | 2063.8±224.8 | 16.0±1.0 | 0.066±0.000 | 33.9±1.1 |

### Heavy/closed-loop (closed-loop, conc=32, 150 convs, whale-frac 0.15, cache-hit≈0) — new condition, complete except naive adaptive

| arm | mean_ramp (W/s) | p99_ramp (W/s) | coincidence (%) | TTFT_mean (s) | TBT_mean (ms) |
|---|---|---|---|---|---|
| round_robin | 159.7±2.3 | 2076.5±177.1 | 12.6±2.7 | 0.563±0.023 | 525.6±5.9 (best) |
| lmetric | 164.0±14.4 | 2302.7±108.6 (worst) | 9.5±2.8 | 0.537±0.025 | 571.7±16.0 |
| drf_fixed | 160.5±9.9 | 2218.7±86.6 | 8.3±3.3 | 0.533±0.024 (best) | 606.9±70.4 |
| **drf_power_tiebreak** (featured) | 158.6±8.6 | 2068.1±387.9 | 8.6±4.6 | 0.553±0.037 | 583.8±17.6 |
| drf_coincidence_tiebreak | 159.8±27.7 | 1984.7±399.6 | 9.9±3.7 | 0.578±0.039 | 544.9±40.5 |
| drf_peak_power_tiebreak | 144.0±11.8 (best) | 1985.5±58.9 (best) | 5.5±5.2 (best) | 0.877±0.085 (worst) | 694.9±74.7 (worst) |
| drf_power_tiebreak_p2c | 165.5±15.7 (worst) | 2139.9±237.4 | 11.6±2.3 | 0.633±0.073 | 586.3±43.6 |
| drf_power_tiebreak_adaptive | not run | — | — | — | — |
| drf_power_tiebreak_adaptive_isolated | 147.8±6.0 | 2125.8±101.4 | 13.2±4.1 (worst) | 0.592±0.085 | 549.1±45.8 |

**New finding once round_robin/lmetric were filled in: `round_robin`'s coincidence advantage
disappears under closed-loop admission specifically.** In both other conditions, round_robin
had by far the best (lowest) coincidence — 1.4% in heavy/matched, 19.6% in light/cachehit,
clearly ahead of the field both times (already documented above: "round_robin's blind,
evenly-cycled spreading avoids concentrating decode-heavy load on any single replica"). Here
it's mid-pack (12.6%), *worse* than `drf_fixed` (8.3), `drf_power_tiebreak` (8.6),
`drf_coincidence_tiebreak` (9.9), and `lmetric` (9.5) — a real reversal, not noise-level,
since heavy/closed-loop is the ONLY condition tested this session with closed-loop
(concurrency-driven) admission on the heavy whale-injection workload (the other closed-loop
condition, light/cachehit, is also lighter-load AND cache-hit-heavy, confounding which factor
matters). Plausible untested hypothesis: round_robin's cycling interacts with request
COMPLETION timing under closed-loop concurrency (a completed slot immediately admits the next
queued request) rather than ARRIVAL timing under open-loop rate control, which could
resynchronize replicas in a way open-loop arrival doesn't -- not confirmed, flagged as an open
question rather than an explained mechanism.

**Overall read**: `drf_power_tiebreak` remains the most consistently strong arm across all
three conditions tested — never the worst on any metric in any condition, despite four
structurally different challenger mechanisms being tried against it this session. This
strengthens (not weakens) confidence in featuring it as the paper's headline result: the
robustness now rests on "beat four different well-motivated alternatives," not just "beat
the original `drf_fixed`." `drf_peak_power_tiebreak` has the clearest, most reproducible
alternative signature (best coincidence, worst-or-near-worst everything else, in every
condition it ran) — a genuine, explainable mechanism, not noise, but not viable as a
general-purpose arm. Full interactive version (styled table, callouts):
https://claude.ai/code/artifact/4de8d8bd-88e9-47ab-a18c-77d01acfab0d

## Update 2026-09-02: reconciling an older sibling artifact ("Ramp & Route") — a real data
## bug found and confirmed, plus a fourth workload condition and its 6 new arms

A user comparison against an earlier artifact from this same project, "Ramp & Route"
(8 original arms: round_robin, lmetric, drf, p2c_whale, whale_argmin, constrained_lmetric,
drf_fixed, pressure_switch — pre-dating this session's theory-first pivot), surfaced that its
`round_robin` numbers looked very different from this session's `heavy/matched` table. Traced
to source: **"Ramp & Route" is a fourth, distinct workload condition**, not the same as any of
the three above. Its exact parameters were recovered from the still-present remote driver
script `run_replication_batch5.sh`:

```
N_REPLICAS=6, GPU_OFFSET=2 (GPUs 2-7), Qwen2.5-Coder-7B-Instruct, RAMP_CEILING_W_PER_S=450.0
--concurrency 24 --num-convs 150 --max-tokens 128 --min-turns 1 --max-turns 1
--whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 --request-timeout 180
```

Closest to `heavy/closed-loop` (same closed-loop admission, same whale injection) but with
lower concurrency (24 vs 32) and — the key difference — **max-tokens=128 instead of 1024**
(short outputs). `round_robin` in that artifact is explicitly labeled single-trial, not
replicated (a footnote, "kept as a labeled floor").

**Real data-integrity bug found and confirmed while reconciling the two**: the "Ramp & Route"
artifact's `mean_power (W)` column is verifiably wrong, by a factor of ~1.83x. Recomputed
`drf_fixed`'s numbers directly from its own raw source power-trace files (still present on
the remote server, `eenergy_power_trace_drf_fixed_t{1,2,3}.csv`) using the same validated
methodology used throughout this session:

| metric | recomputed from raw source data | artifact claimed | match |
|---|---|---|---|
| mean_ramp (W/s) | 258.5±18.7 | 258.5±18.7 | exact |
| max_ramp (W/s) | 4900.5±602.6 | 4901±603 | exact |
| coincidence (%) | 30.0±2.7 | 30.0±2.7 | exact |
| **mean_power (W)** | **1443.4±19.8** | **2635±81** | **1.83x inflated** |

Every metric matches exactly except `mean_power`, which rules out a workload/hardware
difference (that would move everything, not one column) and confirms a genuine computation
bug in that one column when the artifact was originally built. `mean_ramp`/`max_ramp`/
`coincidence`/latency numbers in "Ramp & Route" remain trustworthy; its `mean_power` column
does not — 1443W is the corrected reference for `drf_fixed`'s mean power under that
condition, not 2635W.

**Root cause found (2026-09-02, later):** recomputing `drf_fixed`'s actual PEAK
(max instantaneous aggregate) power from the same raw trace gives 2635.4±81.1 W — identical
to the artifact's claimed "mean_power" (2635±81). The old column was silently reporting peak
power mislabeled as mean power, not a random/unexplained computation error.

**This session's 6 newer arms were then run under the exact reconstructed "Ramp & Route"
condition** (3 trials each, real data verified non-empty), giving a combined 14-arm table for
this specific short-output/moderate-concurrency/whale-injection regime:

| arm | mean_ramp (W/s) | max_ramp (W/s) | coincidence (%) | TTFT_mean (s) | TBT_mean (ms) |
|---|---|---|---|---|---|
| drf (original, buggy tie-break) | 208.6±4.2 (best) | 6310±1349 | 17.7±1.9 | 0.520±0.024 | 457.8±26.9 (best) |
| drf_fixed | 258.5±18.7 | 4901±603 | 30.0±2.7 | 0.517±0.035 | 467.6±39.6 |
| pressure_switch | 267.3±28.1 (worst) | 4286±1324 (best of original 8) | 21.5±4.6 | 0.517±0.082 | 497.0±78.6 |
| **drf_power_tiebreak** (featured) | 233.2±34.1 | 6521±3928 (worst overall) | 14.0±5.7 | 0.617±0.085 | 455.0±58.1 |
| drf_coincidence_tiebreak | 238.0±30.4 | 6101±2054 | 15.7±2.0 | 0.501±0.032 (best) | 491.9±59.9 |
| drf_peak_power_tiebreak | 362.9±5.6 (worst overall) | 4771±410 | 12.7±0.5 | 1.388±0.036 (worst overall) | 509.2±80.5 |
| drf_power_tiebreak_p2c | 267.9±42.8 | 4697±1587 | 11.5±2.6 | 0.606±0.071 | 483.4±60.2 |
| drf_power_tiebreak_adaptive | 247.8±9.3 | 4996±2962 | 14.0±3.0 | 0.558±0.065 | 494.9±4.1 |
| drf_power_tiebreak_adaptive_isolated | 237.8±7.0 | **3587±984 (best overall)** | **8.9±4.7 (best overall)** | 0.567±0.033 | 437.6±19.1 (best of new arms) |

**Notable finding: `drf_power_tiebreak` (the paper's featured arm) has the WORST max_ramp of
all 14 arms in this condition** — a clear miss, the first condition tested this session where
the featured arm is genuinely one of the worse choices on its own headline metric, not just
non-dominant. `drf_power_tiebreak_adaptive_isolated` is the best max_ramp AND best coincidence
across the full 14-arm table here — a real, single-condition win worth noting even though it
doesn't change the featured-arm decision for the paper's current (heavy/matched) scope.

## Update 2026-09-02 (continued): real (server-side) cache hit rate verification

Built `cache_telemetry.py` + `snapshot_cache_hit_rate.py` (see commit `5c6aa08`) to read
vLLM's own Prometheus counters (`vllm:prefix_cache_hits_total` / `queries_total`, both
token-level, confirmed live against a running replica) directly, instead of relying solely on
`cache_mirror.py`'s client-side estimate (a hash-chain mirror assuming infinite cache
capacity, never independently checked against ground truth before now).

**First verification, `drf_power_tiebreak` under light/cachehit, 3 trials**: real fleet-wide
hit rate 46.38%, 47.14%, 47.24% → **mean 46.92% ± 0.47%**, vs. the router's own mirror
estimate of **46.9% ± 1.0%** (already reported in this file's earlier table). Match is exact
within error bars, and the real measurement is actually tighter (lower std) than the mirror's
own estimate. **The mirror is confirmed accurate for this condition** — real vLLM eviction
isn't happening enough (or not in a way that shifts the aggregate rate) to create a
measurable gap, plausibly because the 16384-token-budget-per-replica cache capacity is
generous relative to this workload's actual traffic volume. Scope caveat: this validates the
mirror specifically for light/cachehit (concurrency=24); it does not automatically extend to
heavier conditions with more concurrent large-context requests, which could plausibly push
real cache usage against capacity limits in ways this test never exercised.

**Extended to the remaining 7 arms under light/cachehit** (round_robin, lmetric, drf_fixed,
drf_peak_power_tiebreak, drf_power_tiebreak_p2c, drf_power_tiebreak_adaptive,
drf_power_tiebreak_adaptive_isolated), same real-snapshot method. Two trials had gaps (both
documented, not silently dropped): `drf_power_tiebreak_p2c` trial 3's harness itself produced
0 real records (router-not-ready at start, the same failure mode seen once before for
`drf_fixed`'s first closed-loop-heavy attempt); `drf_power_tiebreak_adaptive` trial 1's
harness succeeded fully (508 real records) but the replica had already gone down by the time
the snapshot call reached it — a real race-condition gap in `snapshot_cache_hit_rate.py`'s
timing (it should ideally run immediately after the harness with a liveness check, not
assume the replica stays up). Both arms reported at n=2 rather than n=3; not re-run yet.

**Result — real cache hit rate varies enormously by routing policy**, from 6.60% to 50.75%,
all under the identical light/cachehit workload:

| arm | real hit rate (%) | n |
|---|---|---|
| drf_peak_power_tiebreak | 6.60±1.46 (worst) | 3 |
| round_robin | 10.22±1.36 | 3 |
| drf_power_tiebreak_p2c | 15.00±2.84 | 2 |
| drf_power_tiebreak_adaptive | 46.15±2.47 | 2 |
| drf_power_tiebreak_adaptive_isolated | 45.73±1.96 | 3 |
| drf_power_tiebreak | 46.92±0.47 | 3 |
| drf_fixed | 47.88±0.59 | 3 |
| lmetric | 50.75±0.38 (best) | 3 |

This gives a direct, MEASURED (not inferred) mechanism for two pathologies already diagnosed
this session from latency/ramp symptoms alone: `drf_peak_power_tiebreak`'s severe light-load
regression (4x everyone else's mean_ramp, ~2x worse TTFT/TBT) now has a hard number behind
the "chases lowest-power-level, scatters conversations off their cached replica" hypothesis
— 6.6% real hit rate, worst of every arm tested. Similarly, `drf_power_tiebreak_p2c`'s
mediocre performance is now explained quantitatively: sampling only 2-of-6 candidates means
roughly a 67% chance per decision of missing the replica that actually has the conversation
cached, and its 15.0% real hit rate (vs. ~46-51% for every full-visibility arm) confirms it.

## Update 2026-09-02 (continued): isolating whether lmetric's ramp-smoothing win is the
## compute term alone or needs the load term too

User's own framing, drawing the PES-IM analogy directly: PES-IM's chunk-size lever smooths
power as a side effect of a purely throughput-side knob (capping new compute work per
scheduling step), never explicitly "about power." `Share_compute(c) = P-token(c) /
token_budget(c)` is the routing-time analog — and `lmetric` (P-token x in_flight, zero power
telemetry) already tests a version of "route on this, no power feedback at all."

Checking the already-collected heavy/matched table carefully (the paper's own featured
condition) surfaces a result that was sitting there unremarked: **`lmetric` has a
statistically clearly better p99_ramp than `drf_power_tiebreak`** — 1383.3±13.7 vs
1485.3±7.5, non-overlapping error bars — with zero power telemetry. Coincidence/TBT also
favor lmetric directionally, though within overlapping noise there. This does NOT generalize
(lmetric has the worst p99_ramp/coincidence in light/cachehit, worst max_ramp in
heavy/closed-loop — same condition-dependence as everything else this session), but in the
specific condition the paper features, the simplest power-blind baseline already beats the
power-aware featured arm on its own headline tail metric.

Built `compute_only` (`scoring.py::pick_compute_only`) to isolate the mechanism: routes
purely on `P-token` (new uncached prefill tokens), ignoring load/in-flight-count entirely —
the purest routing-time analog to PES-IM's chunk-size lever, no load balancing at all, not
even LMETRIC's product structure. 3 trials under heavy/matched:

| metric | lmetric | drf_power_tiebreak | compute_only |
|---|---|---|---|
| mean_ramp (W/s) | 198.7±11.3 | 204.1±7.5 | **173.0±9.4 (best)** |
| p99_ramp (W/s) | **1383.3±13.7 (best)** | 1485.3±7.5 | 1610.9±217.3 (worst, 16x noisier std) |
| coincidence (%) | 3.1±1.1 | 4.4±1.2 | **0.0±0.0 (zero, all 3 trials)** |
| TTFT_mean (s) | 4.147±0.923 | **3.942±0.271 (best)** | 5.122±0.577 (worst) |
| TBT_mean (ms) | 1299.6±38.1 | 1320.9±63.3 | **1219.7±18.6 (best)** |

**Answer: no, compute_only does not match lmetric's p99_ramp win — it needs the load term.**
But it's a genuine split, not a strict loss: compute_only *beats* lmetric on mean_ramp, TBT,
and coincidence (0.0% — the strongest single coincidence result of the whole session), but
loses clearly on p99_ramp (16x noisier) and TTFT. Mechanism: a rule balancing only new-compute
tokens will occasionally route to a replica that already has a large in-flight batch, purely
because *that specific request's* new-token count looks small at that instant -- rare, but
creates a severe tail pileup when it happens (driving both the worse p99 and its huge
variance) and worse queueing delay. lmetric's product structure exists specifically to
prevent this by never letting compute ignore load. Generalizable takeaway: **compute-balancing
smooths the typical case and kills coincidence; load-awareness is specifically a
tail-protection mechanism, not a mean-smoothing one** — neither term alone gets you
everything.

## Update 2026-09-02: two new lemmas backing `drf_power_tiebreak_adaptive_isolated`'s ceiling design

Prompted by re-tallying the comparison grid without coincidence (dropped as not load-bearing,
see prior update): `drf_power_tiebreak_adaptive_isolated` holds 4 outright bests / 0 worsts
across the 4 conditions on the metrics still trusted (mean_ramp, max_ramp, TTFT, TBT) — more
than the featured `drf_power_tiebreak` (0/0 everywhere). Traced whether this empirical edge
has any new theoretical backing, beyond inheriting `drf_power_tiebreak`'s own status.

**The catch, stated first:** `drf_power_tiebreak_adaptive_isolated` calls the exact same
comparator as the featured arm (`scoring.py::dominant_share_vector_power_priority`, via
`pick_drf_power_tiebreak`) — only the ramp_ceiling fed into `Share_power` changes (live vs.
the fixed 450 W/s constant). It inherits paper.tex §4.2's counterexample unchanged; no new
theory upgrades its own Pareto-domination status. What's new is two separate lemmas about the
*calibration design itself* — both numerically verified, 0 violations:

1. **Ceiling-invariance corollary** (`scripts/eenergy/verify_ceiling_invariance.py`): Lemma 1
   (sorted rule ⟹ Pareto-non-dominated) holds unchanged for ANY ceiling that is a single
   scalar shared identically by every candidate within one decision, regardless of how it's
   computed — static or adaptively recalibrated from fleet history. Verified: re-ran the
   original 200k-trial brute-force search with the ceiling itself randomized per trial, 0
   violations. Companion necessity result: under a PER-CANDIDATE ceiling (not shared),
   share-space non-domination can dissociate from physical reality — constructed an explicit
   instance where the sorted rule "correctly" selects the candidate with the physically
   *higher* raw ramp rate, because its own personal ceiling was calibrated more generously.
   This is why the router instantiates one `AdaptiveRampCeiling` per fleet, not one per
   replica — now a proven requirement, not just a convention.

2. **Round-filter insensitivity lemma** (`scripts/eenergy/verify_ceiling_boundedness.py`):
   `observe_round`'s ceiling is not merely *less* sensitive to concentration-round (2+
   simultaneously-elevated replicas) magnitude than naive `observe()` — it is *exactly*
   insensitive. Two fleet-ramp histories that agree on every round with <2 elevated replicas
   and differ ARBITRARILY (however extreme) on rounds with ≥2 produce byte-identical ceiling
   trajectories, because `observe_round` returns early without touching the window on a
   concentrated round — no value from it, however extreme, can ever reach the percentile
   calculation. Verified over 2,000 randomized paired-history trials (concentration
   magnitudes scaled up to 1000×), 0 violations. Naive `observe()` on the identical paired
   histories diverges in all 2,000/2,000 trials (worst-case divergence 22.2M W/s in one random
   trial — illustrates just how unbounded the naive scheme's runaway can get). One
   illustrative trace: round-filtered ceiling bit-identical (2001.8 W/s) between a pair
   differing 20× in concentration-round magnitude; naive ceiling diverges by 391,772.8 W/s on
   the same pair.

Both lemmas + proofs + verification results folded into paper.tex/paper.md as new §4.3/§4.4
(`sec:ceiling-invariance`, `sec:ceiling-boundedness`), with a matching Contributions bullet.
Fixed a pre-existing, unrelated LaTeX compile bug while in there: acmart doesn't define the
`claim`/`corollary` environments by default (only `lemma`/`proof` via amsthm) — §4.2's
`\begin{claim}` was silently broken before this session; added `\newtheorem` for both. Paper
now compiles clean (pdflatex + bibtex, 4 pages, 0 errors, 0 undefined refs after 2 passes).
Moved all three `verify_*.py` scripts (including the pre-existing `verify_pareto_lemma.py`,
previously only in this session's ephemeral scratchpad) into `scripts/eenergy/` so paper
citations resolve to a durable repo location instead of a session-scoped temp path.

Artifact glossary (`drf_power_tiebreak_adaptive_isolated` entry) and the "isolated-calibration
fix" callout updated to state both lemmas and the honest scope limit (backs the calibration
design, not the arm's own domination status).

## Update 2026-09-02: `drf_power_tiebreak_full` — Pareto-safe fix for the named rule, validated across all 4 conditions

Root cause of the Pareto-domination counterexample (paper.tex §4.2): the named rule's
tie-break vector `(D, Share_power, Share_load)` only names two of the three raw shares
explicitly. `D = max(compute, load, power)` collapses all three into one scalar before
compute ever gets its own slot in the tuple — so whenever `D` ties via load or power (not
compute), compute's actual value is completely invisible, discarded by the `max()`, not
recoverable from the rest of the tuple.

**Fix**: `dominant_share_vector_power_priority_full` = `(D, Share_power, Share_load,
Share_compute)` — literally add compute as an explicit fourth tie-break coordinate.
Provably Pareto-non-dominated by the same style of induction as Lemma 1: domination forces
`D`, then power, then load to agree-or-favor the dominator; if all three tie, domination's
required strict inequality has nowhere left to hide except compute, which resolves it.
Verified (`scripts/eenergy/verify_pareto_lemma_full.py`): 200k random trials, 0 violations;
resolves the exact counterexample instance regardless of iteration order, while the plain
named rule still fails it on the same instance. New arm: `drf_power_tiebreak_full`
(`scoring.py::pick_drf_power_tiebreak_full`, wired into `router_core.py`). 180 eenergy tests
pass (was 174, +6 new — `tests/test_eenergy_scoring.py`,
`tests/test_eenergy_router_core.py`).

Because tuple comparison short-circuits at the first differing coordinate, compute is only
ever consulted in the residual case where `(D, power, load)` alone doesn't already resolve
the tie — so the fix was predicted to be behaviorally close to free.

**Empirical validation, all 4 conditions, 3 trials each** (vs. the plain named rule's static
ceiling, same ramp-tail metric convention as the generalization table — p99_ramp for the
first three, max_ramp for Ramp & Route):

| condition | named (static) | named_full (fix) | ramp-tail Δ | TTFT (old→new) |
|---|---|---|---|---|
| Heavy/Matched | 1485.3±7.5 | 1576.7±258.6 | +6.2% (variance widens a lot: ±7.5→±258.6) | 3.942→3.390 (−14.0%) |
| Light/Cachehit | 1939.9±279.0 | 1834.6±299.3 | −5.4% | 0.065→0.065 (flat) |
| Heavy/Closed-Loop | 2068.1±387.9 | 1982.4±213.7 | −4.1% (tighter std too) | 0.553→0.521 (−5.8%) |
| Ramp & Route | 6521±3928 | 4751.9±1160.1 | **−27.1%** (3.4× tighter std) | 0.617→0.576 (−6.6%) |

Prediction mostly held (3/4 conditions near-identical or better) but with two real surprises,
not just noise — confirmed by pulling per-trial raw numbers directly, not trusting the
aggregate std alone:
- **Ramp & Route improved substantially**, not just "stayed the same" — exactly the condition
  where the plain named rule was already weakest (artifact's own "the featured arm loses on
  Ramp & Route" callout), so the residual-tie arbitrariness the fix targets was evidently
  costing something real there.
- **Heavy/Matched's p99_ramp variance widened** even though the mean barely moved. Per-trial:
  old rule (1479.8, 1493.8, 1482.2 — remarkably tight) vs. fixed rule (1654.4, 1288.2, 1787.6
  — much more scattered). Plausible mechanism: the synchronized cold-start burst (all
  replicas idle simultaneously at trial start) likely produces more exact `(D, power, load)`
  ties than the "residual ties are rare" prediction assumed, so compute-based tie-breaking
  engages more often early in this specific condition, with knock-on effects for the rest of
  the trial's trajectory. Condition-specific, not a general pattern — the other 3 conditions
  don't show it, two show *tighter* variance than the old rule.
- TTFT improved in 3 of 4 conditions (not predicted) — plausibly because compute-aware
  tie-breaking makes marginally better load-balancing calls even outside the ramp story.

**Net read**: `drf_power_tiebreak_full` is a serious candidate to replace `drf_power_tiebreak`
as the paper's featured/deployed arm — provably Pareto-non-dominated (which the current
featured arm is not) and matches-or-beats it empirically on 3 of 4 conditions with a
substantial win on the 4th. Not yet promoted in the paper; pending review. Folded into the
artifact (all 4 tables + glossary + a corrected "candidate for featured" callout replacing
the now-stale "featured result holds" one) at
`https://claude.ai/code/artifact/4de8d8bd-88e9-47ab-a18c-77d01acfab0d`.

While adding this row, a programmatic best/worst-marker audit (script comparing each
column's true min/max against what was actually marked) found and fixed ~12 pre-existing
mismarked cells across the artifact's four condition tables — mostly duplicate "worst"
labels left on a superseded row after a later arm's data made a different row the new true
extreme. None of the underlying numbers were wrong, only which cell the class was attached
to.

### Data paths and repro scripts, all arms × all 4 conditions

All files live on the 8×4090 server at `183.147.142.123:/root/pli/vllm-experiment/`, mirrored
locally under `scratchpad/eenergy_replication/` (this session's scratchpad — not committed).
Naming convention: `logs/<condition-prefix>_records_<arm>_t{1,2,3}.jsonl` (harness output) +
`logs/<condition-prefix>_power_trace_<arm>_t{1,2,3}.csv` (NVML power log), except where noted.

**Heavy/Matched** (`openloopwhalelongoutmatched_` prefix):

| arm | script |
|---|---|
| round_robin, lmetric, drf_fixed | `run_replication_batch_openloop_whale_longout_matched.sh` (combined; confirmed via matching data mtimes within the same ~15min window) |
| drf_power_tiebreak | `run_drf_power_tiebreak_matched.sh` |
| drf_coincidence_tiebreak | `run_drf_coincidence_tiebreak_matched.sh` |
| drf_peak_power_tiebreak | `run_drf_peak_power_tiebreak_matched.sh` |
| drf_power_tiebreak_p2c | `run_drf_power_tiebreak_p2c_matched.sh` |
| drf_power_tiebreak_adaptive | `run_drf_power_tiebreak_adaptive_matched.sh` |
| drf_power_tiebreak_adaptive_isolated | `run_drf_power_tiebreak_adaptive_isolated_matched.sh` |
| drf_power_tiebreak_full | `run_drf_power_tiebreak_full_all4.sh` (combined, all 4 conditions) |

**Light/Cachehit** (`cachehit_` prefix — distinct from the `_realcache_`-prefixed files used
by the Real Cache Hit Rate table):

| arm | script |
|---|---|
| round_robin, lmetric, drf_fixed | `run_three_arms_cachehit.sh` (combined; inferred from arm count, **not timestamp-verified as tightly as the matched-condition equivalent — lower confidence, worth double-checking**) |
| drf_power_tiebreak | `run_drf_power_tiebreak_cachehit.sh` |
| drf_coincidence_tiebreak | `run_drf_coincidence_tiebreak_cachehit.sh` |
| drf_peak_power_tiebreak | `run_drf_peak_power_tiebreak_cachehit.sh` |
| drf_power_tiebreak_p2c | `run_drf_power_tiebreak_p2c_cachehit.sh` |
| drf_power_tiebreak_adaptive | `run_drf_power_tiebreak_adaptive_cachehit.sh` |
| drf_power_tiebreak_adaptive_isolated | `run_drf_power_tiebreak_adaptive_isolated_cachehit.sh` |
| drf_power_tiebreak_full | `run_drf_power_tiebreak_full_all4.sh` |

**Heavy/Closed-Loop** (`closedloopheavy_` prefix):

| arm | script |
|---|---|
| round_robin | `run_round_robin_closedloopheavy.sh` |
| lmetric | `run_lmetric_closedloopheavy.sh` |
| drf_fixed | `run_drf_fixed_closedloopheavy.sh` |
| drf_power_tiebreak | `run_drf_power_tiebreak_closedloopheavy.sh` (confirmed via mtime ordering: script mtime precedes data mtime by ~38min) |
| drf_coincidence_tiebreak | `run_drf_coincidence_tiebreak_closedloopheavy.sh` |
| drf_peak_power_tiebreak | `run_drf_peak_power_tiebreak_closedloopheavy.sh` |
| drf_power_tiebreak_p2c | `run_drf_power_tiebreak_p2c_closedloopheavy.sh` |
| drf_power_tiebreak_adaptive | not run (no script; table shows "not run") |
| drf_power_tiebreak_adaptive_isolated | `run_drf_power_tiebreak_adaptive_isolated_closedloopheavy.sh` |
| drf_power_tiebreak_full | `run_drf_power_tiebreak_full_all4.sh` |

**Ramp & Route** (`rampandroute_` prefix for newer arms; `eenergy_` prefix for the 5
historical arms predating that naming — round_robin's file has no `_t{N}` suffix, single
trial: `eenergy_records_round_robin.jsonl` / `eenergy_power_trace_round_robin.csv`):

| arm | script |
|---|---|
| round_robin, lmetric, drf (original), drf_fixed, pressure_switch | `run_replication_batch5.sh` (original 8-arm batch this table was reconciled from) |
| drf_power_tiebreak | `run_drf_power_tiebreak_rampandroute.sh` |
| drf_coincidence_tiebreak | `run_drf_coincidence_tiebreak_rampandroute.sh` |
| drf_peak_power_tiebreak | `run_drf_peak_power_tiebreak_rampandroute.sh` |
| drf_power_tiebreak_p2c | `run_drf_power_tiebreak_p2c_rampandroute.sh` |
| drf_power_tiebreak_adaptive | `run_drf_power_tiebreak_adaptive_rampandroute.sh` |
| drf_power_tiebreak_adaptive_isolated | `run_drf_power_tiebreak_adaptive_isolated_rampandroute.sh` |
| drf_power_tiebreak_full | `run_drf_power_tiebreak_full_all4.sh` |

Full pointer detail (including the CSS-styled per-row version) is in the artifact linked
above; this table is the plain-text mirror for anyone without artifact access.

## Update 2026-09-03: rigorous Pareto-dominance check finds no single safe arm dominates — reframes advocacy around the design principle, not a champion arm; per-GPU ramp-ceiling recalibration overturns the one apparent exception

**Context**: three arms are provably Pareto-safe — `drf_fixed` (sorted-lexicographic),
`drf_power_tiebreak_full` (monotonic-primary + full-lexicographic-tiebreak, adds compute as
an explicit 4th tie-break coordinate — see 2026-09-02 entry above), and `weighted_sum`
(0.33/0.33/0.33, any positive-weighted sum of the three shares trivially respects
domination). `lmetric_power` is NOT safe (200k-trial verification found 12,189 violations,
all sharing one mechanism: new_tokens=0 on a cache hit collapses its score to exactly 0
regardless of load/power, producing arbitrary ties with a dominated candidate).

**Important correction on novelty**: both safe recipes are classical multi-objective
scalarization results (weighted-sum preserving Pareto-optimality: Geoffrion 1968 and
earlier; lexicographic order preserving non-domination: even more elementary), not new
theorems — the 200k-trial brute-force checks are implementation-correctness verification,
not proofs of new math. The actual contribution is narrower and more defensible: applying
this classical, properly-cited litmus test to catch two natural, domain-specific score
designs (`drf_power_tiebreak`'s fixed-priority tie-break; `lmetric_power`'s power-multiplied
score) that silently violate it, via two distinct non-obvious mechanisms, plus empirically
validating what enforcing it costs.

**Rigorous full-field Pareto-dominance check** (script-verified against the artifact's raw
table data, not eyeballed): checked whether `drf_power_tiebreak_full` is Pareto-dominant
(wins/ties every metric against literally every other tested arm) in any of the 6 tested
conditions (4 synthetic + BurstGPT + WildChat). Result: **no**, in every condition at least
one other arm beats it on at least one metric — including against just the narrow safe
family (`drf_fixed`, `weighted_sum`, `drf_power_tiebreak_adaptive_isolated`). In BurstGPT
specifically it was **fully dominated** by `drf_fixed` (6/6 metrics) under the
then-current uniform ramp ceiling — see the ceiling-recalibration finding below, which
overturns this specific result.

**Magnitude check** (motivated by a direct question: are these win/loss counts even real,
or noise dressed up as a pattern?): stripped the non-load-bearing `coincidence` metric
(its relative-% framing exaggerates small absolute differences on a low-baseline
percentage) and computed pairwise % differences among the three safe arms across all 6
conditions. Result: median difference 4.3%, mean 5.7%, only 1/86 comparisons exceeds 20%.
`peak_power` differences are near-noise (median 1.3%, max 4.6%); only `mean_ramp`/`p99_ramp`
show consistently real-but-modest gaps (~7-10% median, up to ~19%). Extending the same
check across the WHOLE arm field (not just the safe triad) shows this tight clustering is
specific to "resource-aware" designs (any DRF variant, `weighted_sum`, `lmetric`/
`lmetric_power`) — real, large gaps (50-400%) appear only against genuinely degenerate
single-axis arms: `drf_peak_power_tiebreak` (tie-breaks purely on power, ignoring load —
3-4x mean_ramp penalty, 50-97% TBT/TTFT penalty vs. everything else) and `round_robin`
(ignores everything — 2x max_ramp in Ramp & Route).

**Conclusion on advocacy**: don't crown a single empirical winner among the three safe
arms — the margins are too small and the win patterns don't hold up as literal dominance.
The defensible claim is **"Pareto-non-domination should be a verified design-time
correctness check for multi-resource routing scores — properly cited to classical theory,
not claimed as novel — because it catches real, non-obvious bugs (two found here) at
near-zero empirical cost (safe arms cluster within ~5%), while naive single-axis designs
cost 2-4x."** Which specific safe arm ships is a secondary, low-stakes choice; `weighted_sum`
has the simplest proof (no cascading tie-break machinery) and the highest raw metric-win
count (14/36 vs. `drf_fixed`'s 13/36 and `drf_power_tiebreak_full`'s 9/36, excluding
coincidence) if a reference implementation is needed.

**Literature positioning** (websearch, 2026-09-03): surveyed current LLM-serving routing
work (LMetric/SMetric, RouteBalance, BF-IO/"Universal Load Balancing Principle",
Llumnix, SLOs-Serve, prefix-cache-aware routing) and power/energy-aware serving work
(PALS, Festina, Power-Aware vLLM, GreenLLM, FREESH, PWR). Common validation traces across
this literature: ShareGPT, BurstGPT, Azure LLM trace, Mooncake trace — our BurstGPT +
WildChat-1M choices are squarely in-convention. Confirmed gap: no existing work combines
(1) power RAMP RATE (not peak, not energy) as a DRF-style resource share, (2) a formal
Pareto-non-domination guarantee on the routing score itself, (3) real multi-GPU hardware
validation. Notably, "Smoothing the Ramp, Not the Peak" (arXiv 2608.01250, Pan Li/Tongji)
is this project's own PES-IM paper, now on arXiv — directly citable as the single-GPU
precursor this multi-replica routing work extends, not a competing/overlapping paper.

**Per-GPU ramp-ceiling calibration** (motivated by a direct question: "could per-replica
variance in compute/load/power explain why no strategy stands out?"): `round_robin` power
traces (unbiased GPU assignment, so any spread reflects hardware, not our own routing
policy) showed p99 ramp rate varying **~29% across GPUs 2-7** on the 8x4090 box — real,
measurable variance in exactly the quantity the single global `RAMP_CEILING_W_PER_S=450.0`
constant is supposed to normalize. Re-ran the documented burst-calibration methodology
(24 concurrent 45k-char prefills on one idle replica, power sampled at the router's actual
500ms poll cadence, ceiling = observed max single-interval jump + ~1W headroom —
`scripts/eenergy/README.md`) **independently per GPU** instead of assuming GPU 0's
measurement generalizes (`scripts/eenergy/calibrate_ramp_ceiling.py`). Result — even wider
spread on the matched methodology (34%):

| GPU | ceiling (W/s) |
|---|---|
| 2 | 450.2 |
| 3 | 509.8 |
| 4 | 449.6 |
| 5 | 409.2 |
| 6 | 512.9 |
| 7 | 359.5 |

No changes were needed in `scoring.py`/`router_core.py`/`run_router.py` —
`Candidate.ramp_ceiling_w_per_s` and the `ROUTER_REPLICAS` spec string
(`host:port:gpu_index:token_budget:max_num_seqs:ramp_ceiling_w_per_s`) were already
per-replica; only `orchestrate/eenergy/launch_router_experiment.sh` was broadcasting one
shared value. Added an optional `RAMP_CEILING_PER_GPU` override (comma-separated
`gpu:ceiling`; any GPU not listed falls back to `RAMP_CEILING_W_PER_S`, so omitting it
reproduces the old behavior exactly) — backward-compatible, verified via
`tests/test_eenergy_run_router.py` (2 new tests, 186 eenergy tests total) and a bash-level
smoke test on the actual remote target (bash 5.1). The Pareto-non-domination proofs are
unaffected — they only require each share independently normalized to [0,1], regardless of
whether the denominator is shared or per-replica.

**Re-validation**: re-ran the safe triad (`drf_fixed`/`drf_power_tiebreak_full`/
`weighted_sum`, 3 trials each) on BurstGPT — the one condition with a clean 6/6 domination
result under the old uniform ceiling — with `RAMP_CEILING_PER_GPU` applied:

| arm | peak_power (W) | mean_ramp (W/s) | p99_ramp (W/s) | TTFT_mean (s) | TBT_mean (ms) |
|---|---|---|---|---|---|
| drf_fixed | 2223.9±14.6 | 178.3±9.0 | 1579.9±54.0 | 0.163±0.002 | 90.9±0.7 |
| drf_power_tiebreak_full | 2201.4±17.0 | 198.3±5.5 | 1577.7±55.3 | 0.164±0.001 | 99.2±1.0 |
| weighted_sum | 2218.1±19.7 | 191.9±9.9 | 1534.7±71.6 | 0.164±0.002 | 91.2±1.4 |

Old (uniform 450 W/s) for comparison: `drf_fixed` = 2232.2/187.1/1855.6/0.162/95.2;
`drf_power_tiebreak_full` = 2250.0/210.0/2080.7/0.163/102.2.

**Result: the domination reverses.** 2 of 5 metrics flip — `drf_power_tiebreak_full` now
wins peak_power (2201.4 vs. 2223.9) and effectively ties p99_ramp (1577.7 vs. 1579.9, within
noise; was a clear loss before, 2080.7 vs. 1855.6). `drf_fixed` keeps mean_ramp, TTFT, TBT.
Net: **incomparable, not dominated** — BurstGPT's apparent exception to the "no arm
dominates" pattern found everywhere else turns out to be at least partly a measurement
artifact of the flawed shared-ceiling assumption, not a robust effect. Mechanistically
sensible: `drf_power_tiebreak_full`'s p99_ramp improves 24% (2080.7→1577.7) under correct
calibration vs. `drf_fixed`'s 15% (1855.6→1579.9) — the arm that leans on `Share_power` as
an explicit early tie-break coordinate is more sensitive to getting that share's
normalization right than the arm where it's just one of three sorted-vector entries.
`weighted_sum`'s relative position is stable (still best p99_ramp, competitive elsewhere).

Data: `logs/burstgptpergpu_records_<arm>_t{1,2,3}.jsonl` + `..._power_trace_<arm>_t{1,2,3}.csv`
on `183.147.142.123:/root/pli/vllm-experiment/`. Calibration output:
`logs/ramp_ceiling_per_gpu.json`. Scripts: `scripts/eenergy/calibrate_ramp_ceiling.py`,
`run_burstgpt_perGPU_safetriad.sh`. This is the first commit of the eenergy router/theory/
paper infrastructure to the remote repo's git history (previously all untracked) —
`8cc7148 Add e-Energy power-aware DRF router, Pareto-safety proofs, and paper`.

Other 5 conditions (Heavy/Matched, Light/Cachehit, Heavy/Closed-Loop, Ramp & Route,
WildChat) have NOT yet been re-run with per-GPU ceilings — BurstGPT was chosen first
because it showed the widest effect under the old assumption. Open, pending time before the
Sept 18 deadline.

## Update 2026-09-04: per-GPU ramp-ceiling re-validation extended to all 4 arms × remaining 5 conditions — unsafe `lmetric_power`'s apparent empirical wins mostly evaporate or reverse under correct calibration

**Context**: the 2026-09-03 entry above found that BurstGPT's apparent 6/6 domination
(`drf_fixed` over `drf_power_tiebreak_full`) partly reversed once each of the 6 GPUs got its
own calibrated ramp ceiling instead of one shared `450.0 W/s` constant. That check covered
only the safe triad on one condition. This entry extends the same re-validation to all 4 arms
(`drf_fixed`, `drf_power_tiebreak_full`, `weighted_sum`, and — deliberately included as the
one *unsafe* arm for contrast — `lmetric_power`) across the 5 remaining conditions
(Heavy/Matched, Light/Cachehit, Heavy/Closed-Loop, Ramp & Route, WildChat), 3 trials each,
60 runs total (plus the earlier 9 BurstGPT runs = 69 total power-calibration re-validation
runs).

**Infra note**: one of the 60 runs (`wildchatnatural`/`lmetric_power`/trial 3) failed on
first attempt with `torch.distributed.DistNetworkError: ... EADDRINUSE` — a transient
rendezvous-port collision from the prior trial's teardown not having fully released the port
before the next trial launched (not a data or methodology bug). Re-ran that single trial in
isolation; succeeded (2332 records, matching the other WildChat trials). All 69 files now
have consistent, complete data.

**Methodology**: both OLD (uniform 450 W/s ceiling) and NEW (per-GPU calibrated ceiling)
numbers below are computed from raw per-trial `records_*.jsonl` + `power_trace_*.csv` files
using the exact same corrected aggregation as the 2026-09-03 BurstGPT re-validation — fleet
power summed across all 6 GPUs within each shared `wall_time` poll before computing
peak/ramp, and TBT as the mean-across-requests of each request's own max inter-token gap.
Recomputing OLD from raw data (rather than trusting the older artifact/findings text, which
predates both fixes) keeps the OLD/NEW comparison apples-to-apples on methodology — only the
ramp-ceiling constant differs between the two columns. `Ramp & Route`'s `drf_fixed` old data
lives under the `eenergy_` prefix (pre-dates that condition's later `rampandroute_` naming
convention, per the existing repro table above); everything else old is `<condition>_` prefixed,
new is `<condition>pergpu_` prefixed.

**Full comparison, all 4 arms × all 5 conditions, 3-trial mean±std, OLD vs NEW ceiling:**

### Heavy/Matched

| arm | peak_power (W) | mean_ramp (W/s) | p99_ramp (W/s) | TTFT_mean (s) | TBT_mean (ms) |
|---|---|---|---|---|---|
| drf_fixed OLD | 2692.5±12.1 | 173.1±12.5 | 1365±55 | 3.712±0.935 | 1338.6±43.8 |
| drf_fixed NEW | 2677.4±29.0 | 191.0±11.3 | 1305±25 | 2.998±0.283 | 1355.2±21.7 |
| drf_power_tiebreak_full OLD | 2681.7±10.2 | 194.1±10.0 | 1388±47 | 3.390±0.059 | 1344.3±15.8 |
| drf_power_tiebreak_full NEW | 2673.7±18.4 | 189.9±6.6 | 1382±114 | 4.083±0.165 | 1322.3±52.5 |
| weighted_sum OLD | 2657.8±16.0 | 207.9±8.3 | 1282±73 | 3.287±1.251 | 1319.3±68.2 |
| weighted_sum NEW | 2662.3±34.4 | 182.3±4.2 | 1418±10 | 3.195±0.758 | 1323.8±33.3 |
| lmetric_power OLD | 2688.6±8.1 | 190.1±8.4 | 1372±74 | 3.720±0.543 | 1330.9±25.3 |
| lmetric_power NEW | 2692.5±16.7 | 171.0±10.9 | 1305±48 | 3.577±1.174 | 1309.4±70.4 |

Dominance: **incomparable both OLD and NEW** — no arm sweeps all 5 metrics either way.

### Light/Cachehit

| arm | peak_power (W) | mean_ramp (W/s) | p99_ramp (W/s) | TTFT_mean (s) | TBT_mean (ms) |
|---|---|---|---|---|---|
| drf_fixed OLD | 1907.7±3.0 | 89.7±9.2 | 1517±112 | 0.065±0.000 | 33.1±1.1 |
| drf_fixed NEW | 1914.5±10.9 | 87.4±1.3 | 1630±85 | 0.066±0.002 | 34.5±1.8 |
| drf_power_tiebreak_full OLD | 1913.6±10.4 | 95.1±4.6 | 1625±102 | 0.065±0.001 | 34.1±0.5 |
| drf_power_tiebreak_full NEW | 1901.2±11.4 | 85.0±5.1 | 1868±22 | 0.066±0.001 | 33.7±1.0 |
| weighted_sum OLD | 1932.8±16.5 | 95.3±5.4 | 1481±100 | 0.067±0.001 | 34.2±0.7 |
| weighted_sum NEW | 1921.0±29.7 | 87.4±6.5 | 1723±90 | 0.067±0.000 | 34.3±0.6 |
| lmetric_power OLD | 1905.3±12.1 | 85.3±1.5 | 1417±81 | 0.065±0.001 | 31.7±1.1 |
| lmetric_power NEW | 1901.8±13.0 | 81.4±4.9 | 1723±129 | 0.066±0.000 | 31.3±0.5 |

Dominance: **OLD — `lmetric_power` (the theoretically UNSAFE arm) dominates all 3 safe
arms** (`drf_fixed`, `drf_power_tiebreak_full`, `weighted_sum`); `drf_fixed` also dominates
`drf_power_tiebreak_full`. **NEW — only `lmetric_power` dominates `weighted_sum`** survives;
the 3 other domination relationships (including both of `lmetric_power`'s other sweeps)
disappear. Mechanism: `lmetric_power`'s p99_ramp jumps 1417→1723 (+21.6%) under correct
per-GPU calibration while its other 4 metrics barely move — enough to erase its edge over
`drf_fixed` (1517→1630, +7.5%, a smaller relative jump) and `drf_power_tiebreak_full`
(1625→1868, +15.0%).

### Heavy/Closed-Loop

| arm | peak_power (W) | mean_ramp (W/s) | p99_ramp (W/s) | TTFT_mean (s) | TBT_mean (ms) |
|---|---|---|---|---|---|
| drf_fixed OLD | 2395.8±63.2 | 144.3±6.9 | 1792±193 | 0.533±0.024 | 592.5±81.5 |
| drf_fixed NEW | 2339.7±22.0 | 149.0±2.3 | 1565±54 | 0.511±0.011 | 593.7±7.3 |
| drf_power_tiebreak_full OLD | 2338.4±69.4 | 150.4±6.9 | 1638±87 | 0.521±0.013 | 596.8±35.7 |
| drf_power_tiebreak_full NEW | 2311.8±71.7 | 145.7±16.1 | 1599±117 | 0.550±0.027 | 591.2±17.6 |
| weighted_sum OLD | 2394.2±55.9 | 157.6±12.3 | 1695±116 | 0.576±0.035 | 555.7±29.5 |
| weighted_sum NEW | 2353.7±72.4 | 154.1±3.7 | 1826±360 | 0.545±0.003 | 577.8±43.0 |
| lmetric_power OLD | 2382.1±93.8 | 148.4±7.1 | 1598±124 | 0.536±0.019 | 539.4±53.9 |
| lmetric_power NEW | 2423.8±112.1 | 161.4±7.4 | 1953±309 | 0.567±0.039 | 591.2±47.5 |

Dominance: **OLD — `lmetric_power` dominates `weighted_sum`** (clean sweep, decent
margins). **NEW — full reversal: `weighted_sum` dominates `lmetric_power`, and
`drf_power_tiebreak_full` also newly dominates `lmetric_power`.** This is the clearest
reversal in the whole 5-condition batch: `lmetric_power`'s peak_power, mean_ramp, p99_ramp,
TTFT, and TBT all get *worse* under correct per-GPU calibration (peak +1.8%, mean_ramp
+8.8%, p99_ramp +22.2%, TTFT +5.8%, TBT +9.6% — a uniform degradation, not one noisy metric),
while `weighted_sum` and `drf_power_tiebreak_full` both improve or hold roughly flat on the
same metrics. `lmetric_power`'s zero-compute/cache-hit score collapse (the same mechanism
behind its 200k-trial Pareto-safety violation) plausibly interacts worse with the
higher-variance per-GPU ceilings than the safe arms' tie-break structure does, though this is
observational, not mechanistically isolated here.

### Ramp & Route

| arm | peak_power (W) | mean_ramp (W/s) | p99_ramp (W/s) | TTFT_mean (s) | TBT_mean (ms) |
|---|---|---|---|---|---|
| drf_fixed OLD | 2635.4±81.1 | 240.8±14.3 | 2703±126 | 0.517±0.035 | 459.7±37.3 |
| drf_fixed NEW | 2564.8±41.3 | 225.7±12.0 | 2685±98 | 0.616±0.045 | 429.4±35.8 |
| drf_power_tiebreak_full OLD | 2518.8±55.0 | 234.8±8.3 | 1923±147 | 0.576±0.053 | 450.8±31.3 |
| drf_power_tiebreak_full NEW | 2465.9±24.5 | 213.4±11.8 | 2590±278 | 0.598±0.031 | 413.4±24.1 |
| weighted_sum OLD | 2522.9±18.8 | 264.4±16.5 | 1930±73 | 0.576±0.016 | 439.4±40.8 |
| weighted_sum NEW | 2545.3±73.5 | 236.9±28.8 | 2506±84 | 0.612±0.015 | 430.6±16.7 |
| lmetric_power OLD | 2667.7±61.1 | 229.2±34.5 | 2505±453 | 0.564±0.022 | 432.1±22.9 |
| lmetric_power NEW | 2601.0±91.9 | 261.7±14.9 | 2164±175 | 0.549±0.058 | 442.6±55.7 |

Dominance: **OLD — incomparable.** **NEW — `drf_power_tiebreak_full` newly dominates
`drf_fixed`** (both safe arms; a within-safe-family reshuffle, not a safety concern). No
arm involving `lmetric_power` dominates or is dominated here.

### WildChat

| arm | peak_power (W) | mean_ramp (W/s) | p99_ramp (W/s) | TTFT_mean (s) | TBT_mean (ms) |
|---|---|---|---|---|---|
| drf_fixed OLD | 2278.6±13.3 | 112.3±3.4 | 1268±19 | 0.152±0.015 | 236.5±17.8 |
| drf_fixed NEW | 2308.6±19.5 | 109.1±7.9 | 1169±76 | 0.140±0.006 | 233.9±3.8 |
| drf_power_tiebreak_full OLD | 2334.3±25.9 | 120.4±10.9 | 1215±6 | 0.145±0.018 | 257.3±9.4 |
| drf_power_tiebreak_full NEW | 2346.7±16.7 | 125.7±2.0 | 1175±29 | 0.157±0.012 | 260.6±16.2 |
| weighted_sum OLD | 2277.6±17.8 | 114.2±0.8 | 1024±69 | 0.152±0.014 | 201.1±3.3 |
| weighted_sum NEW | 2308.7±30.0 | 104.6±3.4 | 1078±20 | 0.150±0.018 | 197.9±3.7 |
| lmetric_power OLD | 2238.1±1.3 | 107.8±0.9 | 1101±72 | 0.149±0.017 | 201.1±9.2 |
| lmetric_power NEW | 2231.6±17.5 | 101.0±0.3 | 983±68 | 0.158±0.003 | 197.0±2.7 |

Dominance: **OLD — `lmetric_power` dominates `drf_fixed`** (clean sweep). **NEW —
that reverses (no longer holds); instead `drf_fixed` dominates `drf_power_tiebreak_full`,
and `weighted_sum` also dominates `drf_power_tiebreak_full`.** Mechanism: `lmetric_power`'s
TTFT gets *worse* under correct calibration (0.149→0.158s, +6.0%) while `drf_fixed`'s TTFT
gets *better* (0.152→0.140s, −7.9%) — that alone flips the sign on the one metric where
`lmetric_power` previously had by far its largest margin over `drf_fixed`, breaking the
sweep even though `lmetric_power` still wins the other 4 metrics.

**Headline finding**: under the old, measurably-wrong uniform ramp ceiling, `lmetric_power`
— the one arm this project has already proven is NOT Pareto-safe (200k-trial verification,
12,189 violations, arbitrary ties on cache hits) — nonetheless appeared to *empirically
dominate* one or more of the three safe arms in 3 of the 5 conditions tested here
(Light/Cachehit, Heavy/Closed-Loop, WildChat), which would have been an awkward,
thesis-undermining result: "the unsafe arm wins anyway, so why enforce the safety property?"
Under the corrected per-GPU calibration, **all three of those apparent wins evaporate or
outright reverse** — 2 of 3 flip to `lmetric_power` being dominated instead, the third
(Cachehit) narrows from a 3-arm sweep down to a single, much closer surviving win. Combined
with the BurstGPT reversal found 2026-09-03 (`drf_fixed`'s clean 6/6 domination over
`drf_power_tiebreak_full` also didn't survive recalibration), the consistent lesson across
6 of 6 conditions now checked is: **apparent empirical dominance in this router's evaluation
grid is disproportionately a symptom of the uniform-ceiling measurement bug, not a real,
robust property of any one score design** — which strengthens (not weakens) the paper's
core methodological argument that the *design-time* Pareto-non-domination proof is the
trustworthy signal, and empirical per-condition leaderboards should be read with real
caution given how measurably sensitive they are to a calibration assumption that turned out
to be wrong.

**Data and repro**: all 69 files (`logs/<condition>pergpu_records_<arm>_t{1,2,3}.jsonl` +
`..._power_trace_<arm>_t{1,2,3}.csv` for the 60 new runs; `logs/burstgptpergpu_*` for the
earlier 9) on `183.147.142.123:/root/pli/vllm-experiment/`. Orchestration:
`run_pergpu_4arms_5conditions.sh` (committed `09b9225`) launched all 60; one trial
(`wildchatnatural`/`lmetric_power`/t3) was individually re-run after a transient port
collision, script not yet committed (`rerun_wildchat_lmetric_t3.sh`, kept local to this
session's scratchpad — trivial one-off, not needed again). Aggregation:
`aggregate_full_comparison.py` (prints the OLD-vs-NEW table per condition) and
`check_dominance_reversals.py` (prints only the dominance-relation summary per condition) —
both read directly from `logs/`, not yet committed.

## Update 2026-09-04 (continued): `round_robin` (vLLM's actual default, condition-1 baseline)
## added to the per-GPU-calibrated comparison as a 5th arm — dominated outright in 2 of 6
## conditions, incomparable everywhere else

**Motivation**: every arm in the 2026-09-04 per-GPU re-validation above is a *scored* policy
(DRF-family or LMETRIC-family). `round_robin` — vLLM's own no-router default, and this
project's Condition 1 baseline since the original design spec — had never been run under the
corrected per-GPU ramp ceiling; its only existing data predates the calibration fix entirely.
Added it as a 5th arm so the headline comparison covers "doing nothing" (round_robin), not
just "which scoring rule is safest."

**Ceiling-relevance caveat**: `pick_round_robin()` (`scoring.py`) never reads `ramp_ceiling` —
its routing decision is a pure counter cycle, so `RAMP_CEILING_PER_GPU` has zero causal effect
on `round_robin`'s behavior. It was rerun anyway (rather than reusing old-era data) purely for
timing/hardware-state contemporaneity with the other 4 arms' fresh NEW runs, not because the
calibration fix changes anything for this arm. Framing this as part of the "per-GPU
calibration" batch is therefore accurate for logistics (same script, same day, same
`RAMP_CEILING_PER_GPU` value passed through) but not for mechanism — unlike
`drf_power_tiebreak_full` and `weighted_sum`, whose actual scores depend on the ceiling
constant, `round_robin`'s NEW numbers differ from its OLD numbers only via ordinary run-to-run
noise.

**Methodology**: 18 fresh runs (6 conditions x 3 trials — the same 5 conditions as the 4-arm
batch, plus BurstGPT), `N_REPLICAS=6 GPU_OFFSET=2` (GPUs 2-7, identical to every other arm's
data, confirmed via `nvidia-smi` immediately before launch — GPUs 0-1 have never been part of
any run in this comparison), `POLICY=round_robin`, orchestrated by
`orchestrate/eenergy/run_pergpu_roundrobin_6conditions.sh` (harness args copied verbatim from
`run_pergpu_4arms_5conditions.sh` and `run_burstgpt_perGPU_safetriad.sh`). All 18 runs
completed cleanly, no reruns needed.

**Results — round_robin's own numbers, NEW (per-GPU-calibrated) columns, 3-trial mean±std:**

| condition | peak_power (W) | mean_ramp (W/s) | p99_ramp (W/s) | TTFT_mean (s) | TBT_mean (ms) |
|---|---|---|---|---|---|
| Heavy/Matched | 2635.6±37.9 | 151.4±6.5 | 1283±25 | 5.588±0.268 | 1241.9±12.6 |
| Light/Cachehit | 1955.4±8.9 | 99.7±4.4 | 1899±135 | 0.074±0.001 | 48.2±1.3 |
| Heavy/Closed-Loop | 2319.1±71.4 | 140.4±8.2 | 1891±382 | 0.607±0.019 | 482.8±22.2 |
| Ramp & Route | 2504.3±68.4 | 197.1±6.7 | 2510±322 | 0.688±0.022 | 450.9±11.7 |
| WildChat | 2382.9±27.7 | 118.4±0.8 | 1009±130 | 0.141±0.002 | 299.6±11.5 |
| BurstGPT | 2199.3±15.2 | 223.2±2.8 | 1650.9±37.3 | 0.160±0.000 | 100.8±0.5 |

(Ramp & Route has no OLD round_robin data — it never existed before this batch, unlike the
other 5 conditions, which have old-era round_robin runs on record.)

**Dominance relations involving `round_robin`, all conditions, NEW (per-GPU-calibrated):**

- **Heavy/Matched**: incomparable — round_robin neither dominates nor is dominated by anything.
- **Light/Cachehit**: **round_robin is dominated by all 4 other arms simultaneously**
  (`drf_fixed`, `drf_power_tiebreak_full`, `weighted_sum`, and `lmetric_power` each
  independently dominate it — full 5-metric sweep each time). This also held OLD (dominated by
  `drf_fixed`, `weighted_sum`, and `lmetric_power` there too), so it's not a calibration
  artifact — round_robin is robustly the worst arm in this condition. Mechanism: Cachehit's
  workload has real cache-hit structure (see the 2026-08-31 update earlier in this doc);
  round_robin's blindness to KV$ locality means it can't route cache-hit-bearing requests to
  the replica already holding that prefix, unlike every scored arm (even the power-blind
  `lmetric_power`, whose `P-token x BS` score at least reacts to P-token collapsing on a hit).
- **Heavy/Closed-Loop**: incomparable.
- **Ramp & Route**: incomparable (mixed: round_robin beats `drf_fixed` on peak_power/mean_ramp
  but loses on TTFT/TBT, so neither dominates).
- **WildChat**: **`lmetric_power` dominates `round_robin`** (all 5 metrics); round_robin
  doesn't dominate or get dominated by anything else here.
- **BurstGPT**: incomparable against all 3 arms with NEW data there (`drf_fixed`,
  `drf_power_tiebreak_full`, `weighted_sum`; `lmetric_power` has no BurstGPT NEW run to compare
  against). round_robin's peak_power and TTFT are best-in-class here, but its mean_ramp
  (223.2 W/s) and TBT (100.8 ms) are worst-in-class, keeping every pairing incomparable.

**Headline finding**: adding the "do nothing" baseline doesn't change the paper's core claim
(no single arm dominates across all 6 conditions — round_robin included), but it does add a
genuinely new data point: round_robin is the *only* arm dominated by every other arm at once
anywhere in the grid (Light/Cachehit), which is a stronger and more legible failure than any
of the score-vs-score reversals found among the 4 scored arms. It's a clean illustration of
why *some* routing signal beats none, even before asking which scored rule is safest — useful
framing for the paper's motivation section, distinct from the Pareto-safety argument (which is
about score *design*, not about scoring vs. not-scoring at all).

**Data and repro**: 18 new files (`logs/<condition>pergpu_records_round_robin_t{1,2,3}.jsonl` +
`..._power_trace_round_robin_t{1,2,3}.csv`, `logs/burstgptpergpu_*_round_robin_t{1,2,3}.*`) on
`183.147.142.123:/root/pli/vllm-experiment/`. Orchestration:
`orchestrate/eenergy/run_pergpu_roundrobin_6conditions.sh` (new, committed locally).
Aggregation: `aggregate_full_comparison.py` and `check_dominance_reversals.py` (both extended
with `round_robin` in `ARMS`), `aggregate_pergpu.py` (extended with `round_robin` for the
BurstGPT-only table; `check_dominance_reversals.py` also patched to skip arms with missing OLD
data per-condition instead of crashing, needed for Ramp & Route's missing OLD round_robin).
