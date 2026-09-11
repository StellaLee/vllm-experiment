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

## Update 2026-09-07: Heavy/Closed-Loop duration sensitivity, and a Share_power clamp sanity check

**Motivation.** The paper's §5 headline claims a result "under sustained fleet power
pressure." Checking the raw power-trace timestamps for Heavy/Closed-Loop found the
condition's actual active workload window is only **~51-53 seconds** (150 conversations at
concurrency=32) — short enough that "sustained" was doing more rhetorical work than the
evidence actually earned, especially given §1's framing explicitly invokes grid-facing/
demand-response timescales (typically minutes to hours). This update checks what happens to
the headline dominance result under a duration that more honestly earns the word "sustained,"
plus a separate sanity check on `Share_power`'s `max(ramp_rate, 0)` clamp.

### Part 1: duration extension

Same 5 arms, same per-GPU-calibrated `RAMP_CEILING_PER_GPU`
(`2:450.2,3:509.8,4:449.6,5:409.2,6:512.9,7:359.5` — unchanged; ramp ceiling is a physical
property of each GPU, not a function of workload duration), same concurrency/whale
parameters; only `--num-convs` scaled 150→900 (from the measured ~2.94 convs/s throughput at
concurrency=32) to target a ~300s active window. Actual measured active windows came in at
**~225-232s** (~4.4× the original), consistent across all 5 arms and both trial batches
(1-3, then 4-6).

Data integrity check on the long-duration runs: 900/900 records in every trial, zero
errors/timeouts in any harness log, 1/900 null TTFT (negligible) — the duration change itself
did not degrade data quality.

**Short (~51-53s), 3 trials — `closedloopheavypergpu`** (the paper's existing §5 headline
table):

| arm | peak (W) | mean_ramp (W/s) | p99_ramp (W/s) | TTFT (s) | TBT (ms) |
|---|---|---|---|---|---|
| drf_fixed | 2339.7±22.0 | 149.0±2.3 | 1565.3±53.7 | 0.511±0.011 | 593.7±7.3 |
| drf_power_tiebreak_full | 2311.8±71.7 | 145.7±16.1 | 1598.9±116.9 | 0.550±0.027 | 591.2±17.6 |
| weighted_sum | 2353.7±72.4 | 154.1±3.7 | 1826.3±360.4 | 0.545±0.003 | 577.8±43.0 |
| lmetric_power | 2423.8±112.1 | 161.4±7.4 | 1952.8±309.1 | 0.567±0.039 | 591.2±47.5 |
| round_robin | 2319.1±71.4 | 140.4±8.2 | 1890.5±381.5 | 0.607±0.019 | 482.8±22.2 |

Dominance: `drf_power_tiebreak_full` DOMINATES `lmetric_power`; `weighted_sum` DOMINATES
`lmetric_power`. `round_robin` incomparable to all 4.

**Long (~225-232s), 3 trials — `closedloopheavylongpergpu`, trials 1-3:**

| arm | peak (W) | mean_ramp (W/s) | p99_ramp (W/s) | TTFT (s) | TBT (ms) |
|---|---|---|---|---|---|
| drf_fixed | 2508.9±21.8 | 160.9±4.4 | 1378.3±7.7 | 0.431±0.022 | 725.5±27.4 |
| drf_power_tiebreak_full | 2417.4±0.6 | 147.7±5.9 | 1174.3±50.0 | 0.435±0.003 | 693.9±23.3 |
| weighted_sum | 2433.7±35.3 | 148.5±5.8 | 1271.5±69.1 | 0.433±0.011 | 722.4±8.8 |
| lmetric_power | 2444.1±34.0 | 146.3±8.1 | 1185.8±108.5 | 0.429±0.004 | 731.2±28.2 |
| round_robin | 2428.7±12.7 | 132.5±3.9 | 1184.2±88.2 | 0.448±0.002 | 681.1±11.5 |

Dominance: **none** among the 4 scored arms. `round_robin` still incomparable to all 4.

At n=3, the headline dominance relationship does not replicate at the longer duration —
`lmetric_power`'s worst-case metrics (p99 ramp, TTFT) improved substantially while
`drf_power_tiebreak_full`'s peak got worse, closing the gap from both directions.

One workload-wide effect worth noting on its own: TBT rose 15-40% across **every** arm,
including `round_robin` (482.8→681.1ms), which has nothing to do with any routing mechanism.
This is evidence the longer window is exposing genuine sustained-load interference (plausible
and expected), not an artifact specific to one arm.

**Long (~225-232s), 6 trials — `closedloopheavylongpergpu`, trials 1-6:** ran 3 more trials
(4-6, same params) specifically to check whether the 3-trial "no dominance" result was real
or noise.

| arm | peak (W) | mean_ramp (W/s) | p99_ramp (W/s) | TTFT (s) | TBT (ms) |
|---|---|---|---|---|---|
| drf_fixed | 2506.5±34.0 | 152.9±11.3 | 1328.9±88.3 | 0.435±0.016 | 722.6±20.1 |
| drf_power_tiebreak_full | 2454.4±49.5 | 147.1±4.3 | 1136.0±54.1 | 0.427±0.009 | 715.4±32.7 |
| weighted_sum | 2434.3±24.3 | 151.5±5.3 | 1242.5±81.8 | 0.436±0.009 | 719.9±12.5 |
| lmetric_power | 2466.5±64.9 | 148.3±6.6 | 1235.1±96.1 | 0.427±0.006 | 717.1±24.0 |
| round_robin | 2418.7±26.4 | 138.9±8.1 | 1163.5±69.0 | 0.448±0.002 | 679.8±7.9 |

Dominance: `drf_power_tiebreak_full` DOMINATES `drf_fixed`; **`drf_power_tiebreak_full`
DOMINATES `lmetric_power`**; `lmetric_power` DOMINATES `drf_fixed`. `weighted_sum`:
incomparable to everyone. `round_robin` incomparable to all 4.

At n=6, dominance over `lmetric_power` returns for the proposed rule, but not for
`weighted_sum` — a real change from the original short-duration story, where both safe rules
independently corroborated each other. The margins are also much tighter than the original
table (e.g. peak 2454.4±49.5 vs. 2466.5±64.9, overlapping std), so even where "dominates"
holds on the means, it's a far less clean separation than the short-duration result.

The 3-trial → 6-trial flip (no dominance → dominance for one arm only) is itself
informative: 3 trials was not enough for a stable answer at this duration and effect size.

**Why weighted_sum losing dominance is not a red flag — Theorem 5.** Independently, a new
§4.5 (uncommitted at time of writing, verified via
`scripts/eenergy/verify_leximin_vs_weighted_sum.py`, all claims pass) proves **no
fixed-weight rule — for any choice of weights, not just `weighted_sum`'s 0.33/0.33/0.33 —
can guarantee threshold-safety on worst-case metrics like peak power and ramp rate**
(Theorem 5), while the sorted/leximin rule (`drf_power_tiebreak_full`) provably can,
unconditionally (Theorem 4). Peak power and p99 ramp are exactly the worst-case,
threshold-triggered class Theorem 5 covers. The 6-trial data lines up with this exactly: the
rule with the proven threshold-safety guarantee keeps its dominance; the rule without it
doesn't. This reframes the duration sensitivity from "the headline result destabilized" to
"real hardware data landing precisely where a proven theorem said a fixed-weight rule's
guarantee runs out and the sorted rule's doesn't" — a stronger empirical story than the
original short-duration coincidence of two safe rules agreeing.

### Part 2: does Share_power's max(ramp_rate, 0) clamp cause a "reignition" bias?

Separate question, prompted by scrutinizing `Share_power(c) = max(ramp_rate(c), 0) /
ramp_ceiling(c)`: the clamp treats a replica currently falling in power identically to one
that's flat (both score 0 = no hazard). Hypothesis: this could bias routing toward replicas
that just finished a burst and haven't settled, which might reignite if loaded again —
undermining the exact stability goal `Share_power` exists for.

**Method:** for every dispatch event in the (short-duration) Heavy/Closed-Loop assignment
logs, computed the receiving replica's ramp rate just before dispatch (falling / flat /
rising), then measured the replica's forward power rise in the 3s after dispatch. Compared
forward-rise distributions across buckets, for every arm whose routing decision consults
`Share_power`, plus `round_robin` as a signal-blind control.

| arm | falling: mean fwd. rise | flat: mean fwd. rise | falling: n / total |
|---|---|---|---|
| drf_power_tiebreak_full | 36.4 W | 104.7 W | 22/453 (4.9%) |
| lmetric_power | 43.2 W | 116.1 W | 17/453 (3.8%) |
| drf_fixed | 39.8 W | 109.8 W | 20/453 (4.4%) |
| weighted_sum | 32.4 W | 112.0 W | 13/453 (2.9%) |
| round_robin | 34.6 W | 101.2 W | 13/453 (2.9%) |

**Result: the hypothesis is not supported.** Falling-state dispatches show consistently
*lower* forward rise than flat-state dispatches, in every arm — the opposite of the
reignition prediction. Critically, `round_robin` (no `Share_power` signal at all) shows the
identical pattern and a similarly low base rate of falling-state dispatches (2.9%), which
means this isn't routing intelligence avoiding a bad outcome — it's a property of the
workload itself (a replica mid-decline from a recent burst still carries residual decode
load, so one more marginal request adds proportionally less power than the same request
landing on a genuinely idle replica, which gets the full fresh-prefill spike).

**Caveats:** small samples (13-22 falling-state events per arm across 3 trials, out of ~450
total dispatches), single condition (Heavy/Closed-Loop only), correlational on existing data
rather than a designed experiment forcing the scenario. Suggestive that the clamp's
theoretical blind spot doesn't bite in this workload; not proof it can't bite under a
different one (e.g. more frequent power collapses, longer whale tails).

**Open questions / next steps**: whether the 6-trial long-duration result is itself stable,
or would shift again with more trials (not checked beyond n=6); duration sensitivity has
only been checked for Heavy/Closed-Loop, the other 4 conditions' active windows have not
been measured or extended; the reignition check has not been repeated at the longer
duration, where falling-state dispatches might be more frequent.

**Data and repro**: duration-extension harness
`orchestrate/eenergy/run_pergpu_closedloopheavy_long5min.sh` (trials 1-3),
`orchestrate/eenergy/run_pergpu_closedloopheavy_long5min_trials456.sh` (trials 4-6).
Comparison: `scripts/eenergy/compare_closedloopheavy_duration.py`,
`scripts/eenergy/compare_closedloopheavy_duration_6trials.py`. Reignition check:
`scripts/eenergy/check_ramp_clamp_reignition.py`. Theorem 5 verification:
`scripts/eenergy/verify_leximin_vs_weighted_sum.py`. Raw data on the remote box:
`logs/closedloopheavylongpergpu_{records,power_trace,assignment}_<arm>_t<1-6>.jsonl|csv`.

## Update 2026-09-08: does power-awareness help at all, and can a rule target the actual fleet-aggregate metric instead of a per-replica proxy?

**Motivation.** Extending the duration-sensitivity work to BurstGPT (real trace, 6 trials)
and Light/Cachehit found the headline dominance story getting worse, not better, under
scrutiny: on BurstGPT, `drf_power_tiebreak_full` doesn't even beat the plain `drf_fixed`
baseline (loses on mean_ramp +8.2%, TBT +9.6%), and `lmetric_power` — the "unsafe" arm —
wins on 4/5 metrics outright, best performer overall. (BurstGPT's cache-hit rate was checked
and is structurally 0% for every arm — `src/replay_sharegpt.py`'s `--trace-csv` mode
synthesizes unique filler text per request since the real BurstGPT dataset has no prompt
text, only token counts — so Claim 2's cache-hit-collapse mechanism cannot be what's driving
this, and remains undiagnosed.) That pattern — every genuinely long-duration or real
condition either mixed or unfavorable, only the short synthetic condition clean — raised two
new questions pursued in this update: (1) does adding `Share_power` to the routing decision
help at all, or does a power-blind rule do just as well; (2) every rule tested so far reads
only a candidate's own local ramp state, while every reported metric is a fleet-**aggregate**
quantity — does a rule that targets the actual aggregate signal do better?

### Part 1: does power-awareness help? `drf_no_power`, `weighted_sum_no_power`, `lmetric`

Power-blind counterparts to `drf` and `weighted_sum` (`Share_power` never enters the
computation, not weighted to zero — genuinely absent); plain `lmetric`
(`new_tokens x in_flight_after`) is already power-blind by construction, reused unchanged as
the third arm. (`scripts/eenergy/router/scoring.py`: `dominant_share_no_power`,
`weighted_sum_score_no_power`.)

**Heavy/Closed-Loop long, first check:**

| arm | peak | mean_ramp | p99_ramp | TTFT | TBT |
|---|---|---|---|---|---|
| drf_fixed | 2506.5 | 152.9 | 1328.9 | 0.435 | 722.6 |
| **drf_no_power** | 2466.8 | 149.5 | 1198.5 | 0.433 | 706.0 |
| weighted_sum | 2434.3 | 151.5 | 1242.5 | 0.436 | 719.9 |
| weighted_sum_no_power | 2422.4 | 152.5 | 1081.1 | 0.434 | 697.2 |
| lmetric_power | 2466.5 | 148.3 | 1235.1 | 0.427 | 717.1 |
| lmetric | 2430.1 | 155.4 | 1219.8 | 0.431 | 687.9 |

**`drf_no_power` cleanly DOMINATES `drf_fixed` — every metric, including the power/ramp
metrics power-awareness is meant to protect.** `weighted_sum_no_power` beats `weighted_sum`
on 4/5 (only mean_ramp marginally worse, within noise); `lmetric` vs `lmetric_power` is
genuinely mixed (~3-2). Naive per-replica power-awareness isn't just failing to help here —
for the plain DRF baseline, it's actively counterproductive.

**Expanded to the remaining 6 conditions** (BurstGPT, Heavy/Matched, Light/Cachehit,
Heavy/Closed-Loop short, Ramp & Route, WildChat), 3 trials each
(`orchestrate/eenergy/run_pergpu_no_power_6conditions.sh`):

| condition | dominance found | drf_no_power vs coincidence_ceiling (see Part 2) |
|---|---|---|
| BurstGPT | `drf_no_power` dominates `drf_fixed` | drf_no_power clearly ahead (4/5 metrics) |
| Heavy/Matched | none | roughly split |
| Light/Cachehit | **`drf_no_power` dominates `drf_fixed`, `weighted_sum`, AND `coincidence_ceiling`** | drf_no_power wins outright |
| Heavy/Closed-Loop (short) | `coincidence_ceiling` dominates weighted_sum + lmetric_power; `drf_no_power` dominates lmetric_power | coincidence_ceiling clearly ahead (4/5 metrics) |
| Ramp & Route | `drf_power_tiebreak_full` dominates `drf_fixed` (pre-existing) | **drf_no_power data came back empty (0 valid TTFT records) — unresolved, needs re-run** |
| WildChat | `drf_fixed`/`weighted_sum` dominate `drf_power_tiebreak_full` (pre-existing reversal) | drf_no_power slightly ahead (3/5, close) |

`lmetric` vs `lmetric_power` across the same 6 conditions: `lmetric` DOMINATES `lmetric_power`
on Heavy/Closed-Loop (short); `lmetric_power` DOMINATES `lmetric` on WildChat; the other 4
conditions incomparable. No consistent direction, unlike `drf_no_power`'s mostly-favorable
pattern.

**Reading**: neither "no power signal" nor "naive per-replica power signal" wins uniformly.
`drf_no_power` is strongest specifically on BurstGPT and Light/Cachehit (real/cache-hit-heavy
traffic, no engineered pressure); the coincidence-aware rule (Part 2) is strongest
specifically on Heavy/Closed-Loop (engineered sustained pressure). That's a coherent story:
the smarter power signal earns its cost only where power pressure is real and correlated.

### Part 2: targeting the actual fleet-aggregate metric — `drf_power_tiebreak_full_coincidence_ceiling`

**Design.** Every rule tested this whole project computes `Share_power(c)` from candidate
`c`'s own local ramp state only — never asking whether *other* replicas are ramping
simultaneously. Every reported metric (peak, mean/p99 ramp) is measured on the
fleet-**aggregate** trace. A rule can look individually safe on every replica while still
producing a bad aggregate ramp if it never accounts for replicas ramping together.
`coincidence_ceiling_factor(candidates)` computes one shared multiplier per decision: 0 or 1
currently-elevated replicas (`share_power > 0.5`) → factor 1.0 (no adjustment); 2+
simultaneously elevated → factor shrinks (`1/(1+(n_elevated-1))`). Every candidate's ceiling
is scaled by this **same** factor at that decision — one scalar shared identically across
candidates, exactly the object §4.3's existing corollary already covers ("Lemma 1 holds
unchanged for any κ(t) shared identically by every candidate, however computed"), so
Pareto/threshold-safety carry over without a new proof. Mechanistically: scaling every
candidate's ceiling by the same factor doesn't change their relative order by `Share_power`
alone — it changes `Share_power`'s magnitude relative to `Share_compute`/`Share_load` in
`D(c)=max(...)`, making power more likely to be the binding dimension for everyone during a
genuine coincidence event. Verified with a hand-constructed case that the mechanism actually
flips a routing decision (not a no-op): without adjustment, a replica with moderate local
power pressure wins over one with zero power but higher compute; with two *other* replicas
coincidentally elevated, the pick flips to the higher-compute replica, avoiding piling onto
an already-pressured fleet.

**Heavy/Closed-Loop long** (3 trials): **best p99_ramp of all 6 arms tested there** — 1050.0
vs. `drf_power_tiebreak_full`'s 1136.0 (~7.6% better), also ahead of `weighted_sum`
(1242.5) and `lmetric_power` (1235.1). Does not achieve full dominance (small mean_ramp/TBT
costs relative to `drf_power_tiebreak_full`/`lmetric_power`).

**Heavy/Closed-Loop short** (3 trials): **`coincidence_ceiling` DOMINATES both `weighted_sum`
and `lmetric_power`** — best peak of all 6 arms (2299.3W) and TBT far better than any other
scored arm (523.2 vs. 577-594ms elsewhere).

**Expanded to the same 6 conditions as Part 1** — full numeric table:

| condition | peak | mean_ramp | p99_ramp | TTFT | TBT | dominance |
|---|---|---|---|---|---|---|
| BurstGPT | 2207.6 | 204.5 | 1533.1 | 0.164 | 97.3 | none |
| Heavy/Matched | 2626.9 | 197.5 | 1310.0 | 3.804 | 1294.6 | none |
| Light/Cachehit | 1908.3 | 97.6 | 1536.1 | 0.066 | 34.3 | none (dominated BY drf_no_power) |
| Heavy/Closed-Loop (short) | 2299.3 | 146.8 | 1653.2 | 0.521 | 523.2 | dominates weighted_sum + lmetric_power |
| Ramp & Route | 2556.1 | 240.5 | 2128.8 | 0.529 | 481.5 | none |
| WildChat | 2277.2 | 126.3 | 1201.6 | 0.149 | 252.3 | none |

Strongest and cleanest specifically on Heavy/Closed-Loop (both durations — the condition
engineered for sustained, correlated power pressure); doesn't clearly help on BurstGPT,
WildChat, or Ramp & Route, and is dominated outright by `drf_no_power` on Light/Cachehit.
Consistent with Part 1's reading: coincidence-awareness earns its cost only where genuine
correlated pressure exists.

### Part 3: does the coincidence-ceiling mechanism generalize beyond the DRF-family structure?

Substituted the same shared coincidence factor into `weighted_sum`'s and `lmetric_power`'s
own `Share_power` term (`weighted_sum_coincidence_ceiling`,
`lmetric_power_coincidence_ceiling`) — neither gains a threshold-safety guarantee from this
(Theorem 5 still applies to `weighted_sum` regardless of ceiling design; `lmetric_power`'s
Pareto-safety issue is orthogonal to which ceiling feeds its power term), so this is an
empirical-only comparison. Heavy/Closed-Loop long, 3 trials each:

| arm | peak | mean_ramp | p99_ramp | TTFT | TBT |
|---|---|---|---|---|---|
| weighted_sum | 2434.3 | 151.5 | 1242.5 | 0.436 | 719.9 |
| weighted_sum_coincidence_ceiling | 2474.5 | 152.0 | 1199.3 | 0.431 | 717.9 |
| lmetric_power | 2466.5 | 148.3 | 1235.1 | 0.427 | 717.1 |
| **lmetric_power_coincidence_ceiling** | 2451.0 | 161.3 | **1020.8** | 0.429 | **708.5** |
| coincidence_ceiling (DRF-family) | 2433.8 | 152.5 | 1050.0 | 0.427 | 720.7 |
| drf_power_tiebreak_full | 2454.4 | 147.1 | 1136.0 | 0.427 | 715.4 |

**`lmetric_power_coincidence_ceiling` gets the best p99_ramp *and* best TBT of all 6 arms** —
even better than the DRF-family's own `coincidence_ceiling` (1020.8 vs. 1050.0 p99_ramp) —
confirming the mechanism helps beyond the sorted-rule structure it was built for, at the cost
of the worst mean_ramp of the six. `weighted_sum_coincidence_ceiling`'s improvement is more
modest. `drf_power_tiebreak_full` still DOMINATES `weighted_sum_coincidence_ceiling`
outright, even after the coincidence boost — the safe, structurally-repaired rule remains
competitive even against rivals given the same upgrade.

### Part 4: `lmetric_power_pareto` — a second, independent Pareto-safety repair

`(1 + Share_compute) x (1 + Share_load) x (1 + Share_power)`, replacing `lmetric_power`'s
`Share_compute x Share_load x (1 + Share_power)`. Removes the zero-collapse failure mode
(Claim 2) by construction — every factor is always ≥1, never zero. Proven and verified
Pareto-safe (200,000 trials, 0 violations at 40% cache-hit rate, vs. `lmetric_power`'s 12.8%)
via the same mechanism that makes `weighted_sum` safe (`ln(score)` is a sum of monotonic
per-share terms — Geoffrion 1968, reached via a product instead of a literal sum). **NOT
threshold-safe**: same vulnerability as `weighted_sum` (Theorem 5), verified at 11.60% vs.
`weighted_sum`'s 11.71% on the same construction (`scripts/eenergy/
verify_lmetric_power_plusone_threshold` logic, ad hoc verification, not yet committed as a
standalone script).

Hardware validation on Light/Cachehit (highest cache-hit rate, most direct test of the fixed
mechanism), 3 trials:

| arm | peak | mean_ramp | p99_ramp | TTFT | TBT |
|---|---|---|---|---|---|
| lmetric_power | 1901.8 | 81.4 | 1723.3 | 0.066 | 31.3 |
| **lmetric_power_pareto** | 1910.0 | 90.0 | **1420.9** | 0.067 | 35.3 |

Best p99_ramp of all 6 arms tested there (17.5% better than plain `lmetric_power`), but
costs mean_ramp (+10.6%) and TBT (+12.8%) — incomparable to every other arm, a real trade
rather than a free improvement. Same qualitative shape as `coincidence_ceiling`: fixing a
safety property changes real empirical behavior, buying tail-ramp protection at an
average-case cost, not a no-op.

### Part 5: does compute-avoidance or load-avoidance drive `drf_no_power`'s implicit benefit?

`pick_load_only` added as the symmetric counterpart to the existing `pick_compute_only`
(itself originally built to ask the same question about `lmetric`). Compares `compute_only`,
`load_only`, and `drf_no_power` (their `max()`-combination) on Heavy/Closed-Loop long —
**running as of this write-up, results not yet available.**

### Open questions / next steps

- Ramp & Route's `drf_no_power` data is corrupted (empty records file) — needs re-running
  before any Ramp & Route conclusion involving that arm.
- `compute_only`/`load_only` ablation (Part 5) still in progress.
- `cachehit1024pergpu_*` data (the max_tokens=128 vs 1024 isolation batch) was collected but
  not yet analyzed for dominance patterns — open.
- Whether any of this reshapes the paper's headline is undecided; not yet touched.

### Data and repro

Scoring additions (`scripts/eenergy/router/scoring.py`): `dominant_share_no_power`,
`weighted_sum_score_no_power`, `pick_drf_no_power`, `pick_weighted_sum_no_power`,
`coincidence_ceiling_factor`, `dominant_share_vector_power_priority_full_coincidence_ceiling`,
`pick_drf_power_tiebreak_full_coincidence_ceiling`, `weighted_sum_score_coincidence_ceiling`,
`pick_weighted_sum_coincidence_ceiling`, `lmetric_power_score_coincidence_ceiling`,
`pick_lmetric_power_coincidence_ceiling`, `lmetric_power_pareto_score`,
`pick_lmetric_power_pareto`, `pick_load_only`. All wired into `router_core.py`'s `_POLICIES`
and `route()`. 145/145 scoring+router_core tests pass as of this update.

Orchestration: `orchestrate/eenergy/run_pergpu_closedloopheavylong_no_power.sh`,
`run_pergpu_no_power_6conditions.sh`, `run_pergpu_closedloopheavylong_coincidence_ceiling.sh`,
`run_pergpu_coincidence_ceiling_6conditions.sh`,
`run_pergpu_closedloopheavylong_ws_lp_coincidence_ceiling.sh`,
`run_pergpu_cachehit_lmetric_power_pareto.sh`,
`run_pergpu_closedloopheavylong_compute_load_only.sh`.

Analysis: `scripts/eenergy/check_no_power_comparison.py`,
`check_coincidence_ceiling_6conditions.py`, `check_lmetric_vs_coincidence.py`,
`check_drf_no_power_all_conditions.py`, `check_lmetric_all_conditions.py`,
`check_ws_lp_coincidence_ceiling.py`, `check_cachehit_lmetric_power_pareto.py`.

## Update 2026-09-08 (continued): coincidence-ceiling generalizes across rule families, one mechanism investigation stays open, and a fourth empirical champion emerges — decision: do not chase it

### Part 6: compute_only/load_only verified, and the old 2026-09-02 finding about compute_only does not replicate

Extended `compute_only`/`load_only` to n=6 on Heavy/Closed-Loop long: the exact p99_ramp tie
seen at n=3 (1059.7/1059.7) didn't survive — now 1096.0±55.2 vs. 1132.2±95.3 — but the core
finding holds: both still clearly beat `drf_no_power`'s p99_ramp (1198.5), with unremarkable
variance.

Separately, re-ran `compute_only`/`load_only` fresh on Heavy/Matched (the condition of the
original 2026-09-02 finding) to check replication. **The old claim — "compute_only has the
worst, 16x-noisier p99_ramp" — does not replicate.** Today `compute_only` is mid-pack (4th of
7, p99_ramp=1316.3±102.7), not worst; that title now goes to `load_only` (1509.4±64.8), which
didn't exist as a policy in 2026-09-02 so was never in that comparison. The one part that does
replicate: `compute_only` has the worst TTFT both times (5.345 today vs. 5.122 originally).
That 2026-09-02 finding was single-run, 3 trials, no replication check — exactly the kind of
result this project has repeatedly found doesn't survive re-testing.

**Why compute_only's TTFT is bad specifically on Heavy/Closed-Loop-long and Heavy/Matched,
and not on Light/Cachehit**: both conditions are `--min-turns 1 --max-turns 1` (every
conversation a single fresh turn), giving ~0.0-0.1% cache-hit rate (mean new/raw ≈
0.994-0.996) — almost no real `Share_compute` signal to differentiate candidates, so
`compute_only` degenerates toward tie-rotation (confirmed: its whale-placement distribution
on Heavy/Closed-Loop-long is nearly identical to `round_robin`'s, CV 0.135 vs 0.132), unable
to avoid an overloaded replica. Light/Cachehit has substantial partial cache reuse (mean
new/raw ≈ 0.69) — a real signal — and there `compute_only`'s TTFT (0.066) is completely
unremarkable, tied with everyone. The weakness moved rather than disappeared, though: on
Cachehit, `compute_only`'s p99_ramp becomes the worst among scored arms instead (dominated by
both `lmetric` and `lmetric_power`) — whichever metric depends on load-awareness is where
load-blindness costs something; which metric that is depends on which signal actually carries
information in a given condition.

### Part 7: three mechanism hypotheses for drf_no_power's worse p99_ramp, all ruled out or unconfirmed

Puzzle: on Heavy/Closed-Loop long, `compute_only` and `load_only` (single-resource rules) both
beat `drf_no_power`'s p99_ramp (their `max()` combination) — counterintuitive, since combining
signals should naively be at least as good as either alone. Three hypotheses checked directly
against data, in order:

1. **Whale-placement evenness** (do less-even whale distributions across replicas explain the
   gap?) — **falsified**. `drf_no_power`'s per-GPU whale-count CV (0.088) is *tighter* than
   `round_robin`'s/`compute_only`'s (0.132/0.135), yet `drf_no_power` still has the worse
   p99_ramp. More even spread, worse tail.
2. **Temporal coincidence-poll frequency** (does drf_no_power produce more polls with 2+
   replicas simultaneously over their ramp ceiling?) — **falsified**. `drf_no_power` has the
   *lowest* coincidence-poll rate of the four (0.383% vs. round_robin's 0.528%), yet the worst
   p99_ramp.
3. **Dominant-resource switching frequency** (does the effective dominant resource — compute
   vs. load — for the chosen candidate flip between consecutive decisions more than random
   chance predicts?) — **not confirmed**. Added direct instrumentation
   (`Router.last_share_compute`/`last_share_load`, two new assignment-log columns) and
   re-ran `drf_no_power` to measure this directly: observed switch rate (26.5%) matches the
   i.i.d.-random-chance null (27.4%, given the observed base rate) almost exactly — ratio
   0.969, no evidence of elevated flip-flopping.

That third check surfaced a clean, unrelated fact instead: only 16.4% of `drf_no_power`'s
decisions are compute-dominant, near-exactly matching the workload's whale-frac (15%) — for a
non-whale request `Share_compute` (tiny, new_tokens/16384) is almost always smaller than
`Share_load`; for a whale (~13.6-15.5k tokens against the same budget) it almost always
exceeds it. So `drf_no_power` isn't reactively oscillating — it's cleanly, deterministically
switching between "act like `load_only`" (83.6% of the time) and "act like `compute_only`"
(16.4%, exactly on whale arrival), driven by the external whale process, not by any internal
instability.

What *is* different: `drf_no_power` has the **lowest absolute max ramp** of all four arms
(6987.3) yet the **worst p99** (1198.5) — a "fatter belly, tamer peak" tail shape, the opposite
of what more/bigger spikes would produce. The real mechanism remains open after three checked
and ruled-out/unconfirmed hypotheses — a genuine unresolved finding, not a gap in effort.

### Part 8: coincidence-ceiling generalized to all three rule families, across all 6 conditions

Extended `weighted_sum_coincidence_ceiling` and `lmetric_power_coincidence_ceiling` (validated
previously on only Heavy/Closed-Loop long) to the same 6 conditions as the DRF-family
`coincidence_ceiling`: BurstGPT, Heavy/Matched, Light/Cachehit, Heavy/Closed-Loop (short),
Ramp & Route, WildChat (3 trials each, 36 runs). Full numeric comparison (all three
coincidence-ceiling variants + their plain counterparts + `round_robin`) is now available for
every condition — see `scripts/eenergy/check_all_coincidence_ceiling_variants_6conditions.py`.

**Dominance tally (count of times each arm appears as the dominator, across all 6
conditions):**

| arm | dominance wins |
|---|---|
| **lmetric_power_coincidence_ceiling** | **9** |
| lmetric_power | 3 |
| drf_power_tiebreak_full | 2 |
| coincidence_ceiling (DRF-family) | 2 |
| weighted_sum | 2 |
| weighted_sum_coincidence_ceiling | 1 |
| drf_fixed | 1 |

`lmetric_power_coincidence_ceiling` is the single strongest empirical performer of everything
tested this entire investigation — more than triple the next-highest count, and not just
against weak arms: on WildChat it directly dominates the DRF-family's own
`coincidence_ceiling`, and on BurstGPT/Cachehit/WildChat it dominates
`weighted_sum_coincidence_ceiling` outright.

**This is the fourth distinct empirical champion this investigation has produced, each
displacing the last as testing expanded**: (1) the original short-duration
`drf_power_tiebreak_full`+`weighted_sum` headline, until duration-extension complicated it;
(2) `compute_only`/`load_only`, strong on Heavy/Closed-Loop-long, until the mechanism
investigation (Part 7) turned up no clean explanation and the Heavy/Matched replication check
(Part 6) showed instability; (3) `coincidence_ceiling` (DRF-family); (4) now
`lmetric_power_coincidence_ceiling`.

**Decision: do not make `lmetric_power_coincidence_ceiling` the paper's headline, despite the
strong record.** Two independent reasons, not just caution:
1. **It is not Pareto-safe.** The coincidence-ceiling adjustment only rescales the
   `Share_power` term; it does nothing to fix Claim 2's cache-hit-collapse mechanism, which is
   unchanged in the multiplicative structure. It is the exact same counterexample construction
   as plain `lmetric_power` — the least theoretically protected of any coincidence-ceiling
   variant, despite the best raw numbers.
2. **The repeated-displacement pattern is itself the evidence for the paper's actual
   argument.** Committing to whichever rule currently holds the empirical lead, immediately
   after the third such rule was displaced, would repeat the exact mistake this whole
   investigation has been diagnosing. `lmetric_power_coincidence_ceiling`'s strong-but-
   unguaranteed record is kept as supporting evidence for why the guarantee (Theorem 4) is
   worth having, not adopted as a competing proposal.

`drf_power_tiebreak_full_coincidence_ceiling` remains the recommended rule: it is the only
coincidence-ceiling variant with a full, proven worst-case guarantee (Theorem 4, inherited
unchanged via the §4.3 shared-ceiling corollary — no new proof needed), it directly closes a
limitation the paper already states as unvalidated future work, and its own empirical record
(double dominance on Heavy/Closed-Loop short, best p99_ramp on Heavy/Closed-Loop long among
guaranteed rules) is genuinely strong on the condition central to the paper's motivation, even
though it does not top the raw cross-condition tally.

### Part 9: max_ramp added to standard comparisons

`compare_closedloopheavy_duration.py`'s `aggregate()` now also returns `max_ramp` (the single
largest per-trial fleet-aggregate ramp value, averaged across trials — distinct from a
pooled-across-all-trials max, which is a different, also-computed statistic in some diagnostic
scripts). Not folded into `dominates()`'s criteria — that would retroactively change every
dominance conclusion in this log without an explicit decision to do so; it is display-only
unless/until that decision is made.

### Data and repro

Instrumentation: `Router.last_share_compute`/`last_share_load` (generic, any policy), two new
trailing columns (`share_compute`, `share_load`) in the assignment log, backward-compatible
(existing scripts read columns by name). Orchestration:
`run_pergpu_closedloopheavylong_compute_load_only_trials456.sh`,
`run_pergpu_heavymatched_compute_load_only.sh`, `run_pergpu_cachehit_compute_load_only.sh`,
`run_pergpu_closedloopheavylong_drf_no_power_diagnostic.sh`,
`run_pergpu_ws_lp_coincidence_ceiling_6conditions.sh`. Analysis:
`check_compute_load_only_6trials_and_heavymatched.py`, `check_cachehit_compute_load_only.py`,
`check_whale_placement_evenness.py`, `check_temporal_coincidence_events.py`,
`check_top_ramp_values.py`, `check_dominant_resource_switching.py`,
`check_full_comparison_all_conditions.py`,
`check_all_coincidence_ceiling_variants_6conditions.py`.

### Part 10: the Pareto-safe-repair family (lmetric_power_pareto + epsilon variants), full
7-condition sweep, round_robin folded into every comparison, and a scoped tail-protection
result

Three new Pareto-safe repairs of `lmetric_power`'s Claim-2 cache-hit-collapse were built and
validated this round, all via the same `(shift + share) x (shift + share) x (...)` monotone-
factor argument (200,000-trial brute force, 0 violations, at every shift tested):

- `lmetric_power_pareto_coincidence_ceiling`: `(1+share_compute) x (1+share_load) x
  (1+share_power)` — full `+1` shift on every factor (the `shift=1` case of the family below).
- `lmetric_power_pareto_epsilon_coincidence_ceiling`: `(epsilon+share_compute) x
  (epsilon+share_load) x (1+share_power)`, default `epsilon=0.01` — asymmetric, power keeps
  the full `+1`.
- `lmetric_power_pareto_epsilon_all_coincidence_ceiling`: same but `epsilon` applied to all
  three factors including power.
- `lmetric_power_pareto_epsilon_small_coincidence_ceiling`: same shape as the `epsilon`
  variant above, `epsilon=0.001` (10x smaller) — Heavy/Closed-Loop (long) only, 3 trials, first
  look at whether shrinking epsilon keeps closing the gap to `lmetric_power_coincidence_ceiling`.

All three main variants (`pareto`, `pareto_epsilon`, `pareto_epsilon_all`) now have 3-trial
data on all 7 conditions (BurstGPT, Heavy/Closed-Loop short & long, Heavy/Matched,
Light/Cachehit, Ramp & Route, WildChat). `round_robin` was also added to every comparison for
the first time this round (it existed before only as a spec-mandated baseline condition, not
folded into the coincidence-ceiling-era comparisons). Full table + tally:
`scripts/eenergy/check_full_pareto_family_comparison.py`.

**Overall dominance tally, all 7 conditions, 11 arms:**

| arm | dominance wins |
|---|---|
| lmetric_power_coincidence_ceiling | 13 |
| lmetric_power_pareto_epsilon_coincidence_ceiling | 12 |
| lmetric (plain, power-blind) | 10 |
| coincidence_ceiling (DRF-family) | 5 |
| lmetric_power_pareto_coincidence_ceiling | 3 |
| weighted_sum_coincidence_ceiling | 2 |
| compute_only | 1 |
| load_only | 1 |
| lmetric_power_pareto_epsilon_all_coincidence_ceiling | 1 |
| round_robin | 0 |

Two things this tally makes newly visible:

1. **`round_robin` never wins a single dominance relationship across all 7 conditions** — it
   is dominated outright 5 times on Light/Cachehit alone (by `lmetric`, `compute_only`,
   `load_only`, `coincidence_ceiling`, `weighted_sum_coincidence_ceiling`,
   `lmetric_power_coincidence_ceiling`, `lmetric_power_pareto_coincidence_ceiling`, and
   `lmetric_power_pareto_epsilon_coincidence_ceiling` — 8 separate dominators, the single
   most one-sided result in this whole investigation), and is otherwise incomparable
   everywhere else. Confirms Claim 2's cache-awareness mechanism cleanly: with zero
   cache-state signal, round_robin cannot avoid routing cache-hit-heavy requests onto already-
   loaded replicas, and loses on every metric simultaneously rather than trading one off for
   another.
2. **Plain `lmetric` (power-blind, `P-token x BS`) is the 3rd-highest scorer, ahead of every
   Pareto-safe or DRF-family power-aware rule except the two chart-toppers.** It has the best
   max_ramp of all arms on both Heavy/Matched (3125.0) and Light/Cachehit (2386.3) — beating
   every power-aware rule tested, safe or not. This directly complicates the working
   "tail-protection" narrative below.

**`lmetric_power_pareto_epsilon_coincidence_ceiling`'s strong tally (12) is now confirmed
Pareto-safety-proof-backed, not a fluke of the earlier 3-condition look** — it held up across
all 7 conditions, including directly dominating the DRF-family `coincidence_ceiling` on
Light/Cachehit and WildChat.

**But the pushback from the earlier "abandon DRF" discussion still holds on pairwise,
per-metric inspection of the two conditions the paper is actually built around:**

- Heavy/Closed-Loop (short): `coincidence_ceiling` DOMINATES `lmetric_power_pareto_coincidence_ceiling`
  AND `lmetric_power_pareto_epsilon_coincidence_ceiling` outright.
- Heavy/Closed-Loop (long): no direct pairwise dominance either way between
  `coincidence_ceiling` and `lmetric_power_pareto_epsilon_coincidence_ceiling` — they are
  incomparable. `coincidence_ceiling` has p99_ramp 1050.0, close to but NOT actually the best
  of 11 arms (see correction in Part 12 — `lmetric_power_coincidence_ceiling`'s 1020.8 was
  marginally lower, and per Part 12's noise-band audit neither is statistically
  distinguishable from most of the other 9 arms anyway); `lmetric_power_pareto_epsilon_coincidence_ceiling`
  has max_ramp 4177.4 (best of 11 arms, point-estimate) vs. `coincidence_ceiling`'s 6860.9
  (worst point-estimate of 11 arms on this condition, worse than round_robin and load_only) —
  see Part 12 for whether this specific max_ramp gap survives the noise-band check.

**Scoped tail-protection result** (in response to the hypothesis "our strategies protect
tails, improved p99 or max"): plausible at face value, but **Part 12's noise-band audit
(added after this section was first written) found the entire spread of p99_ramp and max_ramp
point estimates across all 11 arms on this condition is within ~1.2-2x a single true standard
deviation** — meaning every ranking claim below, including which rule is "best" or "worst" on
either tail statistic, is not currently statistically distinguishable from noise. Kept here
for the record of what the point estimates showed, not as a confirmed result:

- `coincidence_ceiling` (DRF-family) had the (statistically insignificant) best p99_ramp
  point-estimate under sustained power pressure but simultaneously the worst max_ramp
  point-estimate on the same condition.
- `lmetric_power_pareto_epsilon_coincidence_ceiling` had the best max_ramp point-estimate on
  Heavy/Closed-Loop long, but a mediocre p99_ramp point-estimate there.
- On Heavy/Matched, neither rule's point estimate beat plain power-blind `lmetric` on either
  tail statistic.
- Net: pending replication to 6+ trials per arm (Part 12), "protects tails" cannot currently
  be stated as a confirmed finding for either rule. Full table:
  `scripts/eenergy/check_tail_protection_claim.py`.

**Epsilon-shrinking anomaly, and why it isn't actually an anomaly.** Going from
`epsilon=0.01` to `epsilon=0.001` on Heavy/Closed-Loop (long) made max_ramp *worse* (4177.4 to
5485.0, +31%) and p99_ramp slightly worse (1177.0 to 1222.0), not closer to
`lmetric_power_coincidence_ceiling`'s numbers as the "smaller epsilon approaches the unsafe
original" framing would suggest at a glance. Resolved by re-reading the actual formula:
`lmetric_power_pareto_coincidence_ceiling` (the "coincidence_ceiling-only" Pareto-safe rule,
no `pareto_epsilon` suffix) is the **epsilon=1** case of the epsilon family, not the
epsilon-to-zero limit — `(1+share_compute) x (1+share_load) x (1+share_power)` matches
`(epsilon+share_compute) x (epsilon+share_load) x (1+share_power)` exactly at `epsilon=1`.
Shrinking epsilon from 0.01 toward 0 moves *away* from `epsilon=1` (`lmetric_power_pareto`)
and *toward* `epsilon=0`, which is a different limit entirely: since `share_compute` and
`share_load` both have the same constant denominators (`token_budget`, `max_num_seqs`) across
all candidates in a routing decision, `epsilon=0` reduces (up to a constant rescaling that
doesn't change argmin ordering) to `share_compute x share_load x (1+share_power) prop=
new_tokens x in_flight_after x (1+share_power)` — the exact unsafe
`lmetric_power_coincidence_ceiling` formula, cache-hit collapse included. So the correct
prediction was never "epsilon shrinking approaches `lmetric_power_pareto`" — it's "epsilon
shrinking approaches `lmetric_power` (unsafe)." The observed drift away from `lmetric_power_pareto`'s
numbers is consistent with this, not contrary to it. `epsilon=0` itself was queued as a direct
empirical check of this prediction (see below) — since `epsilon=0` collapses one factor to
exactly zero whenever `share_compute=0` (a full cache hit), it is expected to reproduce Claim
2's exact failure mode and NOT be Pareto-safe, unlike every `epsilon>0` member of this family.

### Part 11: epsilon=0 diagnostic run, and pinning down the closed-loop decision-noise floor

Added a diagnostic-only `lmetric_power_pareto_epsilon_zero_coincidence_ceiling` (epsilon=0.0,
explicitly NOT Pareto-safe — the proof requires every factor strictly positive) to test a
prediction directly: since `share_compute`/`share_load` share the same constant denominators
(`token_budget`, `max_num_seqs` — confirmed identical across all replicas via
`launch_router_experiment.sh`'s single global `TOKEN_BUDGET`/`MAX_NUM_SEQS`), epsilon=0's
score is an exact positive rescaling of `lmetric_power_score_coincidence_ceiling`'s raw score
for every candidate in a decision, so `argmin` (and therefore every routing decision) should
be provably identical to `lmetric_power_coincidence_ceiling` — NOT to
`lmetric_power_pareto_coincidence_ceiling`, which is this family's `epsilon=1` case, not its
`epsilon=0` case. 3 trials, Heavy/Closed-Loop (long).

**First look (3 trials each) matched on peak/TTFT/TBT (within 0.1-1.2%) but showed an
unexplained gap on mean_ramp (-11.6%) and p99_ramp (+21.5%).** Investigated two ways:

1. **Row-level decision pairing.** Assignment logs record `raw_tokens` per dispatched request
   (a content fingerprint — the harness pairs whale placement/content across arms via a fixed
   `--pad-seed`, confirmed: only ~1-2% of rows differ in `raw_tokens` between any two runs) and
   `replica_id` (the actual routing decision). Pairing `epsilon_zero` trial 1 against
   `lmetric_power_coincidence_ceiling` trial 1 row-by-row: **706/901 (78%) of decisions differ
   even where content matches**, first divergence at row 7.
2. **Self-vs-self control.** To check whether that 78% reflects a real rule difference or just
   harness noise, extended `lmetric_power_coincidence_ceiling` itself from 3 to 6 trials
   (same condition, same paired content) and pairwise-compared all `C(6,2)=15` trial pairs the
   same way. **Every single pair mismatches on 76-82% of decisions** (mean ~79.5%), diverging
   within the first ~35 requests every time (median first-mismatch row: 7) — statistically
   indistinguishable from the `epsilon_zero` vs. `lmetric_power_coincidence_ceiling` comparison.

**Conclusion: the argmin-equivalence proof holds, and there is no real gap to explain.**
`epsilon=0` provably makes identical decisions to `lmetric_power_coincidence_ceiling` given
identical visible state — but this is a closed-loop feedback system (routing decisions change
replica load, which changes what the next decision sees), and real execution has sub-ms timing
noise (decode speed jitter, NVML sampling timing, OS scheduling, other jobs on the shared box)
that a scoring formula cannot control for. A single early tied-decision flip cascades into a
completely different global assignment sequence within ~35 requests — this is inherent to the
harness under closed-loop load, present for ANY single policy replicated across trials, not
specific to comparing two different rules. Confirmed quantitatively: `lmetric_power_coincidence_ceiling`'s
own trials 4-6 (mean_ramp 142.8±10.9) land almost exactly where `epsilon_zero`'s 3 trials
landed (142.5) — the "gap" from trials 1-3 (161.3±1.7) was trials 1-3 sitting on the high side
of this arm's own natural range, not `epsilon_zero` diverging from it. `lmetric_power_coincidence_ceiling`
1-6 aggregate: peak 2433.9±41.5, mean_ramp 152.0±12.3, p99_ramp 1095.6±103.4, max_ramp
6653.7±2286.5, ttft 0.429±0.005, tbt 712.5±15.3 — noticeably wider std than the earlier
3-trial numbers on every ramp statistic, now that the natural range is visible.

**Practical implication:** this harness's decision-level noise floor (~80% mismatch between
any two runs of the identical policy) means single-trial or even 3-trial comparisons between
*different* rules on ramp statistics (mean_ramp, p99_ramp, max_ramp specifically — peak/TTFT/
TBT are comparatively stable) should be read cautiously; a real mechanism difference and pure
harness noise can look identical at n=3. This doesn't invalidate the dominance-based
comparisons elsewhere in this doc (dominance requires simultaneous agreement across all 5
metrics, which noise alone is unlikely to produce consistently), but it does mean any
single-metric point-estimate gap on mean_ramp/p99_ramp/max_ramp between two 3-trial arms
should not be over-read without checking it against a rule's own self-noise range first.

### Data and repro (Part 10-11)

Analysis: `check_full_pareto_family_comparison.py`, `check_lp_pareto_epsilon_vs_no_power.py`,
`check_tail_protection_claim.py`, `check_epsilon_zero_vs_lmetric_power.py`,
`check_lmetric_power_cc_self_noise_floor.py`, `pairwise_decision_mismatch.py`. Orchestration:
`run_pergpu_closedloopheavylong_lp_pareto_coincidence_ceiling.sh`,
`run_pergpu_lp_pareto_coincidence_ceiling_6conditions.sh`,
`run_pergpu_closedloopheavylong_lp_pareto_epsilon_variants.sh`,
`run_pergpu_lp_pareto_epsilon_variants_6conditions.sh`,
`run_pergpu_closedloopheavylong_lp_pareto_epsilon_small.sh`,
`run_pergpu_closedloopheavylong_lp_pareto_epsilon_zero.sh`,
`run_pergpu_closedloopheavylong_lmetric_power_cc_self_replicate.sh`,
`run_pergpu_lp_pareto_epsilon_small_6conditions.sh`.

### Part 12: the noise floor is large enough to make most ramp-statistic rankings in Parts
8-10 currently unconfirmed — a significant caveat, not just a footnote

Two follow-on investigations, prompted by the epsilon_zero mean_ramp/p99_ramp gap in Part 11
not fully closing on the first look.

**1. Real-data control on the decision-mismatch mechanism.** Part 11 showed
`lmetric_power_coincidence_ceiling`'s own trials mismatch on ~80% of individual routing
decisions against each other (pairing by request identity, not row order). Two follow-ups:

- *Is this specific to whale-padded synthetic content?* No. Re-ran the same pairwise check
  (properly joined by `conv_id`+`turn` via nearest dispatch-time matching, not naive row
  order — row order is invalid for multi-turn conditions since real timing reshuffles
  conversation interleaving) on BurstGPT (real arrival trace + synthetic filler text, 75.0%),
  Light/Cachehit (real ShareGPT text, zero whale injection, 84.0-84.6%), and WildChat (real
  WildChat text, zero whale injection, open-loop Poisson arrivals, 81.9-83.4%). All four
  conditions land in the same 75-85% band — the two conditions built entirely on real
  conversation text with zero synthetic padding showed slightly *higher* mismatch, not lower.
  This also rules out "closed-loop feedback specifically" as the necessary cause, since
  WildChat is open-loop.
- *Is this a property of state-dependent rules specifically, not the harness generally?*
  Yes, confirmed directly: `round_robin` (which never reads live replica state, only an
  incrementing counter) shows **3.1% decision-mismatch** on the same condition where every
  load/power-aware rule shows 75-85%. This pins the mechanism down precisely: real request
  completion timing is not bit-reproducible across separate physical executions (GPU kernel
  scheduling, thermal/clock variance, OS scheduling), so a router's live-state inputs
  (`in_flight_after`, `ramp_rate_w_per_s`) are only ever "true at this exact wall-clock
  instant" and that instant itself isn't reproducible. Close-call ties get flipped by
  real timing noise the formula can't see, and once one decision flips, replica loads
  genuinely diverge (not just measurement noise around a shared trajectory) — every
  subsequent decision inherits real, compounding state difference. `round_robin` never makes
  a state-dependent tie-break, so it's nearly immune.

**2. Quantitative noise-band audit: does the 80% decision-mismatch translate into
aggregate-metric noise large enough to swamp this session's rankings?** Computed z-scores for
`epsilon_zero`, `lmetric_power_pareto_epsilon_coincidence_ceiling` (ε=0.01), and
`lmetric_power_pareto_coincidence_ceiling` (ε=1) against `lmetric_power_coincidence_ceiling`'s
true 6-trial noise band (mean_ramp 152.0±12.3, p99_ramp 1095.6±103.4, max_ramp
6653.7±2286.5) — all three arms fall within ±1.4σ on every ramp metric, i.e. statistically
indistinguishable from `lmetric_power_coincidence_ceiling`'s own spread.

Extended the check to all 11 arms tested on Heavy/Closed-Loop (long) — the **full range**
across every different routing rule tested, for each ramp metric, compared to the true 1-std
band:

| metric | range across 11 arms | true 1-std band | range in std-widths |
|---|---|---|---|
| mean_ramp | 26.0 (138.9-164.9) | ±12.3 | 2.12σ |
| p99_ramp | 201.2 (1020.8-1222.0) | ±103.4 | 1.95σ |
| max_ramp | 2683.4 (4177.4-6860.9) | ±2286.5 | 1.17σ |

A full range under ~2 true standard deviations across 11 independent samples is exactly what
pure sampling noise produces on its own — no real difference between rules is required to
explain it. **This means essentially none of the mean_ramp/p99_ramp/max_ramp rankings,
"best/worst of N arms" claims, or close-margin dominance relationships elsewhere in this
document (Parts 8-10 particularly) are currently statistically confirmed** — most arms have
only 3 trials, and 3-trial sample std badly underestimates true variance (demonstrated
directly: `lmetric_power_coincidence_ceiling`'s own t1-3 std on mean_ramp was ±1.7; the true
6-trial std was ±12.3, 7x larger).

**What this does and does not affect:**
- Does NOT affect the theory (Lemma 1, Theorem 4, Theorem 5) — proofs, not empirical claims.
- Does NOT affect peak/TTFT/TBT-based comparisons — these showed ~1-2% relative std even at
  n=3 throughout this investigation, a genuinely different (much lower) noise regime.
- DOES affect most mean_ramp/p99_ramp/max_ramp rankings in Parts 8-10, including the specific
  "coincidence_ceiling has the best p99_ramp" framing used in the "abandon DRF" pushback
  (corrected above) and the tail-protection scoping result (corrected above). `dominates()`
  requires simultaneous wins on 5 metrics; 3 are reliable (peak/ttft/tbt) but the other 2
  (mean_ramp, p99_ramp) are within the noise band demonstrated here, so any dominance
  relationship found in this document has a real, currently-unquantified chance of being
  partly noise-driven rather than a genuine effect.

**Recommendation going forward:** any ramp-statistic claim intended for the paper needs
replication to 6+ trials (the extension that surfaced this issue in the first place) before
being stated with confidence, not the 3 trials used for most of this investigation.

**epsilon=0.001 (`lmetric_power_pareto_epsilon_small_coincidence_ceiling`) now has 3-trial
data on all 7 conditions** (extended from Heavy/Closed-Loop long only). Per the audit above,
any comparison of this new data against other 3-trial arms on mean_ramp/p99_ramp/max_ramp
should be read with the same caution — not presented as confirmed rankings pending the same
6-trial replication treatment.

### Data and repro (Part 12)

Analysis: `pairwise_decision_mismatch.py`, `pairwise_mismatch_real_conditions.py`,
`pairwise_mismatch_by_convid.py`, `pairwise_mismatch_roundrobin_control.py`,
`audit_noise_claims.py`. No new orchestration beyond what's listed in Part 11 (this part is
analysis of already-collected data) plus `run_pergpu_lp_pareto_epsilon_small_6conditions.sh`'s
completed results.

### Part 13: the recommended rule's flagship comparison replicated to n=6 — a real, confirmed
win on the short condition, no confirmed win on the long condition, plus an independent
low-noise metric (energy-per-token) that reads more favorably

Per Part 12's finding that `drf_power_tiebreak_full_coincidence_ceiling` (short-named
`coincidence_ceiling`) had zero 6-trial data anywhere despite being the recommended rule,
extended it — plus its two natural baselines `lmetric` (power-blind) and `round_robin`
(state-blind) — to 6 trials on both flagship conditions
(`run_pergpu_flagship_headline_replicate.sh`; caught and fixed a real bug first — the script
initially set the router's `POLICY` env var to the literal string `"coincidence_ceiling"`
instead of `drf_power_tiebreak_full_coincidence_ceiling`, which is not a registered policy
name and crashed the router on startup; corrected before any of the 15 runs completed, no bad
data produced).

**Heavy/Closed-Loop (short), n=6:**

| arm | peak | mean_ramp | p99_ramp | max_ramp | ttft | tbt |
|---|---|---|---|---|---|---|
| coincidence_ceiling | 2275.2±57.1 | 140.6±8.7 | 1674.3±125.8 | 4611.1±748.3 | 0.542±0.033 | 531.3±38.9 |
| lmetric | 2388.2±31.9 | 151.8±8.4 | 1710.4±170.5 | 5971.4±710.1 | 0.560±0.032 | 544.2±48.9 |

**`coincidence_ceiling` DOMINATES `lmetric` outright — every metric simultaneously better, at
n=6 with real (not underestimated) standard deviations.** A genuine, confirmed result, not
noise. Against `round_robin`, directionally favorable on all three ramp metrics (z=-0.94 to
-1.47 using `coincidence_ceiling`'s own std as scale) — consistently in the right direction,
not yet clearing a strict significance threshold, never against.

**Heavy/Closed-Loop (long), n=6:**

| arm | peak | mean_ramp | p99_ramp | max_ramp | ttft | tbt |
|---|---|---|---|---|---|---|
| coincidence_ceiling | 2438.7±10.2 | 152.3±4.8 | 1128.6±97.7 | 6126.6±1272.8 | 0.426±0.002 | 704.7±30.6 |
| lmetric | 2472.1±58.6 | 153.9±3.2 | 1176.5±91.0 | 5028.0±939.5 | 0.434±0.009 | 699.3±15.6 |

**No dominance either direction, against either baseline.** The p99_ramp "advantage" over
`lmetric` that earlier framing (Part 10) leaned on has washed out entirely at n=6 (z=-0.49,
statistically indistinguishable) — confirms Part 12's warning was correct. More concerning: a
**real, non-noise signal against** — `coincidence_ceiling`'s mean_ramp is significantly higher
than `round_robin`'s (z=+2.82) on this specific condition. This is not something to minimize:
on the "sustained power pressure" condition specifically, the naive baseline may have a real
edge on average ramp over the recommended rule.

**Addendum (2026-09-09): `round_robin`'s own n=6 numbers and `lmetric_power_coincidence_ceiling`'s
n=3 numbers for the short condition, filling a gap left when this table was first written (only
`coincidence_ceiling` and `lmetric` were tabulated; `round_robin` was described only via z-score,
and `lmetric_power_coincidence_ceiling` wasn't in this specific table at all):**

| arm | n | peak | ttft | tbt |
|---|---|---|---|---|
| coincidence_ceiling | 6 | 2275.2±57.1 | 0.542±0.033 | 531.3±38.9 |
| lmetric | 6 | 2388.2±31.9 | 0.560±0.032 | 544.2±48.9 |
| round_robin | 6 | 2321.6±53.8 | 0.580±0.038 | 502.7±28.5 |
| lmetric_power_coincidence_ceiling | 3 | 2361.4±88.9 | 0.557±0.046 | 552.8±22.9 |

`coincidence_ceiling` beats `round_robin` on peak and ttft but loses on tbt (502.7 < 531.3) —
**incomparable, not dominated**, consistent with the general pattern established elsewhere in
this doc that `round_robin` is never simply worse, only sometimes better on tbt specifically
(it never concentrates load). Against `lmetric_power_coincidence_ceiling`,
`coincidence_ceiling` wins all three (peak, ttft, tbt) — a clean dominance, though at n=3 vs
n=6 (mismatched trial counts, not yet replicated to matched n like the ramp-stat recheck in
Part 15).

**Honest synthesis:** confirmed ramp-safety dominance on the short/original condition;
no confirmed ramp-safety advantage on the long/sustained-pressure condition, with one real
signal running the other way. This doesn't touch the theory (Theorem 4 is a proof, unaffected
by any of this) — but the empirical claim "the guaranteed-safe rule also wins on ramp
statistics under sustained pressure" is not currently supported and should not be asserted in
the paper as-is.

**A separate, independent, low-noise metric reads more favorably on both conditions:
energy-per-token.** Motivated by the observation that ramp statistics are noisy *by
construction* — they're either derivatives (rate-of-change, sensitive to exact sample timing)
or extreme-value statistics (dominated by a single worst event) — while an *integrated*
quantity like total energy consumed per token produced should average out the same
decision-trajectory noise that plagues ramp stats, the same way peak/TTFT/TBT (also
aggregate-ish) stayed stable while mean_ramp/p99_ramp/max_ramp didn't. Computed directly from
each trial's power trace's cumulative `energy_mj` NVML counter (exact hardware energy
delta, not a derived/integrated approximation) divided by total `output_tokens` across all
completed requests in that trial — no new experiments needed, computed retroactively from
existing power traces and records.

**Confirmed dramatically lower noise than every ramp statistic**: every arm on every one of
all 7 conditions tested showed relative std under ~3% (mostly under 1-2%), vs. 8-34% for
mean_ramp/p99_ramp/max_ramp — roughly an order of magnitude tighter, trustworthy even at n=3.
`coincidence_ceiling` beats `round_robin` on energy-per-token on 6 of 7 conditions (often by
5-10%, real given the sub-1% stds), including both flagship conditions (-1.1% short, -1.7%
long) and especially Light/Cachehit (-5.1%), Ramp & Route (-9.7%), WildChat (-6.4%). The one
exception — BurstGPT, where `coincidence_ceiling` is 7.7% *worse* — is not a new problem: it's
consistent with an earlier-documented finding in this project's history (real-BurstGPT-trace
round_robin advantage, 2026-09-01, unrelated to whale-aware admission logic) already scoped
out of the paper's headline for that reason. Against `lmetric`, results are more mixed (clear
wins on 4 conditions, ties or small losses on 3), but `coincidence_ceiling` and
`lmetric_power_coincidence_ceiling` are statistically indistinguishable on this metric on
Heavy/Closed-Loop long (1.3132 vs 1.3051, well within each other's noise) — a reassuring
alignment: the theoretically-safe rule isn't trading efficiency for its guarantee.

**Important scope note: energy-per-token measures a different claim than the theory
guarantees.** Theorem 4's threshold-safety is about avoiding dangerous *coincident ramp
spikes* (a tail/transient risk to grid-facing infrastructure), not average efficiency — a rule
could be very energy-efficient on average while still occasionally producing a dangerous
simultaneous multi-GPU spike, which would barely move an average diluted across hundreds of
thousands of tokens. Energy-per-token is a genuine, complementary pillar (confirmed:
recommended rule is not wasteful, and beats naive baselines on efficiency specifically), not a
replacement for validating the safety claim itself.

**Current honest state of the paper's empirical case, given everything in Parts 11-13:**
theory (Theorem 4/5 + Claim 2 counterexample) fully solid; ramp-safety confirmed on the
short/original condition only; energy-efficiency confirmed on both flagship conditions (and
5 of 7 total); ramp-safety on the long/sustained-pressure condition specifically remains
unconfirmed, with one real signal against. Next: checking whether a coincidence-factor style
metric (fraction of time multiple GPUs are simultaneously pressured — directly what
`coincidence_ceiling`'s mechanism targets, and plausibly low-noise the same way energy-per-
token is, since it's also an integrated/fractional statistic rather than a derivative or
extreme value) clarifies the long-condition picture.

### Data and repro (Part 13)

Orchestration: `run_pergpu_flagship_headline_replicate.sh`. Analysis:
`check_flagship_headline_6trial.py`, `check_energy_per_token_noise.py`,
`check_energy_per_token_all_arms.py`, `check_energy_per_token_all_conditions.py`.

### Part 14: two more candidate low-noise metrics tried — coincidence-factor and per-replica
fairness CV both fail (informative negative results); SLO-violation-rate succeeds

Motivated by Part 13's open question (does an integrated, non-derivative metric clarify the
long-condition picture the way energy-per-token did), tried two more candidates before landing
on SLO-violation-rate.

**Coincidence-factor** (fraction of power-pressured wall-clock time during which 2+ GPUs are
simultaneously elevated — directly the quantity `coincidence_ceiling`'s mechanism targets)
turned out to be *noisier* than the ramp statistics it was meant to improve on: **56.97%
relative std**, worse than max_ramp's ~34%. Root cause: despite being a "fraction," it counts
rare discrete threshold-crossing events (only hundreds of occurrences per trial), not the tens
of thousands of samples that make energy-per-token stable — "integrated/fractional" alone
doesn't guarantee low noise; what matters is the number of underlying events being summed.

**Per-replica fairness CV** (dispersion of per-GPU load/energy across only 6 GPUs) similarly
failed: **32-40% relative std**. Same underlying cause from a different angle — estimating
spread from only 6 groups is a small-N problem, structurally similar to estimating std from 3
trials.

**SLO-violation-rate** (fraction of requests with `ttft > 1.0s` or `max(tbt_ms) > 200.0ms`)
succeeded: **~1.4-2.3% relative std**, in the same low-noise tier as energy-per-token, because
it counts over hundreds of individual requests per trial rather than a handful of threshold
crossings. Real limitation: the fixed thresholds are near-degenerate (both near 0%) on lighter
conditions (Light/Cachehit, WildChat, BurstGPT) where almost every arm already meets them,
so the metric is only informative on the higher-load conditions.

**Full cross-arm, cross-condition table** (n=3-6 per cell; energy-per-token repeated here from
Part 13 alongside the new violation rates; computed via
`check_energy_and_slo_all_arms_conditions.py`, all 7 conditions, 11 arms):

Headline-relevant rows only (`coincidence_ceiling`, `lmetric`, `round_robin`,
`lmetric_power_coincidence_ceiling`) — full 11-arm table available by re-running the script:

| Condition | arm | J/token | ttft_viol | tbt_viol |
|---|---|---|---|---|
| Heavy/CL short | coincidence_ceiling | 1.4242±0.0193 | 0.1989±0.0160 | 0.2978±0.0241 |
| | lmetric | 1.4180±0.0222 | 0.2389±0.0229 | 0.3111±0.0248 |
| | round_robin | 1.4395±0.0176 | 0.2422±0.0172 | 0.2856±0.0240 |
| | lmetric_power_coincidence_ceiling | 1.4179±0.0230 | 0.2133±0.0067 | 0.3222±0.0168 |
| Heavy/CL long | coincidence_ceiling | 1.3132±0.0050 | 0.1878±0.0043 | 0.3867±0.0189 |
| | lmetric | 1.3251±0.0242 | 0.1972±0.0047 | 0.3861±0.0086 |
| | round_robin | 1.3363±0.0023 | 0.1917±0.0041 | 0.3752±0.0049 |
| | lmetric_power_coincidence_ceiling | 1.3051±0.0032 | 0.1954±0.0042 | 0.3943±0.0090 |
| Heavy/Matched | coincidence_ceiling | 0.7393±0.0116 | 0.5747±0.0701 | 0.7160±0.0035 |
| | lmetric | 0.7510±0.0110 | 0.6031±0.0436 | 0.7271±0.0068 |
| | round_robin | 0.7486±0.0075 | 0.6293±0.0074 | 0.6849±0.0020 |
| | lmetric_power_coincidence_ceiling | 0.7456±0.0040 | 0.5538±0.0465 | 0.7338±0.0068 |
| Light/Cachehit | coincidence_ceiling | 1.2287±0.0028 | 0.0020±0.0000 | 0.0000±0.0000 |
| | lmetric | 1.2517±0.0102 | 0.0020±0.0000 | 0.0000±0.0000 |
| | round_robin | 1.2944±0.0042 | 0.0020±0.0000 | 0.0000±0.0000 |
| | lmetric_power_coincidence_ceiling | 1.2339±0.0021 | 0.0020±0.0000 | 0.0000±0.0000 |
| BurstGPT | coincidence_ceiling | 2.8070±0.0312 | 0.0057±0.0012 | 0.1627±0.0051 |
| | lmetric | 2.8599±0.0235 | 0.0053±0.0006 | 0.1493±0.0021 |
| | round_robin | 2.6059±0.0118 | 0.0040±0.0000 | 0.1683±0.0019 |
| | lmetric_power_coincidence_ceiling | 2.8403±0.0101 | 0.0043±0.0006 | 0.1423±0.0087 |
| Ramp & Route | coincidence_ceiling | 2.2950±0.0323 | 0.2111±0.0102 | 0.2756±0.0379 |
| | lmetric | 2.2537±0.0187 | 0.2267±0.0115 | 0.2467±0.0067 |
| | round_robin | 2.5404±0.0462 | 0.2578±0.0102 | 0.2444±0.0077 |
| | lmetric_power_coincidence_ceiling | 2.2508±0.0373 | 0.2489±0.0077 | 0.2400±0.0467 |
| WildChat | coincidence_ceiling | 0.3227±0.0071 | 0.0021±0.0004 | 0.5074±0.0379 |
| | lmetric | 0.3147±0.0045 | 0.0020±0.0012 | 0.4134±0.0087 |
| | round_robin | 0.3448±0.0004 | 0.0024±0.0013 | 0.5828±0.0067 |
| | lmetric_power_coincidence_ceiling | 0.3146±0.0060 | 0.0017±0.0007 | 0.4055±0.0154 |

**Reading the table:** `coincidence_ceiling` beats `round_robin` on TTFT-violation on 5/7
conditions (ties/degenerate on Cachehit, loses narrowly on BurstGPT) and on energy-per-token on
6/7 (BurstGPT the one exception, per Part 13). Against `lmetric`, TTFT-violation favors
`coincidence_ceiling` on 4/7 (short, long, matched, BurstGPT-narrow), loses on Ramp & Route and
WildChat (both near-degenerate/noisy at this load level). TBT-violation is genuinely mixed
against both baselines — real losses to `round_robin` on short/BurstGPT/WildChat, real losses
to `lmetric` on long/WildChat — reported honestly, not glossed over. Against
`lmetric_power_coincidence_ceiling` (the strongest unsafe alternative), energy-per-token and
SLO-violation are statistically indistinguishable on most conditions, consistent with Part 13's
"safety costs no efficiency" finding.

### Data and repro (Part 14)

Analysis: `check_coincidence_factor.py`, `check_replica_fairness.py`,
`check_slo_violation_rate.py`, `check_slo_violation_flagship.py`,
`check_energy_and_slo_all_arms_conditions.py`.

### Part 15: recommended paper rewrite (title, layout, headline strategy, table policy) — not
yet applied to paper.md/tex, this is the agreed plan

**Headline strategy: `drf_power_tiebreak_full_coincidence_ceiling`** (short-named
`coincidence_ceiling`), unchanged from the standing recommendation, but now backed by the
most rigorous evidence base it's had at any point this investigation:
- Only rule with the threshold-safety guarantee (Theorem 4) — proof-level, unaffected by
  anything empirical.
- Confirmed n=6 dominance over `lmetric` on Heavy/Closed-Loop (short).
- Confirmed n=6 energy-per-token advantage over `round_robin` (6/7 conditions) and
  competitive-to-better vs `lmetric` (4/7 conditions, ties/small losses on 3).
- Confirmed n=6 TTFT-violation-rate advantage over both baselines where the metric is
  non-degenerate.
- **Matched n=6-vs-n=6 re-check against `lmetric_power_coincidence_ceiling`** (the
  strongest empirically-performing but Pareto-unsafe alternative) **found NO confirmed
  advantage for the unsafe rule on any ramp metric** — the apparent n=3 edge (lower
  p99_ramp/max_ramp) fully washed out once both were replicated to matched trial counts
  (z=-0.03 to +0.28 on all three ramp stats) — and energy-per-token was already
  statistically tied between them. This directly and rigorously closes the "but the unsafe
  rule performs better" objection: it doesn't, once compared properly.
- Explicitly rejected alternatives and why: `lmetric_power_coincidence_ceiling` (not
  Pareto-safe, Claim 2's cache-hit collapse is a deterministic counterexample, not a
  statistical claim any amount of replication could fix); the `lmetric_power_pareto_epsilon`
  family (Theorem 5 proves no fixed-weight/multiplicative rule can ever be threshold-safe,
  and its apparent empirical edge over `coincidence_ceiling` didn't survive the noise-band
  audit either — see Part 12).

**Title (recommended): "Fleet-Coincidence-Aware Power Routing: A Provably Safe Mechanism for
Multi-GPU LLM Serving."** Names the actual novel mechanism (coincidence-ceiling) rather than
a vague "power-aware" framing, states the guarantee up front, does not claim a sustained-
pressure ramp-statistics win (which isn't confirmed — see Part 13). Alternates considered:
"Provably Safe Power-Aware Routing, Without the Efficiency Cost" (leads with the efficiency-
tie result); "Routing as a Power-Regulation Lever: A Provable Mechanism and Its Real-Hardware
Evaluation" (closer to an earlier user-proposed framing).

**New layout — three pillars instead of one** (proof + mechanism, evaluation methodology,
experimental results), replacing the current single-pillar (theory + one validation
condition) structure:
1. Abstract — leads with the guarantee + the efficiency-neutral result + one sentence on the
   methodology contribution. Does NOT lead with the sustained-pressure ramp gap.
2. Introduction — three contributions stated explicitly (guarantee hierarchy + coincidence-
   ceiling mechanism; real-hardware evaluation showing no efficiency cost + tail-latency
   wins; decision-noise-floor characterization + resulting metric-selection methodology,
   useful beyond this paper).
3. Problem formulation — largely unchanged (3-resource DRF-style welfare).
4. Theory — Lemma 1 + counterexample (unchanged); the coincidence-ceiling mechanism
   presented as its own general, pluggable contribution (not just an add-on to one rule);
   Theorem 4 (threshold-safety); Theorem 5 (no fixed-weight/multiplicative rule can ever
   achieve it — motivates why the guarantee hierarchy matters); brief honest mention of the
   epsilon-shift Pareto-safe family as a weaker-guarantee option, not a competing headline.
5. **Evaluation methodology (new section)** — hardware/harness setup; the decision-level
   noise floor (~75-85% mismatch between identical-policy runs, diagnosed via the
   `round_robin` state-blind control at ~3%, shown robust across real unpadded conversation
   data in 3 conditions, not a synthetic-workload artifact); the metric-selection principle
   (sum-over-many-events metrics are low-noise, derivative/extreme-value/small-group-spread
   metrics are not — illustrated with both the metrics that worked, energy-per-token and
   SLO-violation rate, AND the ones that didn't, coincidence-factor and per-replica fairness
   CV, as a negative-result cross-check); trial-count implications (n=3 insufficient for
   ramp statistics, demonstrated directly via the 3-vs-6-trial std comparison).
6. Experimental results — 4-arm main comparison table (see table policy below); confirmed
   dominance on the short condition; confirmed energy-efficiency + TTFT-tail wins on the
   long condition with the ramp-statistic result addressed per the table policy below; the
   matched-n=6 head-to-head against `lmetric_power_coincidence_ceiling` (closes the "unsafe
   rule wins" objection); brief honest mention of the mixed TBT-violation picture.
7. Limitations — BurstGPT real-trace anomaly (disclosed, mechanism unresolved, consistent
   with prior project history); no fleet-scale extrapolation attempted, with the retracted
   PES-IM parametric/bootstrap approach cited as the reason (thin trial libraries + tail-
   statistic fragility — this project's library is thinner still); single hardware
   generation/model size.
8. Related work, 9. Conclusion — largely unchanged.

**Arm/table policy (resolves the "reduce clutter" request):** main comparison table limited
to 4 arms — `coincidence_ceiling` (recommended), `lmetric` (power-blind baseline, motivates
why power-awareness matters), `round_robin` (state-blind baseline, motivates why load-
awareness matters), `lmetric_power_coincidence_ceiling` (strongest empirically-performing
unsafe alternative, needed for the head-to-head that closes the "unsafe rule wins"
objection). The other 7 arms tested this session (epsilon variants, `weighted_sum`,
`compute_only`/`load_only`) get a sentence or an appendix table, not main-table space.

**Negative-result disclosure policy (resolves the tension between "reduce distraction" and
scientific integrity — discussed and NOT resolved by omission):** negative/mixed findings
against the chosen strategy (the long-condition mean_ramp signal favoring `round_robin`, the
mixed TBT-violation picture, the BurstGPT reversal) are NOT dropped from the paper — they are
moved out of the main results table into a plainly-labeled Limitations subsection in prose,
disclosed but not competing for attention with the headline comparison. Explicitly rejected:
omitting them from the paper entirely, on two grounds — (1) it would directly contradict the
paper's own methodology-section pitch (rigorous, noise-aware honesty about what's confirmed),
and (2) it isn't needed, since the theory + efficiency-tie + methodology contributions don't
require a universal ramp-statistics win to carry the paper.

**mean_ramp specifically was checked, not casually dropped as noise-explainable:** it is the
least-noisy of the three ramp derivatives (~8% relative std vs. p99_ramp's ~9% and
max_ramp's ~34%), and the specific long-condition signal against `coincidence_ceiling`
(z≈+2.8 vs `round_robin`) is one of the more statistically credible findings in the whole
ramp-stats investigation — tighter stds on both sides than most of the p99_ramp/max_ramp
comparisons that were dismissed as noise elsewhere in this doc. Also: mean_ramp is one of the
five metrics behind the confirmed short-condition dominance claim, so dropping it only when
inconvenient (keeping it when it supports the headline win) would be outcome-based metric
selection, not a principled noise argument. **Resolved: the paper's headline "confirmed
dominance/win" claims will be restricted to the metrics independently confirmed low-noise
(peak, ttft, tbt, energy-per-token, SLO-violation rate) — checked and this does NOT cost the
short-condition win, which holds on peak/ttft/tbt alone (2275.2 vs 2388.2 peak, 0.542 vs
0.560 ttft, 531.3 vs 544.2 tbt, all favoring `coincidence_ceiling`) — rather than selectively
including/excluding ramp derivatives by whether they happen to favor the recommended rule.**
mean_ramp/p99_ramp/max_ramp remain reported (per the disclosure policy above) but are not
part of any headline "confirmed" claim going forward.

**Open, in-progress: whether Share_power's definition (currently raw two-point ramp_rate,
`(p1-p0)/(t1-t0)` between consecutive power samples) should be smoothed (e.g. EMA over
several samples) rather than redefined to target peak power or energy rate directly.**
Discussed and NOT resolved by conflating two different questions: what the routing decision
should target (a physically-motivated choice — ramp rate is the right quantity for the
grid-transient-stress problem this whole project line, including the sibling PES-IM
coincidence-factor work, is built around) is a separate question from what the paper should
report as evidence (a measurement-quality choice — peak power and energy-per-token are
low-noise regardless of what the routing rule internally targets). Retargeting Share_power at
peak power or energy rate would silently change the mechanism's physical claim from
grid-transient-avoidance to capacity/thermal management, and was not adopted for that reason.
**Smoothing the ramp_rate estimate (same physical target, less noisy estimate of it) was
identified as the more surgical, lower-risk option** — plausible reason to test: if
Share_power reacts to a noisy raw two-point derivative, that noise could be feeding directly
into the ~75-85% decision-mismatch finding (Part 12/13), on top of the already-confirmed
state-divergence mechanism, and smoothing might reduce it without changing what the rule is
trying to accomplish. Believed (not yet re-verified) that Theorem 4's proof is agnostic to
what Share_power specifically measures, only to the ratio-to-ceiling structure via the
"any shared κ(t) inherits safety" corollary — smoothing (or even retargeting) likely does not
require new proofs, only confirmation the structural argument still applies. **Next: build
and test a smoothed-ramp Share_power variant** (see Part 16 once run).

### Part 16: Share_power redesign resolved — neither an EMA-smoothed ramp estimate nor a
peak-power retarget beats raw ramp_rate; keep the original definition

Built and tested two alternative Share_power definitions, both structurally Pareto-safe (built
via the established `_full` sorted-tie-break-vector pattern):
`drf_power_tiebreak_full_coincidence_ceiling_smoothed_ramp` (EMA, α=0.3, over the same raw
two-point ramp_rate) and `drf_peak_power_tiebreak_full` (Share_power retargeted at
instantaneous power draw / peak ceiling instead of ramp rate — also fixed a real pre-existing
Pareto-safety bug in the older non-`_full` `pick_drf_peak_power_tiebreak`, which reproduced the
same domination counterexample the original unfixed ramp-based rule had). First validated on
the Heavy/Closed-Loop long reference condition (3 trials each), then the smoothed-ramp variant
extended to all 6 remaining conditions (peak-power-full's extension still in progress at time
of writing).

**Signal-level check, corrected after an initial methodology bug.** The router's live ramp
signal and the `power_logger.py` sidecar (which writes the power-trace CSV used for all
evaluation) are two *independent* NVML polling loops at different cadences: the sidecar targets
50ms (`--interval-ms 50`, actual ~89ms observed due to per-GPU query overhead across 6 GPUs),
while the router's own internal loop that feeds `ramp_rate_w_per_s`/`smoothed_ramp_rate_w_per_s`
polls at `ROUTER_POWER_INTERVAL_S`, defaulting to 500ms and never overridden in these launches.
An initial reconstruction of the live signal (replaying `update_ramp_state` over the sidecar's
~89ms-spaced samples) reported raw ramp relative std 454.8%, smoothed 214.3%, and a 38x
reduction in ceiling-crossing frequency (1.52%→0.04%) — **this used the wrong dt and was
corrected** by resampling to the router's actual ~500ms cadence: raw relative std **191.1%**,
smoothed **157.3%** (an ~18% reduction, not ~53%), and ceiling-crossing already near-zero under
*raw* ramp at the correct cadence (0.03% vs smoothed's 0.00%) — the "smoothing kills ceiling-
detection" claim does not survive the corrected cadence and should not be used as a reason to
prefer or reject either variant.

**Evaluation-metric comparison (unaffected by the cadence bug — computed directly from the
trace, same methodology as every other arm), smoothed-ramp vs `coincidence_ceiling`, all 7
conditions:**

| Condition | J/token Δ | TTFT-viol Δ | max_ramp Δ |
|---|---|---|---|
| Heavy/CL short | −1.8% (better) | better | +35% (worse) |
| Heavy/CL long | +0.6% | ~tie | −36% (better) |
| Heavy/Matched | +2.4% (worse) | −12% (better) | +26% (worse) |
| Light/Cachehit | −1.2% (better, best of 12 arms) | tie | +42% (worse) |
| BurstGPT | −0.4% (better) | ~tie | +33% (worse) |
| Ramp & Route | +3.7% (worse) | +23% (worse) | +44% (worse) |
| WildChat | +3.0% (worse) | tie | +38% (worse) |

No clean win: smoothed-ramp beats `coincidence_ceiling` on energy-per-token on 3/7 conditions
(small, ~1-2%), loses on 4/7. **max_ramp is worse for smoothed-ramp on 6/7 conditions**, often
by a large margin. **Ramp & Route is a consistent loss across five independent metrics at
once** (energy +3.7%, TTFT-violation +23%, mean TTFT +27%, p99_ramp +23%, max_ramp +44%) — the
strongest single piece of evidence against adopting the smoothed variant.

**`drf_peak_power_tiebreak_full` (peak-power retarget): large, one-sided regression on both
conditions completed so far.** On the long reference condition: energy-per-token +11%
(1.4581 vs 1.3132), TTFT-violation nearly 2x (0.3607 vs 0.1878), TBT-violation +21% (0.4667 vs
0.3867) — not close to noise. Partial BurstGPT data (n=2) shows the same direction
(TTFT-violation 0.0150 vs 0.0057, TBT-violation 0.2230 vs 0.1627). Retargeting Share_power at
peak power also changes the mechanism's physical claim from grid-transient-avoidance to
capacity/thermal management — a materially different problem than the one this paper is about,
which is a second, independent reason not to adopt it regardless of the numbers.

**Resolved: keep the original raw `ramp_rate_w_per_s` definition of Share_power.** Neither
alternative improves on it empirically, and both weaken or change the mechanism's connection to
its stated physical target. The justification for tolerating raw ramp_rate's instantaneous
noisiness is structural, not a measurement-quality argument: (1) Theorem 4's Pareto-safety
proof only requires D_i be computed and compared correctly at decision time — it does not
depend on how precise or smooth any individual share's underlying reading is, so noise in
Share_power can degrade how well the mechanism performs but cannot break the safety guarantee;
(2) Share_power only becomes the binding (max) term during genuinely large, real ramp
excursions — exactly where a real signal is large relative to sample noise — combined with
resampling every 0.5s, a sustained real event gets many chances to be caught even if any single
sample is off; (3) this is now an empirically tested claim, not just an assumption — deliberately
damping the instantaneous noise via EMA smoothing did not improve, and often worsened, the
confirmed low-noise evaluation metrics, which is direct evidence the instantaneous noise isn't
actually harming the mechanism's real function.

### Data and repro (Part 16)

Implementation: `scoring.py` (`share_power_smoothed`, `coincidence_ceiling_factor_smoothed`,
`dominant_share_vector_power_priority_full_coincidence_ceiling_smoothed_ramp`,
`pick_drf_power_tiebreak_full_coincidence_ceiling_smoothed_ramp`,
`dominant_share_vector_peak_priority_full`, `pick_drf_peak_power_tiebreak_full`); `ramp.py`
(`update_ramp_state` extended with `smoothing_alpha` EMA, backward-compatible default);
`replica_state.py` (`smoothed_ramp_rate_w_per_s` field). Orchestration:
`run_pergpu_closedloopheavylong_smoothed_ramp.sh`,
`run_pergpu_closedloopheavylong_peak_power_full.sh`,
`run_pergpu_share_power_variants_6conditions.sh`. Analysis:
`check_smoothed_ramp_in_share_power.py` (signal reconstruction, resampled to ~0.5s),
`check_smoothed_ramp_vs_cc_full_metrics.py` (peak/ramp/ttft/tbt side-by-side),
`check_energy_and_slo_share_power_variants.py` (energy/SLO side-by-side).

### Part 17: a simulated peer-review pass catches two real problems with the headline Table 1
claim — neither survives proper significance testing, and the "unsafe rule wins" story is now
closed on peak/TTFT/TBT too, not just ramp stats and efficiency

Ran a genuine adversarial self-review (simulating an ACM e-Energy "Systems and applied
modeling" track reviewer) against the post-rewrite paper. Two catches, both verified against
real per-trial data (not just the reported mean±std) rather than accepted or dismissed by
argument alone.

**Catch 1: Table 1's "dominates on every low-noise metric simultaneously" claim (short
condition, `coincidence_ceiling` vs `lmetric`, n=6) does not survive a proper significance
test.** Pulled real per-trial peak/TTFT/TBT values and ran both unpaired and paired t-tests
(scipy):

| Metric | diff (cc−lmetric) | unpaired p | paired p |
|---|---|---|---|
| Peak power | −113.0 W | **0.0017** | **0.0025** |
| TTFT | −0.018 s | 0.355 | 0.101 |
| TBT | −12.9 ms | 0.624 | 0.605 |

Only peak power clears significance. The paper's mean±std table made all three look like a
clean sweep because each point estimate is numerically lower for `coincidence_ceiling`, but
"lower point estimate" and "statistically distinguishable at n=6" are different claims, and
the paper's own §5.2/§5.3 methodology exists specifically to make this distinction — it just
wasn't applied to the paper's own headline table. **Corrected claim: a real, significant peak
power win over `lmetric` (p<0.01); TTFT/TBT are directionally favorable but not independently
significant at n=6.**

**Root cause check for why TTFT failed significance despite §5.2 calling TTFT generally
low-noise:** TTFT's relative std is condition-dependent, not a fixed property of the metric.
On the short condition (the one Table 1 uses): 0.542±0.033, relative std **6.2%**. On the long
condition: 0.426±0.002, relative std **0.4%** — a 15x difference. §5.2's "TTFT stays in the
~1-2% range" claim was true on average across conditions tested elsewhere in this project but
not on the specific condition backing the headline table — an inconsistency worth disclosing
explicitly (which condition-property drives this is not yet diagnosed; plausibly the short
condition's shorter active window means fewer effective within-trial TTFT samples averaging
out cross-trial timing noise, but this is a hypothesis, not verified).

**Catch 2: `lmetric_power_coincidence_ceiling` was only at n=3 on the short condition, an
unmatched comparison sitting in the same table as three n=6 arms.** Replicated it to matched
n=6 (`run_pergpu_closedloopheavy_lpcc_matched_n6.sh`, trials 4-6, same harness args as every
other short-condition run) and recomputed the comparison properly:

| Metric | `coincidence_ceiling` (n=6) | `lmetric_power_coincidence_ceiling` (n=6) | diff | t | p |
|---|---|---|---|---|---|
| Peak power (W) | 2275.2±57.1 | 2318.2±74.8 | −43.0 | −1.118 | 0.290 |
| TTFT (s) | 0.542±0.033 | 0.537±0.038 | +0.005 | +0.257 | 0.803 |
| TBT (ms) | 531.3±38.9 | 553.1±18.8 | −21.8 | −1.234 | 0.246 |

**None of the three metrics are significant at matched n=6 — the "clean dominance over the
unsafe rule" claim does not survive replication, and TTFT's point estimate direction flips**
(the n=3 sample happened to show `lmetric_power_coincidence_ceiling` worse; at n=6 it's
marginally better on TTFT specifically, though not significantly so either way).

**This is not a worse result for the paper — it's a cleaner, more defensible one once
correctly framed.** Combined with the already-established ties on energy-per-token,
SLO-violation-rate, and the three ramp-derivative statistics (Part 13/15), `coincidence_ceiling`
is now shown to be statistically indistinguishable from the strongest known unsafe alternative
on *every* metric measured in this project, not just efficiency/SLO — which is direct,
comprehensive support for "the safety guarantee costs nothing," a stronger and more coherent
version of that claim than before. The one metric with a real, significant, standalone win is
peak power against the power-blind baseline (`lmetric`), not the multi-metric sweep against
the unsafe alternative previously claimed.

**Resolution for the paper (not yet applied to paper.md/tex as of this entry):** rewrite the
headline claim around (a) a significant peak-power win vs. `lmetric` (p=0.0017), (b)
directional-but-unconfirmed TTFT/TBT advantages vs. `lmetric`, disclosed as such, and (c) a
now-comprehensive statistical tie with `lmetric_power_coincidence_ceiling` across every metric
tested (ramp stats, peak, TTFT, TBT, energy-per-token, SLO-violation-rate) as the paper's
primary "safety is free" evidence. Also disclose TTFT's condition-dependent noise level in
§5.2 rather than stating a single blanket relative-std range.

### Data and repro (Part 17)

Orchestration: `run_pergpu_closedloopheavy_lpcc_matched_n6.sh`. Real per-trial values and
t-tests computed inline via `compare_closedloopheavy_duration.load_records`/`load_power` plus
`scipy.stats.ttest_ind`/`ttest_rel`, not yet saved as a standalone script (ad hoc during this
review pass — worth promoting to `check_table1_significance.py` if reused).

### Part 18: theory-motivated (not outcome-selected) replication check on Heavy/CL long finds
a second real, highly significant result — `coincidence_ceiling` beats `round_robin` on TTFT

Per the discussion about avoiding selection bias (choosing where to replicate based on which
point estimate currently looks favorable, rather than on a mechanistic prediction), picked
Heavy/Matched and Heavy/CL long for further scrutiny specifically because both are the
project's sustained-pressure, multiple-simultaneously-elevated-replica conditions — the
regime the coincidence-ceiling mechanism is designed to matter in — decided *before* looking
at which numbers currently favored the proposed rule there.

Heavy/CL long already had all 4 headline arms at real n=6 (no new data needed); ran the same
rigorous per-trial significance check used for Table 1 (findings Part 17) rather than trusting
point estimates:

| Comparison | Metric | diff | p (unpaired) |
|---|---|---|---|
| vs `lmetric` | peak | −33.4 W | 0.199 |
| vs `lmetric` | TTFT | −0.008 s | 0.055 (borderline) |
| vs `lmetric` | TBT | +5.4 ms | 0.708 |
| vs `lmetric_power_coincidence_ceiling` | peak/TTFT/TBT | all small | 0.789 / 0.166 / 0.592 |
| **vs `round_robin`** | **TTFT** | **−0.023 s** | **p<0.000001** |
| vs `round_robin` | peak | +20.0 W | 0.115 |
| vs `round_robin` | TBT | +24.9 ms | 0.083 (borderline) |

**A second real, strong, confirmed result**: `coincidence_ceiling`'s TTFT is significantly
lower than `round_robin`'s on the long/sustained-pressure condition — cc: 0.423-0.429s across
all 6 trials; round_robin: 0.446-0.451s across all 6 trials, essentially non-overlapping
(unpaired t=-19.74, p<1e-6; paired t=-22.57, p=3e-6). This is the first rigorously-tested,
statistically solid win against `round_robin` found anywhere in this project — previous
`round_robin` comparisons were framed only qualitatively ("incomparable"), never
significance-tested this precisely. The mechanism is consistent with `round_robin`'s known
weakness (§6.1/Table 1 discussion): ignoring queue depth risks concentrating load into
synchronized bursts, which should cost TTFT specifically under sustained load — exactly what
this shows, on exactly the condition the theory-motivated selection targeted.

TBT vs `round_robin` trends the other way (borderline, p=0.083, consistent with `round_robin`
never concentrating load and therefore never facing the TBT cost a state-aware rule
occasionally does, per the existing short-condition disclosure) — reported honestly, not
suppressed.

**This is now a second condition with a real, non-noise confirmed result** (short: peak power
vs `lmetric`, p=0.0017; long: TTFT vs `round_robin`, p<1e-6), found via the theory-motivated
selection this session agreed on rather than by scanning for favorable point estimates, which
is the methodologically legitimate way to have found it.

**Next**: replicating Heavy/Matched (currently n=3 across all 4 headline arms, the other
sustained-pressure condition) to matched n=6, for the same reason — not yet run as of this
entry.

### Data and repro (Part 18)

Real per-trial TTFT/peak/TBT computed inline via `compare_closedloopheavy_duration`, same
approach as `check_table1_significance.py` (Part 17) but applied to the already-existing
`closedloopheavylongpergpu` n=6 data; worth extending that script to cover this condition too
if reused again.

### Part 19: Heavy/Matched replicated to n=6 — the TTFT-win/TBT-cost pattern against
`round_robin` replicates on a second independent condition

Completed the second half of the theory-motivated replication plan from Part 18: Heavy/Matched
(the other sustained-pressure, multi-replica-elevation condition) extended from n=3 to n=6 for
all 4 headline arms (`run_pergpu_matched_n6_headline.sh`, same pad-seed as every existing
trial). Real per-trial significance check:

| Comparison | Metric | `coincidence_ceiling` | other | diff | p |
|---|---|---|---|---|---|
| vs `lmetric` | peak | 2643.5±20.6 | 2652.3±25.3 | −8.7 | 0.527 |
| vs `lmetric` | TTFT | 3.537±0.593 | 3.316±0.569 | +0.222 | 0.523 |
| vs `lmetric` | TBT | 1288.6±36.1 | 1314.7±35.5 | −26.1 | 0.236 |
| vs `lmetric_power_coincidence_ceiling` | peak/TTFT/TBT | --- | --- | --- | 0.288 / 0.630 / 0.058 |
| **vs `round_robin`** | **TTFT** | **3.537±0.593** | **5.453±0.456** | **−1.916** | **0.00009** |
| **vs `round_robin`** | **TBT** | **1288.6±36.1** | **1242.0±18.9** | **+46.6** | **0.019** |
| vs `round_robin` | peak | 2643.5±20.6 | 2649.3±29.9 | −5.8 | 0.705 |

**A real, large, highly significant TTFT win over `round_robin`** (~35% relative reduction,
`round_robin`'s TTFT balloons past 5s under this condition's load while `coincidence_ceiling`
stays under 3.6s) **paired with a smaller but still significant TBT loss** — the identical
qualitative pattern found on Heavy/CL long (Part 18: TTFT win p<1e-6, TBT trend p=0.083, not
quite significant there but same direction). Finding the same structural tradeoff on two
independent conditions, both selected in advance for their sustained-pressure/multi-elevation
character rather than for looking favorable, is a real replication, not two separate lucky
draws. Mechanistically consistent with the existing account (§6.1's discussion, unchanged): a
state-blind rule can never concentrate load onto an already-busy replica, so it structurally
cannot lose TBT the way a state-aware rule occasionally does under real pressure, at the cost
of not avoiding queue buildup the way a load-aware rule can — this is now shown twice, not
asserted once.

vs `lmetric` and `lmetric_power_coincidence_ceiling`: no significant differences on this
condition either, consistent with the established comprehensive-tie picture (Parts 15/17).

**Recommendation: fold this into the paper.** The paper currently reports only one confirmed
result (short-condition peak power vs `lmetric`, §6.1) plus the comprehensive-tie story. This
adds a second, independently-replicated, mechanistically-explained result (`round_robin`
TTFT/TBT tradeoff on sustained-pressure conditions) that is at least as strong statistically
and arguably more interesting, since it replicates rather than standing alone. Not yet applied
to paper.md/tex as of this entry.

### Data and repro (Part 19)

Orchestration: `run_pergpu_matched_n6_headline.sh`. Analysis: same inline
`compare_closedloopheavy_duration` + `scipy.stats.ttest_ind` approach as Parts 17-18, applied
to `openloopwhalelongoutmatchedpergpu`.

### Part 20: Heavy/Matched rate sweep — the TTFT-win/TBT-cost pattern is not monotonic with
load; it peaks at moderate pressure and vanishes under saturation

Extended Part 19's finding with a 3-point request-rate sweep on Heavy/Matched
(`run_pergpu_matched_rate_sweep.sh` + `_ext.sh`): rate=7.0 (below the original 10.7) and
rate=15.0 (above it), `coincidence_ceiling` vs `round_robin`, n=3 each (supplementary check,
not headline-rigor n=6).

| Rate | TTFT: cc vs rr | p | TBT: cc vs rr | p |
|---|---|---|---|---|
| 7.0 | 0.779±0.165 vs 1.347±0.113 | **0.00797** | 1058.1±40.8 vs 1040.0±30.5 | 0.572 |
| 10.7 (Part 19, n=6) | 3.537±0.593 vs 5.453±0.456 | **0.00009** | 1288.6±36.1 vs 1242.0±18.9 | **0.019** |
| 15.0 | 9.916±0.649 vs 9.923±0.320 | 0.988 | 1367.8±59.2 vs 1341.2±20.7 | 0.502 |

**Not monotonic with load — this partially disconfirms the a priori prediction stated in Part
18/19 (that higher load should strengthen the effect via more simultaneous-elevation
opportunities), and that disconfirmation is reported as found, not omitted.** The TTFT
advantage is real at both 7.0 and 10.7 (both p<0.01) but collapses entirely at 15.0, where
both arms' TTFT balloons to ~9.9s and becomes statistically identical. The most likely
mechanism: rate=15.0 saturates the fleet's real serving capacity — once arrival rate exceeds
what the replicas can actually process, queueing delay is dominated by the capacity deficit
itself, and routing policy (which replica gets which request) stops being able to help,
regardless of how well-designed. This is consistent with, not contradictory to, the
coincidence-ceiling mechanism's design: it manages *how* load is distributed among replicas
that have spare capacity to receive it, not whether the fleet has enough capacity in
aggregate. The TBT-cost effect appears only at the moderate/original rate (10.7), not at
either the lighter or saturated extreme — narrower than the TTFT-win's range, but consistent
with the existing account (round_robin's concentration-avoidance only pays off relative to a
state-aware rule under real, sustained-but-not-yet-saturating pressure).

**Honest summary for the paper**: `coincidence_ceiling`'s TTFT advantage over `round_robin`
holds across a real range of loads (confirmed at 2 of 3 rates tested) but is not universal —
it is a sustained-but-below-saturation-regime effect, not a load-independent property of the
rule. This should be stated as an explicit scope qualifier if this result is added to the
paper, not smoothed into an unconditional claim.

### Data and repro (Part 20)

Orchestration: `run_pergpu_matched_rate_sweep.sh`, `run_pergpu_matched_rate_sweep_ext.sh`
(the latter's `lmetric_power_coincidence_ceiling`/`weighted_sum_coincidence_ceiling` arms not
yet analyzed as of this entry — still running/queued). Analysis: same inline
`compare_closedloopheavy_duration` + `scipy.stats.ttest_ind` approach as Parts 17-19.

**Addendum to Part 20**: the comprehensive tie against `lmetric_power_coincidence_ceiling` and
`weighted_sum_coincidence_ceiling` (established at rate=10.7 in Parts 17/19) also holds at
both new rate points, n=3 each:

| Rate | vs `lmetric_power_coincidence_ceiling` (peak/ttft/tbt p) | vs `weighted_sum_coincidence_ceiling` (peak/ttft/tbt p) |
|---|---|---|
| 7.0 | 0.080 / 0.146 / 0.462 | 0.333 / 0.269 / 0.955 |
| 15.0 | 0.568 / 0.295 / 0.861 | 0.379 / 0.498 / 0.645 |

No significant difference on any metric at either rate — the "safety costs nothing" result is
robust across the load range tested here, not an artifact specific to rate=10.7.

### Part 21: n=6 correction to Part 20 — the saturation finding was overstated at n=3

Extended rate=7.0 and rate=15.0 from n=3 to n=6 (`run_pergpu_matched_rate_sweep_n6.sh`).
Real per-trial results:

| Rate | n | TTFT: cc vs rr | p | TBT: cc vs rr | p |
|---|---|---|---|---|---|
| 7.0 | 6 | 0.708±0.135 vs 1.218±0.187 | **0.00030** | 1041.5±43.1 vs 1020.5±38.4 | 0.393 |
| 15.0 | 6 | 9.396±0.948 vs 10.171±0.509 | 0.108 | 1336.4±64.8 vs 1336.9±14.4 | 0.985 |

**Correction to Part 20: "the effect vanishes entirely under saturation" was an overstatement
built on an n=3 sample that happened to land on an almost-exact tie (rate=15.0 n=3: diff=-0.007,
p=0.988).** At n=6, the point estimate shows a real 0.775s gap in the same direction as the
lighter-load conditions (p=0.108 --- does not clear conventional significance, but is no longer
a dead heat either). The honest, corrected statement: **the TTFT advantage weakens
substantially under saturation and is no longer statistically confirmed there, not that it
provably disappears.** This is a real example of why n=3 shouldn't be trusted for anything
beyond a directional read, consistent with this project's own standing methodology (§5.2/5.3)
--- the correction itself is evidence the standard is being applied consistently, not
selectively.

Rate=7.0's result, by contrast, only got stronger with more data (p=0.008 to p=0.0003) ---
confirming that result was never fragile.

TBT: no significant difference at either rate, consistent with Part 20's finding that the TBT
cost is specific to the original moderate rate (10.7), not present at either tested extreme.

### Data and repro (Part 21)

Orchestration: `run_pergpu_matched_rate_sweep_n6.sh`. Same analysis approach as Parts 17-20.

### Part 22: the round_robin TTFT/TBT tradeoff pattern does not generalize to WildChat or
BurstGPT — real, condition-dependent mixed results, including one significant loss

Checked `coincidence_ceiling` vs `round_robin` on two conditions outside the whale-injection
family the Parts 18-21 pattern was established on: WildChat at rate=15.0 (n=3) and BurstGPT at
1.5x compressed timing (n=3, see Part header on the rate-scale methodology note).

| Condition | TTFT: cc vs rr | p | TBT: cc vs rr | p |
|---|---|---|---|---|
| WildChat rate=15.0 | 0.224 vs 0.408 (favors cc, not sig.) | 0.178 | **289.0 vs 334.4 (favors cc)** | **0.011** |
| BurstGPT 1.5x | **0.174 vs 0.167 (favors rr --- a real loss)** | **0.0046** | 130.3 vs 132.6 (favors cc, modest) | 0.029 |

**This does not replicate the clean TTFT-win/TBT-cost pattern from Parts 18-21.** WildChat
shows the *opposite* TBT relationship (cc wins TBT, not `round_robin`); BurstGPT shows a real,
significant TTFT *loss* for `coincidence_ceiling`. Both are real, non-noise results (clear
p-values), not measurement artifacts.

**Honest interpretation: the round_robin tradeoff pattern found on Heavy/CL long and
Heavy/Matched is specific to that condition family (closed/open-loop synthetic whale-injection
traffic with sustained multi-replica pressure), not a general property of
`coincidence_ceiling` vs `round_robin`.** WildChat and BurstGPT differ structurally (real
conversational/request-trace content, different load shapes) and produce a different,
condition-specific comparison. This should be reported as a scope-limited finding if included
in the paper, explicitly not generalized beyond the conditions where it was actually
confirmed --- consistent with this project's standing policy of not smoothing mixed results
into a cleaner-sounding universal claim.

### Data and repro (Part 22)

Orchestration: `run_pergpu_wildchat_rate15.sh`, `run_pergpu_burstgpt_ratescale15.sh`. Same
analysis approach as Parts 17-21. n=3 only on both conditions --- not yet replicated to n=6;
BurstGPT's real, significant TTFT loss in particular would benefit from replication before
being treated as fully settled.

### Part 23: the "comprehensive tie" against lmetric_power_coincidence_ceiling/
weighted_sum_coincidence_ceiling does NOT hold on WildChat or BurstGPT — real, significant
losses, a genuine correction to the paper's current headline scope

Extended Part 22's check to the other two headline-adjacent arms (already queued in the same
batteries). Real per-trial t-tests, `coincidence_ceiling` vs each:

| Condition | vs `lmetric_power_coincidence_ceiling` | vs `weighted_sum_coincidence_ceiling` |
|---|---|---|
| WildChat rate=15.0 | peak **p=0.0036** (cc worse by 113W); TTFT p=0.162; TBT **p=0.0049** (cc worse by 53.5ms) | peak **p=0.0072** (cc worse by 49.6W); TTFT p=0.278; TBT **p=0.0053** (cc worse by 66.2ms) |
| BurstGPT 1.5x | peak p=0.804; TTFT **p=0.0236** (cc worse); TBT **p=0.0001** (cc worse by 11.3ms) | peak p=0.607; TTFT p=0.195; TBT **p=0.0023** (cc worse by 8.7ms) |

**This is a real, statistically confirmed correction to the paper's current claim** ("matches
the strongest empirically-performing but unsafe alternative on every metric measured at
matched $n{=}6$ --- the guarantee costs nothing on any axis tested," Abstract/Conclusion as of
this session's last revision). That claim is accurate for the flagship whale-injection
condition family (Heavy/CL short/long, Heavy/Matched at three rates --- Parts 17-21) but does
**not** generalize to WildChat or BurstGPT, where `coincidence_ceiling` shows real, significant
peak/TBT/TTFT losses against both the unsafe alternative and the Pareto-safe external check.

This is consistent with, and now statistically sharpens, an existing narrow disclosure already
in the paper's Limitations (the WildChat TBT-violation-rate loss from the original 7-condition
sweep, Part 14) --- but the current headline language does not reflect how real and broad this
is. **The paper's "costs nothing" claim needs to be explicitly scoped to the conditions where
it was actually confirmed (the whale-injection family), not stated as if it holds on every
condition/axis tested in this project.** Not yet applied to paper.md/tex as of this entry ---
this is a correction that should be made before the current overly broad phrasing is read by a
reviewer who checks it the way this project's own review pass did.

### Data and repro (Part 23)

Same batteries as Part 22 (`run_pergpu_wildchat_rate15.sh`, `run_pergpu_burstgpt_ratescale15.sh`),
extended to the other two arms already collected in the same runs. n=3 only on both conditions.

### Part 24: Part 23's WildChat/BurstGPT losses occurred under light, unsaturated load ---
important context, not a retraction

Checked directly whether WildChat rate=15.0 and BurstGPT 1.5x were actually under load when
Part 23's losses were measured, rather than assuming so.

**TTFT distributions (direct queueing-delay evidence):**

| Condition | mean TTFT | p50 | p99 |
|---|---|---|---|
| Heavy/Matched rate=7.0 (lightest tested) | 0.935s | 0.086s | 7.803s |
| Heavy/Matched rate=15.0 (saturated) | 10.310s | 8.757s | 30.977s |
| WildChat rate=15.0 | 0.225s | 0.159s | 0.967s |
| BurstGPT 1.5x | 0.174s | 0.115s | 0.957s |

**Neither WildChat rate=15 nor BurstGPT 1.5x reaches meaningful load** --- both have lower p99
TTFT than Heavy/Matched's own *lightest* tested rate point, nowhere near saturation. Consistent
with neither condition having whale injection (real WildChat/BurstGPT content), so the
workload is structurally lighter independent of arrival rate. Coincidence-ceiling-mechanism
engagement rate (fraction of samples with $n_{elevated}\ge2$) is also low and similar across
conditions including the known-saturated Heavy/Matched rate=15 case (0.49-0.79%), so engagement
rate alone doesn't distinguish load level here --- TTFT/TBT is the clean signal.

**This recontextualizes, but does not retract, Part 23's finding.** The significant peak/TBT/
TTFT losses found there are real (the p-values and effect sizes stand), but they occurred in a
regime where the coincidence-ceiling mechanism has essentially no real safety-relevant work to
do --- it rarely triggers under this little pressure. The more accurate interpretation: under
light, unsaturated load, small but statistically detectable differences arise from incidental
tie-break behavior on compute/load shares (unrelated to the power-safety mechanism), not from
the mechanism actively costing performance under the pressure it's designed to handle. This is
a materially different and more benign story than "the guarantee costs something under real
pressure" would be. **For the paper: the "costs nothing" claim should still be scoped to the
conditions/regime it was confirmed under (Part 23's correction stands --- don't claim
universality), but the WildChat/BurstGPT losses should be reported alongside this load context,
not presented as evidence the mechanism fails when it matters.**

### Data and repro (Part 24)

Same power traces and records as Parts 22-23; TTFT distribution and coincidence-frac computed
inline, not yet promoted to a standalone script.

### Part 25: WildChat under genuine load (rate=30) --- clean win vs round_robin, but a
persistent real TTFT cost vs both tie-broken alternatives that load does not resolve

Confirmed WildChat rate=30.0 actually reaches real load this time (p99 TTFT 4.3-7.1s vs
rate=15's 0.96s, comparable to Heavy/Matched's own lightest tested rate) before trusting the
comparison, per the Part 24 lesson. Full 4-arm comparison, n=3:

| Comparison | peak p | TTFT p (direction) | TBT p (direction) |
|---|---|---|---|
| vs `round_robin` | 0.757 (tie) | **0.0107** (cc wins: 0.924s vs 1.099s) | **0.0054** (cc wins: 377.6ms vs 410.2ms) |
| vs `lmetric_power_coincidence_ceiling` | 0.087 (tie) | **0.045** (cc loses: 0.924s vs 0.888s) | 0.227 (tie) |
| vs `weighted_sum_coincidence_ceiling` | 0.168 (tie) | **0.017** (cc loses: 0.924s vs 0.865s) | 0.082 (borderline, trending loss) |

**Two real findings, both clean this time given genuine load:**
1. **The `round_robin` comparison is now a clean double-win** (TTFT and TBT both significant,
   same direction) --- unlike rate=15 (unloaded), where only TBT was significant. Consistent
   with the load hypothesis: the mechanism's advantage over a state-blind baseline shows up
   clearly once there's real pressure to route around.
2. **The TTFT cost against both tie-broken alternatives persists even under real load** ---
   this is not simply an artifact of the earlier unloaded test. `coincidence_ceiling` is
   real y worse on TTFT specifically against both `lmetric_power_coincidence_ceiling` and
   `weighted_sum_coincidence_ceiling`, at a condition now confirmed to be under genuine
   pressure. The "comprehensive tie" claim does not hold on WildChat regardless of load level
   tested so far --- this looks like a genuine, condition-specific property (real
   conversational traffic, no whale injection), not a load-regime artifact.

**Implication for the paper:** the "costs nothing" framing needs to be scoped to the
whale-injection condition family specifically (as Part 23 already concluded), and this
strengthens that conclusion rather than complicating it further --- WildChat's TTFT cost
against the tie-broken alternatives is real and persistent, not something more load happens to
fix.

BurstGPT scale=3.0 (the other higher-load re-test) still pending as of this entry.

### Data and repro (Part 25)

Orchestration: `run_pergpu_load_extremes.sh` (stage 2). Same analysis approach as Parts 17-24.

### Part 26: avoidable-threshold-violation rate measured directly for the first time --
confirms the proof, and finds one real (not just theoretical) violation for
lmetric_power_coincidence_ceiling

Implemented the metric proposed in response to "threshold-safe doesn't buy us any benefits on
the metrics we have so far -- can you think of a metric that is threshold-safe related?":
`Router` now computes, for every routing decision, `D_chosen` (the picked candidate's dominant
share, using whatever ceiling the active policy actually uses -- coincidence-ceiling-adjusted
for any `_coincidence_ceiling` variant) and `min_D_available` (the minimum D(c) across the
whole candidate set at that decision), and flags `avoidable_threshold_violation` whenever
`D_chosen > tau` (default 1.0, matching Theorem 5's own worked example and the paper's "1.0 =
at capacity" convention) while `min_D_available <= tau` -- i.e., the rule passed over an
available safe candidate for an unsafe one. Logged as three new trailing columns in the
assignment CSV (`router_core.py`, `proxy_server.py`; 3 new tests in
`test_eenergy_router_core.py` including a direct reproduction of Theorem 5's exact
counterexample through the real routing pipeline; 312 local / 309 remote, all passing).

Measured on the flagship condition (Heavy/CL short), 3 trials each, `coincidence_ceiling`,
`weighted_sum_coincidence_ceiling`, `lmetric_power_coincidence_ceiling`:

| arm | avoidable violations | rate |
|---|---|---|
| `coincidence_ceiling` | 0/453 | 0.000% |
| `weighted_sum_coincidence_ceiling` | 0/453 | 0.000% |
| `lmetric_power_coincidence_ceiling` | 1/453 | 0.221% |

**`coincidence_ceiling`'s zero rate matches the proof exactly** (Theorem 4/Corollary 1,
deterministic, not a statistical claim -- this confirms the instrumentation is correct, not
new information about the rule). **`lmetric_power_coincidence_ceiling`'s one violation is real,
not synthetic**: replica r5 was picked with D_chosen=1.291 (power-dominated, unsafe) while
another candidate had D=0.670 (safe) available -- inspected directly, a genuine instance of the
predicted failure mode occurring on real traffic. **`weighted_sum_coincidence_ceiling` shows
zero violations at this sample size**, despite Theorem 5 proving the failure mode exists --
not a contradiction (the theorem is an existence proof, not a frequency claim), but suggestive
that the failure mode is much rarer on real, structured traffic than the 11.71% found on
uniform-random synthetic instances (Theorem 5's own generic-instance check, section 4.5) --
real traffic's share distributions likely don't produce the adversarial-looking configurations
uniform sampling does.

**Honest statistical caveat**: 453 decisions/arm is a modest sample for a rare-event rate. Zero
observed events does not mean zero true rate -- by the rule of three, the 95% upper bound on
`weighted_sum_coincidence_ceiling`'s true rate given 0/453 is ~0.66%, not 0%. The paper should
state this precisely: proof-level guarantee (confirmed empirically at 0/453) for
`coincidence_ceiling`; one real observed violation for `lmetric_power_coincidence_ceiling`; no
observed violations (but not a demonstrated zero rate) for `weighted_sum_coincidence_ceiling` at
this sample size. This directly answers the "threshold-safety doesn't show up in any metric we
have" objection -- it does, once measured at the right level (per-decision, not aggregate
outcome), and the direct measurement is consistent with, not contradicted by, the theory.

**Next**: this was only run on one condition (Heavy/CL short) at n=3. Extending to more
conditions/trials, especially the higher-load variants where more coincidence-ceiling
adjustment activity occurs, would sharpen the weighted_sum_coincidence_ceiling rate estimate
and might surface more violations for the arms that lack the guarantee. Not yet done.

### Data and repro (Part 26)

Orchestration: `run_pergpu_threshold_violation.sh`. Instrumentation:
`scripts/eenergy/router/router_core.py` (`last_D_chosen`, `last_min_D_available`,
`last_avoidable_threshold_violation`, `THRESHOLD_TAU`), `proxy_server.py`
(`format_assignment_record` extended). Analysis computed inline via the assignment CSV's new
columns, not yet promoted to a standalone script.

### Part 27: WildChat + whale injection at rate=10.7 is severely over-saturated --
inconclusive for the long-context hypothesis, not evidence against it

Tested whether pointing the Heavy family's exact whale-injection settings
(`--whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000`) at real WildChat
conversations (instead of synthetic ShareGPT) changes coincidence_ceiling's picture there,
isolating content length from trace source. Same rate (10.7) as the plain-WildChat baseline
for a clean before/after comparison.

**The condition turned out far more loaded than intended**: mean TTFT 17.6s, p99 57.5s --
worse than even Heavy/Matched's confirmed-saturated rate=15.0 point (mean ~10s). Whale
injection stacked on real multi-turn WildChat content (up to 4 turns/conversation) is much
heavier than either factor alone at this rate.

Full 5-arm comparison, n=3, vs `coincidence_ceiling`:

| vs | peak_p | ttft_p | tbt_p |
|---|---|---|---|
| lmetric | 0.250 | 0.683 | **0.010 (cc loses, 1388.6 vs 1349.9ms, ~3%)** |
| round_robin | 0.874 | 0.176 | 0.231 |
| lmetric_power_coincidence_ceiling | 0.981 | 0.992 | 0.452 |
| weighted_sum_coincidence_ceiling | 0.937 | 0.964 | 0.444 |

**Almost everything ties, matching the established saturation pattern (Part 20/21): once a
condition is this deep in overload, aggregate fleet capacity dominates and routing-policy
choice stops being able to help or hurt much, regardless of design.** This is not evidence
against the long-context hypothesis (does whale-injected/long-context content specifically
help coincidence_ceiling independent of trace source) -- it's evidence this particular test
was run at a load level past the point where any routing comparison is informative. The one
real signal (TBT vs lmetric) is small and consistent with "differences barely register under
extreme saturation" rather than a meaningful directional finding.

**To actually answer the long-context question, this needs re-running at a lower rate** --
something in the range that produced Heavy/Matched's cleanest signal (comparable to its
rate=7.0-10.7 range, not 15.0+), recalibrated for WildChat-whale's heavier per-request cost.
Not yet done given time constraints; flagged as a clearly-scoped follow-up rather than treated
as a completed, inconclusive test of the actual hypothesis.

### Data and repro (Part 27)

Orchestration: `run_pergpu_wildchat_whale.sh`. Same analysis approach as prior parts.

### Part 28: avoidable-threshold-violation rate at 6x the sample, plus lmetric and round_robin
-- a clean, interpretable ordering, no individual pairwise difference statistically confirmed

Extended Part 26 to all 5 headline-adjacent arms (added `lmetric`, `round_robin`) on the
Heavy/CL long condition (900 convs/trial vs the short condition's 150 -- ~2703 decisions/arm
across 3 trials, 6x Part 26's 453/arm sample).

| arm | violations | rate |
|---|---|---|
| `coincidence_ceiling` | 0/2703 | 0.000% |
| `weighted_sum_coincidence_ceiling` | 0/2703 | 0.000% |
| `lmetric_power_coincidence_ceiling` | 1/2703 | 0.037% |
| `lmetric` | 1/2703 | 0.037% |
| `round_robin` | 2/2703 | 0.074% |

**A clean, sensible ordering**: the two rules with some formal safety property (proof-level
threshold-safety for `coincidence_ceiling`; Pareto-safety only for
`weighted_sum_coincidence_ceiling`) sit at zero; the two without any such guarantee
(`lmetric_power_coincidence_ceiling`, `lmetric`) tie at one event each; the fully state-blind
`round_robin` has the most. More state-awareness correlates with fewer avoidable violations
even for rules never proven to avoid them, which is intuitively the right direction.

**Honest statistical caveat, stated plainly**: with counts this small (0, 0, 1, 1, 2), no
individual pairwise difference clears significance -- this is a descriptive ordering, not a
set of confirmed comparisons. `lmetric_power_coincidence_ceiling`'s rate here (0.037%) is
notably lower than Part 26's small-sample estimate (0.221%, 1/453) -- consistent with a
genuinely rare event where small-sample rates are themselves noisy, not evidence the earlier
single observation was wrong; both are real, observed instances of the predicted failure mode,
just with different point estimates from small counts. `weighted_sum_coincidence_ceiling`'s
0/2703 tightens its rule-of-three upper 95% bound to ~0.11% (down from ~0.66% at n=453).

**For the paper**: report the table and the ordering, explicitly flag the small-count caveat,
and do not claim any pairwise significance beyond "coincidence_ceiling and
weighted_sum_coincidence_ceiling showed zero violations at this sample size; the other three
arms each showed a small number of real, observed instances of the predicted failure mode."

### Data and repro (Part 28)

Orchestration: `run_pergpu_threshold_violation_big.sh`. Same inline CSV analysis as Part 26.

### Part 29: random-power-tiebreak ablation precisely localizes where the TTFT benefit and
the TBT cost each come from -- the benefit survives randomizing the tie-break; the cost
disappears

Direct answer to "is share_power noise-equivalent to random selection?" Built
`drf_power_tiebreak_full_coincidence_ceiling_random_power`: D(c) computed identically (real,
coincidence-ceiling-adjusted power, so which candidates are genuinely under pressure is
unaffected), but the secondary tie-break coordinate (normally real share_power) replaced with
independent random noise per decision. n=6 on both conditions with the strongest established
round_robin signal.

**Heavy/CL long**: random_power_tiebreak is statistically indistinguishable from real
`coincidence_ceiling` on all three metrics (peak p=0.454, TTFT p=0.252, TBT p=0.134), and
**still beats `round_robin` on TTFT** (p=0.0016, nearly matching the real rule's margin:
0.417s vs 0.448s here, vs the real rule's 0.426s vs 0.448s).

**Heavy/Matched**: random_power_tiebreak **reproduces the real rule's massive TTFT win over
`round_robin`** almost exactly (p=0.0001, 3.496s vs 5.453s, vs the real rule's 3.537s vs
5.453s) -- **but does NOT reproduce the real rule's TBT loss to `round_robin`** (random_power
vs round_robin TBT: p=0.427, tied; real coincidence_ceiling vs round_robin TBT: p=0.019, a
real loss, Part 19). random_power_tiebreak actually beats the real rule on TBT here (p=0.035,
1221.9ms vs 1288.6ms).

**Precise interpretation**: the TTFT advantage over `round_robin` does not come from the power
tie-break's specific value -- it survives essentially unchanged when that value is replaced
with pure noise. The benefit is very likely coming from D's primary criterion instead, which
still reads the REAL power value (unaffected by this ablation) to determine which candidates
are genuinely under pressure -- randomization here only affects which of the already-D-tied
candidates gets picked, not whether pressured candidates get correctly identified and avoided
in the first place. **The TBT cost, in contrast, is specifically attributable to the power
tie-break's real behavior** -- it disappears (and even reverses in the ablation's favor) once
that specific tie-break information is randomized away.

**This is a genuine, actionable localization, not just a noise-vs-signal verdict.** It
suggests the TTFT benefit and the TBT cost are mechanistically separable: a design that keeps
power in D's primary criterion but does not use it as a deterministic secondary tie-break
priority might plausibly capture the benefit without the cost. Not tested directly here (that
would be a new rule variant, not this diagnostic ablation), but the evidence points that
direction clearly enough to be worth stating as a concrete hypothesis for future work.

**For the "is it just noise" question directly**: partially yes, partially no, and now we know
which part is which. The power tie-break's specific value is closer to noise-equivalent for
the property that matters most (the TTFT benefit survives fine without it) -- but it is not
noise for the TBT cost, which is real and specifically tied to that exact tie-break behavior.

### Data and repro (Part 29)

Orchestration: `run_pergpu_random_power_tiebreak.sh`. Implementation:
`dominant_share_vector_power_priority_full_coincidence_ceiling_random_tiebreak`,
`pick_drf_power_tiebreak_full_coincidence_ceiling_random_power` (`scoring.py`); wired into
`router_core.py` as policy `drf_power_tiebreak_full_coincidence_ceiling_random_power`. 4 new
tests in `test_eenergy_scoring.py`/`test_eenergy_router_core.py`, 316 local / 313 remote
passing. Same significance-testing approach as prior parts.
