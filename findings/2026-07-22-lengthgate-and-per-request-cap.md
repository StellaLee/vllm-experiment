# Whale-fraction non-stationarity: lengthgate controller, its bugs, and the native fix

**Date:** 2026-07-22
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0 (V1), Qwen2.5-Coder-14B-Instruct, TP=2
**Workload:** NEW non-stationary axis — open-loop Poisson arrivals with a **phase-scheduled
whale-fraction** (`--phase-schedule`), not concurrency. Locked schedule: `6:0.0@60,1.0:0.2@60`
(Phase S: rate=6/s, 0% whales, 60s; Phase W: rate=1/s, 20% whales, 60s), `--duration 360`
(3 full S↔W cycles), single-turn, `max-tokens=256`, whale chars 44–50k, pad-seed 1001.
**Headline:** Built a custom controller (`lengthgate`) to shrink the step budget only when a long
prompt is prefilling. It has two real bugs (both root-caused, one fixed). Then discovered vLLM's
**native `--long-prefill-token-threshold`** flag does the same job structurally better — it caps
the *offending request*, not the *whole step* — and it pareto-dominates static-2048 on both tail
metrics while costing almost nothing in throughput, with zero custom controller code. A follow-up
threshold×budget grid then found the win is **not monotonic in threshold value** (256 is worse
than 512, not better) and that **shrinking the step budget alongside the threshold adds nothing**
— which points at the next real experiment: an *adaptive* threshold on a *static* budget, reusing
lengthgate's own length-gate signal but redirected at the per-request cap instead of the step
budget.

Also confirmed along the way: `--max-num-partial-prefills` / `--max-long-partial-prefills` are
**dead config in this vLLM build** — real CLI flags, validated and logged at startup, but never
read anywhere in `vllm/v1/`'s actual scheduling logic. Only `long_prefill_token_threshold` does
anything.

**Update — the adaptive threshold was built, found broken, fixed, and now wins cleanly.** The
first version (`adaptivelpt`) had a missing-hysteresis bug: the same 4096-token bar governed both
entering and exiting protect mode, so it released a whale while ~4200-4500 tokens were still left,
dumping that remainder uncapped and making tail protection *worse* than doing nothing adaptive at
all (W:goodput 75.5%, below even mono's 78.6%). Fixed with an asymmetric entry/exit gate
(`adaptivelpt2`): **S:TTFT=248ms** (ties mono's 250ms) **and** **W:max=489.2ms / W:goodput=100%**
(matches the always-on static `16384lpt512`'s 490.4ms/100%) — simultaneously, for the first time
in this whole line of work. A separate finding surfaced along the way via a whale-only TPOT
breakdown: static-2048-class arms (`static-2048`, `16384lpt2048`) suffer a **whale-on-whale freeze**
effect that has nothing to do with the adaptive-threshold story — see below.

---

## TL;DR

- **The axis that matters is whale-fraction over time, not concurrency.** Concurrency was already
  shown (prior sessions) not to move the optimal budget — db/α stay flat regardless of load depth.
  Whale-fraction is a different, real non-stationary driver: Phase S wants a large budget
  (throughput), Phase W wants a small one (tail protection).
- **The static three-arm comparison (mono/512/2048) reproduces the known tradeoff cleanly** and
  matches the `db + α·B` mechanistic model to within a few percent — this is the valid baseline
  any adaptive mechanism must beat.
- **Our custom `lengthgate` controller (toggle the step-wide budget by prompt length) has two real
  bugs**, both found by direct chunktrace forensics, not guesswork:
  1. **depth==0 loophole** (fixed) — gating protect-mode on `decode_depth>0` let a whale through
     unsliced whenever it landed in a momentary empty-decode step.
  2. **admission-starvation-then-catchup** (NOT fixed) — while protect mode is active, the whale
     (FCFS-first) consumes the entire shrunk budget every step, starving other admissions; when the
     gate flips back to blast, the backlog dumps into one expensive catch-up step.
- **vLLM's native `--long-prefill-token-threshold` avoids both bugs by construction**: it caps only
  the flagged request's own consumption in a step, never the step's overall budget for everyone
  else. Tested at `threshold=512` with the server budget left at mono's 16384 ceiling
  (`16384+lpt512`): **W:max 490ms** (vs mono's 3023ms, −84%; vs static-2048's 863ms, −43%),
  **W:goodput 100%** (ties static-512's perfect score), **S:TTFT 263ms** (+5% over mono's 250ms,
  still beats static-512's 271ms). This pareto-dominates static-2048 on both tail metrics for
  almost no throughput cost, with no controller logic at all.
- **Reframe: the axis that needed tuning was never the step-wide budget — it was the per-request
  cap.** Both hslo (prior finding) and lengthgate manipulate the shared step budget; both hit
  structural problems traceable to that choice (hslo: no load-sensitive signal to react to on this
  hardware; lengthgate: collateral throttling of bystanders). A **static, per-request** cap needs no
  runtime adaptivity at all to get the benefit.
- **The threshold's own value is NOT monotonic and NOT free.** A 4-way sweep at mono budget
  (off/256/512/2048) found **512 pareto-dominates 256** — smaller isn't automatically safer.
  Mechanism (hypothesis, not trace-confirmed): a smaller threshold makes any long-ish request stay
  mid-prefill for more steps, raising the odds several such requests overlap in one step; the
  threshold bounds each request's *own* slice but never the step's *total* volume, so overlaps
  stack uncapped. The same gap shows up at threshold=2048: `16384lpt2048` has a **worse** tail
  (max=1324.6, gp=82.4%) than plain **static-2048** (max=863.1, gp=98.1%), because static-2048
  bounds the whole step regardless of how many requests share it, while the threshold doesn't.
- **Shrinking the step budget alongside the threshold buys nothing.** `2048lpt512`
  (S:TTFT=263, W:max=454.0, gp=100.0%) is statistically indistinguishable from `16384lpt512`
  (S:TTFT=263, W:max=490.4, gp=100.0%) — once the per-request cap is active, the step budget can
  just stay at its most generous (mono) setting with no further tuning needed.
- **A static threshold pays a throughput tax even when it isn't needed.** `16384lpt512`'s
  S:TTFT (263ms, Phase S, 0% whales) is +5% over mono's 250ms — pure overhead from ordinary
  non-whale prompts occasionally exceeding 512 tokens and getting sliced for no reason during a
  phase with nothing to protect against. This is the opening for an *adaptive* threshold.

---

## Results table

Same phase-schedule workload, all arms compared under identical `SLO_TBT_MS=500`. `budget` =
`--max-num-batched-tokens`, `thr` = `--long-prefill-token-threshold` (`off` = flag unset):

| arm | budget | thr | S:TTFTmean | S:TTFTp95 | W:P99 TBT | W:max TBT | W:goodput% |
|---|---|---|---|---|---|---|---|
| mono | 16384 | off | 250 | 566 | 93.4 | **3023.4** | 78.6% |
| static-512 | 512 | off | 271 | 643 | 127.3 | 261.4 | 100.0% |
| static-2048 | 2048 | off | 249 | 549 | 380.8 | 863.1 | 98.1% |
| lengthgate (custom, post depth==0 fix) | 16384† | off | 249 | 550 | 125.6 | 1667.4 | 81.2% |
| 16384lpt256 | 16384 | 256 | 295 | 680 | 157.9 | 1713.1 | 96.8% |
| **16384lpt512** | 16384 | **512** | 263 | 592 | 195.8 | **490.4** | **100.0%** |
| 16384lpt2048 | 16384 | 2048 | 250 | 562 | 392.8 | 1324.6 | 82.4% |
| 2048lpt512 | 2048 | 512 | 263 | 593 | 215.3 | 454.0 | 100.0% |
| adaptivelpt (pre-fix, buggy) | 16384† | 0↔512 | 248 | 553 | 219.2 | 1126.4 | 75.5% |
| **adaptivelpt2 (hysteresis fix)** | 16384† | 0↔512 | **248** | 566 | 208.6 | **489.2** | **100.0%** |

†lengthgate's *server-side* budget is 16384; the controller toggles the *effective per-step*
budget between 16384 (blast) and 512 (protect) at runtime. adaptivelpt/adaptivelpt2's server-side
budget is likewise static at 16384; the controller toggles `long_prefill_token_threshold` between
0 (off) and 512 (protect), never the budget itself.

n=1080 requests/phase-type, n≈37.6–38.2k pooled TBT samples per arm. Single trial per arm.

### Whale-only breakdown (TTFT and TPOT, isolating actual whale requests, n=10/arm)

The pooled `W:` columns above mix whales (20% of W-phase arrivals) with the 80% ordinary short
requests that also arrive during the W-phase window — diluting the whale-specific signal,
especially for TTFT and TPOT (mean-of-decode-stream metrics, as opposed to pooled per-token TBT).
Isolating records by `pad_chars > 20000`:

| arm | whale TTFT mean/p95 | whale TPOT mean/p95 |
|---|---|---|
| mono | **3252 / 4879** | 48.53 / 92.28 |
| static-2048 | 3483 / 5272 | **73.83 / 380.09** |
| 16384lpt2048 | 3759 / 5188 | **78.83 / 437.18** |
| static-512 | 4075 / 6150 | 51.66 / 110.57 |
| lengthgate | 4058 / 6535 | 47.24 / 110.83 |
| adaptivelpt (pre-fix) | 4458 / 6172 | 58.01 / 204.16 |
| 16384lpt512 | 4624 / 7158 | 53.50 / 128.25 |
| **adaptivelpt2** | 4617 / 7106 | 53.30 / 129.01 |
| 2048lpt512 | 4715 / 6646 | 61.64 / 200.15 |
| 16384lpt256 | 5710 / 8787 | 60.76 / 153.38 |

n=10 whales/arm is thin — p95 and p99 collapse to the same index at this sample size (verified:
identical values both ways), so don't over-read the exact ranking among the middle cluster
(lengthgate/512/16384lpt512/adaptivelpt2 are indistinguishable at this n). The large, consistent
gap between {static-2048, 16384lpt2048} and everyone else is the real signal — see below.
`adaptivelpt2`'s near-exact match to `16384lpt512` (4617/7106 vs 4624/7158 TTFT; 53.30/129.01 vs
53.50/128.25 TPOT) confirms the hysteresis fix: once triggered, a whale's experience under
`adaptivelpt2` is now indistinguishable from being permanently capped — the only difference from
`16384lpt512` is that `adaptivelpt2` relaxes during calm periods instead of staying capped forever.

### New finding: whale-on-whale freeze exposure (independent of the adaptive-threshold story)

`static-2048` and `16384lpt2048` have by far the worst whale-only TPOT tail (p95 380-437ms vs
92-204ms everywhere else) — a mechanism distinct from the pooled `W:max` freeze (which measures a
whale's prefill hitting *other* decoders). This measures what happens to a whale *after* it
finishes prefilling and starts decoding: it's just as exposed to a *later* whale's full-size
admission as any other decoder is. Neither `static-2048` nor `16384lpt2048` ever shrinks below
2048 tokens/step regardless of how many whales are queued, so each such collision costs
`db + α·2048 ≈ 380-435ms` (matching the established mechanistic model) — and a plausible reason
this specific pair is worst: capping to 2048 (rather than mono's one-shot or 512's much smaller
slices) also makes each whale **linger longer** in the system (more steps needed per whale),
increasing the window during which a whale-turned-decoder can collide with a subsequent whale's
prefill. Every arm that shrinks further under whale pressure (mono, 512, lengthgate, all
`lpt512`-class arms) avoids this because either the freeze is too rare to matter (mono: whales
resolve in ~1 step, low residency) or too small to matter (512/lpt512-class: capped admissions).
This is a standalone result, applicable to any arm holding a fixed ~2048 ceiling, independent of
whether the adaptive-threshold controller works or not — worth its own line in the paper.

## Calibration (before the 4-arm run)

Rate-calibrating the phase schedule took 4 bisection rounds, not 1:

- rate=12: catastrophically oversaturated (TTFT 43–72s).
- rate=2.5: zero prefill-bound separation between static-512 and static-2048 in Phase S.
- rate=9: widened absolute TTFT ~20× but the *relative* gap barely moved (+8.8%→+10.3%), and
  **contaminated the W-phase reading** — W:TTFT ballooned from ~700ms to ~3000ms even though
  `rate_W` was unchanged, diagnosed as S-phase backlog spilling across the phase boundary.
- **rate=6 locked in**: uncontaminated, Phase S separation modest (~9%) but real. Accepted the
  modest gap rather than continuing to chase a bigger one — a bigger gap risked reintroducing
  cross-phase contamination.

## The lengthgate controller and its two bugs

`_step_lengthgate` (installed via `scripts/hotpatch_lengthgate.py`): blast a large budget
(16384) by default; shrink to `protect` (512) whenever `pf_remaining` (max prefill-tokens-still-
needed over running prefill-chunks + waiting requests) exceeds `threshold` (4096).

**Bug 1 — depth==0 loophole (found and fixed before presenting results).** The first version
gated protect-mode on `decode_depth > 0` (copying hslo's philosophy, where protecting only matters
if someone is actively decoding). But protecting the whale costs lengthgate *nothing* even when
nobody is decoding — the design intent was length-driven, not decode-pressure-driven. The loophole:
a whale landing in a momentary `decode_depth==0` step got blasted unsliced, freezing anyone who
arrived *during* that multi-second window. Confirmed via chunktrace: exactly 2/2 "blast with
pf_remaining>4096" events had `decode_depth==0` (out of 16 total depth==0 steps in 11336). Fixed by
removing the `and decode_depth > 0` clause — gate on length alone. This was self-caught, not
user-flagged, before the first results were presented as a clean win.

**Bug 2 — admission-starvation-then-catchup (found, NOT fixed).** After fixing bug 1, `W:max`
barely improved (1667ms, still far above static-512's 261ms or even static-2048's 863ms). Root
cause, found by taking the single worst TBT event (1667.36ms) and cross-referencing its wall-clock
time against the chunktrace: a genuine **1.54-second gap in logged step timestamps** immediately
after a protect-mode whale-slicing sequence ended. Mechanism — while protect mode (budget=512) is
active for ~11 consecutive steps, the whale, being first in the FCFS queue and needing far more
than 512 tokens/step, consumes essentially the *entire* budget every step. Any short request that
arrives during that ~1.4s window gets zero admission — it queues untouched. When the gate flips
back to blast (whale's remaining length drops below 4096), every backlogged admission dumps into
one expensive catch-up step. This is a structural consequence of toggling a **step-wide** budget:
shrinking the whole step to protect against one whale collaterally starves every bystander waiting
in that same step. Not fixed in lengthgate's code — instead the conversation pivoted to testing
whether a **per-request** cap avoids the mechanism by construction.

## The native fix: `--long-prefill-token-threshold`

Discovered while re-reading the scheduler for the starvation bug: vLLM already ships a flag that
caps any *individual* request's per-step token consumption to a threshold if its own need exceeds
it — applied at both the running-loop and waiting-loop admission sites
(`if 0 < threshold < num_new_tokens: num_new_tokens = threshold`). Critically, this does **not**
reduce the step's overall `token_budget` — only the flagged request's own slice. Other admissions
in the same step retain full access to whatever budget remains.

Tested as a 5th arm: server budget left at mono's ceiling (`--max-num-batched-tokens 16384`),
`--long-prefill-token-threshold 512`, `DYNAMIC_CHUNK=0` (no custom controller — the flag alone).
Result (see table above): pareto-dominates static-2048 on both W:max (−43%) and W:goodput (+1.9pp)
for a throughput cost of +5.6% S:TTFT vs mono (and actually *beats* static-512 on S:TTFT). No
controller, no bugs, no chunktrace forensics needed — a single launch-time flag.

## Why the reframe matters (budget vs. per-request cap)

Both `lengthgate` and, before it, `hslo` (see
[2026-07-22-hslo-controller](2026-07-22-hslo-controller.md)) manipulate the **step-wide
`token_budget`** — the single number shared by every request admitted in one scheduler iteration.
That was the wrong axis for two different reasons on this hardware:

- **hslo's problem**: nothing to react to. decode-cost (`db`) stays flat regardless of load depth
  on this hardware (KV-capped shallow decode batch), so a controller that reacts to decode pressure
  has no live signal — it just reproduces static-2048 by hand.
- **lengthgate's problem**: even with a real signal (prompt length), shrinking the *whole step's*
  budget to protect against one long request collaterally throttles every other admission sharing
  that step. The fix isn't a smarter step-wide toggle — it's not touching the step budget at all.

`long_prefill_token_threshold` targets the actual quantity that matters (how much *this specific*
oversized request can take per step) and leaves the shared step budget at its most generous
setting for everyone else, all the time. That's a **static, always-on** configuration choice, not
a controller — consistent with, not contradicting, the hslo finding that no runtime adaptivity is
needed on this hardware. The adaptivity that matters is in *how the cap is scoped* (per-request vs.
per-step), not in *reacting to load over time*.

**What stays valid from the whole line of work:** the mono/512/2048 step-wide-budget
characterization (still a real, measured tradeoff whenever all requests share one uniform budget);
the whale-fraction/prompt-length hypothesis (directionally correct — length is the right signal);
the freeze mechanism itself (`step_time ≈ db + α·B`, iteration-heartbeat model). **What was
mis-scoped:** using the step-wide budget as the *only* lever for the adaptive/protective half of
the story, when a per-request cap was available and avoids both lengthgate bugs by construction.

## Dead flags: `max-num-partial-prefills` / `max-long-partial-prefills`

Before the grid, checked whether these two flags (docstrings describe exactly the desired
mechanism — "allow shorter prompts to jump the queue in front of longer prompts") could contribute
to the sweep. Grepped the full `vllm/` package on the box: both are real, validated, logged
`SchedulerConfig` fields and real CLI flags (`engine/arg_utils.py:1379,1383`) — but **zero
references anywhere in `vllm/v1/`'s actual scheduling code**. They parse, validate cross-field
constraints, and log a startup message as if active, but nothing in the admission loop ever reads
them. Confirmed dead in vLLM 0.23.0's V1 scheduler; excluded from the grid.

## The threshold×budget grid

Ran 3 more arms (`rerun_lpt_grid.sh`) to map `long_prefill_token_threshold`'s own frontier and test
whether pairing it with a smaller step budget helps: `16384lpt256`, `16384lpt2048`, `2048lpt512`.
Results are in the table above. Two findings, both against the naive expectation:

1. **512 pareto-dominates 256** on every axis that matters (S:TTFT, W:max, W:goodput) — the
   threshold-vs-tail relationship is not "smaller is safer." See the mechanism hypothesis in the
   TL;DR: the threshold bounds one request's slice, never the step's total volume, so a smaller
   threshold (more steps per long request in flight) raises the odds of uncapped overlaps.
2. **`2048lpt512` ≈ `16384lpt512`** — combining a shrunk step budget with the per-request cap adds
   nothing measurable. The cap alone is already doing the work; the step budget can stay at its
   most generous, simplest setting.

## The adaptive threshold: design, bug, fix

Built per `docs/superpowers/specs/2026-07-22-adaptive-lpt-design.md` /
`docs/superpowers/plans/2026-07-22-adaptive-lpt.md`: make `long_prefill_token_threshold` adaptive
at runtime while the step-wide budget stays static at mono (16384), reusing lengthgate's own
`pf_remaining` signal but redirected at the per-request cap instead of `token_budget`.

**Mechanism.** `long_prefill_token_threshold` is read as a plain `scheduler_config` attribute at
both existing use sites, with no caching — so `scripts/hotpatch_adaptive_lpt.py` inserts a single
per-step block right after the unconditional `token_budget = self.max_num_scheduled_tokens` line
(runs regardless of `DYNAMIC_CHUNK`/`ChunkSizeController` state) that mutates
`self.scheduler_config.long_prefill_token_threshold` directly. One insertion point, versus
lengthgate's four — and fully decoupled from the budget, which never moves.

**Bug — missing hysteresis (found via trace, before trusting the first result).** The initial
version used the *same* threshold (4096) to both enter and exit protect mode:
`threshold = 512 if pf_remaining > 4096 else 0`. Result (`adaptivelpt`): S:TTFT recovered
perfectly (248ms, ties mono) but **W:max got worse, not better** (1126.4ms vs the static
`16384lpt512`'s 490.4ms), and **W:goodput dropped to 75.5% — below mono's own 78.6%**. Root cause,
confirmed directly from the trace: all 8 protect→off transitions happened while `pf_remaining` was
still **4226-4556 tokens** (mean 4374) — just under the entry bar. A whale gets released from
protection while it still has ~4200+ tokens left, and that remainder is admitted **fully uncapped**
at the mono budget, producing exactly the kind of freeze the controller exists to prevent. This is
a straight cost transfer from the whale onto its decode neighbors (confirmed by the whale-only
breakdown above: pre-fix `adaptivelpt`'s own whale TTFT, 4458ms, was *better* than the always-on
`16384lpt512`'s 4624ms — the bug let the whale "cheat" at its neighbors' expense).

**Fix — asymmetric entry/exit gate (hysteresis).** Separate the entry condition
(`pf_remaining > ADAPTIVE_LPT_GATE`, unchanged at 4096) from the exit condition (stay in protect
mode until `pf_remaining ≤ ADAPTIVE_LPT_EXIT_GATE`, default **0** — i.e. don't relax until the
request is essentially fully drained). Requires one bit of per-step state
(`getattr(self, "_alpt_in_protect", False)`, the same read-with-default pattern already used
elsewhere in this file for `_ff_last_tokens`). The patcher was made migration-aware (detects the
old single-gate block and replaces it in place) so the box's already-patched scheduler.py could be
upgraded without restoring from a backup. Verified before rerunning: relax-point `pf_remaining` on
the fixed version now sits at **155-460 tokens** (mean 291) — down from 4226-4556 — confirming the
uncapped tail-dump shrank from ~4300 tokens to ~300.

**Result (`adaptivelpt2`, see table above): S:TTFT=248ms (ties mono), W:max=489.2ms /
W:goodput=100% (matches `16384lpt512` almost exactly, both better than the pre-fix buggy version's
1126.4ms/75.5%).** This is the first arm in this whole line of work to hit both halves of the
original hypothesis simultaneously — mono's throughput, the always-on static threshold's tail
protection, with none of its throughput tax.

## Update (2026-07-24): replicated 3× — the tightest result in this whole line of work

Ran 2 more trials of `adaptivelpt2` against the same phase-schedule workload (same seed,
`run_adaptive_lpt_arm2_replicate.sh`, TRIAL=2/3). 0 preemptions, 1245 records each, matching
the original run's scale exactly.

| | trial 1 | trial 2 | trial 3 | mean ± std |
|---|---|---|---|---|
| S:TTFT mean | 248ms | 248ms | 248ms | **248 ± 0** |
| W:TBT p99 | 208.6ms | 207.4ms | 208.7ms | 208.2 ± 0.7 |
| W:TBT max | 489.2ms | 489.4ms | 488.3ms | **489.0 ± 0.6** |
| W:goodput | 100.0% | 100.0% | 100.0% | **100.0 ± 0** |

This is the tightest replication of any result in the project so far — essentially
indistinguishable across 3 independent trials. Against the (single-instance) baselines,
adaptivelpt2 ties mono's S:TTFT (250ms) and matches static-16384lpt512's W:max/goodput
(490.4ms/100%) in every trial, not just once. The simultaneous-win claim — mono's admission
throughput with the best static threshold's tail protection, and none of its cost — is now
a replicated result, not a single demonstration.

**Why so tight, compared to §5.6/§5.8's replications (which showed real trial-to-trial
spread)?** This experiment reuses the same workload seed across trials (isolating
serving/timing noise, same convention as the other two), but the phase-scheduled design here
runs far longer per trial (360s, 1080+159 records) than the closed-loop single-shot designs in
§5.6/§5.8 — more events per trial likely averages out transient serving noise more thoroughly.
Worth keeping in mind when comparing "replication tightness" across the paper's three
replicated findings: it may reflect experiment duration/sample size as much as underlying
mechanism stability.

**Status: this resolves the single-trial caveat for this finding — it was the last remaining
single-trial leg among §5.6/§5.7/§5.8.** A fresh-seed replicate (workload-draw variance, as
opposed to just serving-timing noise) remains open, same as the other two.

## Next steps

1. ~~**Replicate.**~~ **Done 2026-07-24** — see update above. A fresh-seed replicate (not just
   same-seed reruns) remains open.
2. Decide whether `lengthgate` (the step-budget controller, bug 2 unfixed) is worth fixing directly
   now that the threshold-based approach has a clean, working result, or whether it should be
   retired from the paper narrative in favor of `adaptivelpt2`.
3. Consider whether the **whale-on-whale freeze exposure** finding (static-2048/16384lpt2048
   specifically) deserves its own follow-up — e.g. does it hold at higher whale-fraction / higher
   whale arrival rate, where more whales are concurrently in flight?
4. The 256-vs-512 non-monotonicity (threshold's own value) and the `ADAPTIVE_LPT_GATE=4096` entry
   bar were both reused as-is from earlier validated values, not retuned for the adaptive
   controller specifically — a follow-up sweep of the adaptive controller's own gate/protect
   values is still open, lower priority now that the core mechanism works.

## Artifacts

- Timing/schedule generator: `scripts/replay_timing.py` (`parse_phase_schedule`, `phase_type_at`,
  `generate_phase_arrivals`), tests in `tests/test_replay_timing.py`.
- Client: `src/replay_sharegpt.py` `--phase-schedule` flag, `force_whale` threading through
  `sample_pad_len`/`replay_conversation`.
- Controller: `scripts/hotpatch_lengthgate.py` (idempotent 4-edit patcher for
  `ChunkSizeController` / scheduler call site), fix applied by removing the `decode_depth > 0`
  clause from `_step_lengthgate`. Tests in `tests/test_hotpatch_lengthgate.py`.
- Analyzer: `scripts/analyze_lengthgate.py` (`bucket_lengthgate`, per-arm S/W table + verdict),
  tests in `tests/test_analyze_lengthgate.py`.
- Orchestration: `pilot_lengthgate_rates.sh` (rate calibration), `orchestrate_lengthgate.sh`
  (4-arm run), `rerun_lengthgate_arm.sh` (post-fix rerun → `logs/lgate_ANALYSIS_v2.txt`),
  `rerun_lpt_arm.sh` (5th arm, native flag → `logs/lgate_ANALYSIS_v3.txt`), `rerun_lpt_grid.sh`
  (3 more threshold/budget combos → `logs/lgate_ANALYSIS_v4.txt`), `run_adaptive_lpt_arm.sh` /
  `run_adaptive_lpt_arm2.sh` (pre-fix and post-fix adaptive controller arms →
  `logs/lgate_ANALYSIS_v5.txt`, the final 10-arm table).
- Adaptive controller: `scripts/hotpatch_adaptive_lpt.py` (single-anchor, migration-aware patcher
  — detects and upgrades an already-installed pre-hysteresis block in place). Tests in
  `tests/test_hotpatch_adaptive_lpt.py`. Design spec:
  `docs/superpowers/specs/2026-07-22-adaptive-lpt-design.md`. Plan:
  `docs/superpowers/plans/2026-07-22-adaptive-lpt.md`.
- Design spec (lengthgate): `docs/superpowers/specs/2026-07-22-lengthgate-dynamic-chunk-design.md`.
  Plan: `docs/superpowers/plans/2026-07-22-lengthgate-dynamic-chunk.md`.
- Data: `logs/2026-07-2{2,3}-lgate-b*-t1.jsonl` (+ `-chunktrace.csv` for lengthgate/adaptivelpt
  arms), `logs/lgate_ANALYSIS_v5.txt` (final 10-arm table).
