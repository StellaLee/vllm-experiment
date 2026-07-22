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
metrics while costing almost nothing in throughput, with zero custom controller code.

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

---

## Results table

Same phase-schedule workload, all arms compared under identical `SLO_TBT_MS=500`:

| arm | S:TTFTmean | S:TTFTp95 | W:P99 TBT | W:max TBT | W:goodput% |
|---|---|---|---|---|---|
| mono (16384) | 250 | 566 | 93.4 | **3023.4** | 78.6% |
| static-512 | 271 | 643 | 127.3 | 261.4 | 100.0% |
| static-2048 | 249 | 549 | 380.8 | 863.1 | 98.1% |
| lengthgate (custom, post depth==0 fix) | 249 | 550 | 125.6 | 1667.4 | 81.2% |
| **16384 + long-prefill-token-threshold=512** | 263 | 592 | 195.8 | **490.4** | **100.0%** |

n=1080 requests/phase-type, n≈37.6–38.2k pooled TBT samples per arm.

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

## Next steps

1. **Sweep `long_prefill_token_threshold`** (256 / 1024 / 2048, same phase-schedule, mono budget)
   to map this knob's own throughput-vs-tail Pareto frontier, the same way static budget was
   swept. Expect: lower threshold → better max/goodput, worse S:TTFT (whale itself pays more
   slicing overhead); a sweet spot likely between 512 and 2048.
2. Decide whether `lengthgate`'s bug 2 is worth fixing directly (e.g. reserve headroom for
   non-whale admissions during protect mode) now that a structurally simpler native alternative
   exists, or whether the custom controller should be retired from the paper narrative in favor of
   the native-flag result.
3. Consider whether combining a per-request threshold with a *non-mono* step budget (e.g.
   2048 + lpt512) adds anything, or whether it's redundant once the per-request cap is in place.

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
  `rerun_lpt_arm.sh` (5th arm, native flag → `logs/lgate_ANALYSIS_v3.txt`).
- Design spec: `docs/superpowers/specs/2026-07-22-lengthgate-dynamic-chunk-design.md`. Plan:
  `docs/superpowers/plans/2026-07-22-lengthgate-dynamic-chunk.md`.
- Data: `logs/2026-07-22-lgate-b*-t1.jsonl` (+ `-chunktrace.csv` for lengthgate arms),
  `logs/lgate_ANALYSIS_v3.txt` (final 5-arm table).
