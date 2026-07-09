# Staggered Closed-Loop — the c=15 win is a thundering-herd artifact

**Date:** 2026-07-09
**Question:** The multi-turn closed-loop win (c=15: T3 −45%, T4 −47% TTFT p95 for
combined vs baseline) was measured with all 15 conversations starting at t=0.
Is that benefit real, or an artifact of the synchronized "thundering-herd" start?

## Setup

2×2, single session: {herd, staggered} × {baseline, combined}, concurrency=15,
200 convs × 4 turns, max_tokens=128, max-num-seqs=32, max-num-batched-tokens=2048.
- **herd**: all 15 workers fire at t=0 (original setup).
- **staggered**: 15 workers pull from a shared queue (steady-state c=15 preserved),
  but initial starts spread over a 30s window — removes the t=0 spike. New
  `--stagger-window` mode in `replay_sharegpt.py`; runner `scripts/run_staggered_cl.sh`.
- **combined** = PREFIX_REORDER=1 + DYNAMIC_CHUNK=1 (depth-mode hysteresis) +
  AGING_ALPHA=0.3 — the original win config.
- TPOT here uses the new real-token definition (~20ms), not the old word proxy;
  irrelevant to this question (TTFT/E2EL carry it).

## Results (TTFT p95)

```
Turn    HERD baseline  HERD combined     STAG baseline  STAG combined
T1        228.2ms       228.8 (+0%)        86.9ms        92.2 (+6%)
T2        133.0ms        97.9 (−26%)       67.9ms        79.2 (+17%)
T3        152.9ms        87.2 (−43%)       68.9ms        61.4 (−11%)
T4        131.4ms        77.0 (−41%)       67.4ms        68.9 (+2%)
```

## Findings

1. **Herd reproduces the original win** (T3 −43%, T4 −41%), confirming the setup
   is faithful to the earlier multi-turn result.

2. **The win collapses under staggering.** T3/T4 go from −43%/−41% to −11%/+2%;
   T2 flips from −26% to +17%. Almost the entire benefit was herd-dependent.

3. **The herd inflated the baseline.** Baseline T1 is 228ms (herd) vs 87ms
   (staggered). The synchronized t=0 start piles all 15 first-turns into one queue
   spike, manufacturing a fat baseline tail that the mechanism then "recovers."

4. **The killer comparison:** staggered *baseline* T4 (67ms) beats herd *combined*
   T4 (77ms). Simply not having the herd beats the mechanism running inside it —
   the policy was compensating for a benchmark artifact.

## Conclusion

The −45%/−47% closed-loop win was **substantially a thundering-herd artifact**.
Under staggered (more realistic) arrivals the mechanism is at best marginal
(T3 −11%) and neutral-to-worse elsewhere. Combined with the open-loop null result
(`2026-07-09-openloop-chunk-probe.md`), the strategy has **no robust win in any
realistic regime** — neither open-loop Poisson nor staggered closed-loop. This
sharpens the paper's framing: the apparent benefit is an artifact of
synchronized-start benchmarking; the contribution is the characterization of
when/why prefix-aware chunking + reordering appear to help and don't.

**Caveat:** single trial per mode. The direction is strong and the baseline-
inflation mechanism (228→87ms) is structural rather than noise, so it will
reproduce; exact deltas would firm up with 2–3 trials.

## Repro

```
STAGGER=0  bash scripts/run_staggered_cl.sh   # herd
STAGGER=30 bash scripts/run_staggered_cl.sh   # staggered
python3 src/analyze_ablation.py \
  'baseline:logs/<date>-stagcl-base_<mode>_t1.jsonl' \
  'combined:logs/<date>-stagcl-comb_<mode>_t1.jsonl'
```
