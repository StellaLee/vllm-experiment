# Tail-keyed (p99) SLO controller — the fix engages, but still cannot beat the static frontier

**Date:** 2026-07-13
**Question:** The sensitive-regime SLO controller is blind because it drives the chunk budget
off an EMA of the **mean** per-step latency, which the frequent cheap decode steps dominate
(§6.3; findings/2026-07-10-sensitive-regime.md). Does keying off the **tail** (p99) of a
windowed step-latency instead fix it — and if so, does the fixed controller finally beat the
static frontier?

## Setup
Same sensitive regime as the 07-10 probe: single 4090, Qwen2.5-Coder-7B, uniform 12000-char
(~2300-tok, uncacheable) padding, staggered closed-loop c=10, 80 convs × 3 turns,
max_tokens=512, reorder OFF. New controller mode `CHUNK_MODE=slotail`
(`scripts/hotpatch_slo_tail.py`): budget driven by the p99 of a 128-step latency window vs a
50 ms SLO. Four arms: static8192 (mono ref), static512 (PS frontier), slo (EMA-of-mean,
blind), slotail (p99, the fix). **Single trial.**

## Result
| arm | TPOT mean | p99 | max | TTFT mean | budget lo/med/hi (%ceil) |
|---|---|---|---|---|---|
| static8192 | 32.0 | 50.9 | 432.6 | 463 | static |
| static512  | 31.0 | **44.1** | 506.8 | 609 | static |
| slo        | 33.6 | 51.9 | **838.5** | 475 | 512/7424/8192 (49%) |
| slotail    | 31.5 | 52.9 | 553.6 | 610 | 512/**512**/8192 (8%) |

## Findings
1. **The fix engages — diagnosis confirmed.** slotail's p99 signal drives the budget to the
   floor (median 512, 8% at ceiling) where the blind slo stays high (median 7424, 49%). The
   mean signal *was* the bug: a tail signal makes prefill stalls visible and the controller
   reacts.
2. **Engaging caps the extreme max** (slotail max 553 < slo max 838).
3. **But the fixed controller still does NOT beat the static frontier.** slotail collapses onto
   ~static512's operating point (budget→512 → pays static512's +32% TTFT, 610 vs 463) yet its
   TPOT p99 (52.9) is *worse* than static512's (44.1): its residual AIMD excursions upward
   re-expose it to stalls. It reaches the frontier's costs without its benefit.

## Conclusion (sharpens §6)
Converts "our controller is blind" into a stronger claim: *we built the tail-keyed fix the
theory prescribes; it engages exactly as intended (confirming the mean-signal was the bug),
but still cannot beat a static budget, and adaptation's own excursions hurt the tail — so the
ceiling is the **mechanism** (work conservation), not the controller signal.* And the static
frontier it lands near is itself a poor trade (+32% TTFT for no robust TPOT-tail gain).

**Caveat:** single-trial; TPOT p99/max are noisy (slo sat 49% at ceiling here vs 97% in the
07-10 run). The engagement result (budget 512 vs 7424) is a large clear effect; the latency
deltas need 3-trial replication before quoting exact numbers (batch with R10 sensitive-regime
replication). Logs: `logs/2026-07-13-sp99-*_t1.jsonl`, budget from `-sp99-{slo,slotail}-server.log`.
