# Decode-length sweep — chunking's TPOT (decode-stall) benefit EMERGES with generation length; the max-tokens=128 null was a short-decode artifact

**Date:** 2026-07-20
**Question:** All prior mono-vs-chunk runs used `max-tokens=128` (short decode), which
suppresses chunking's *main* benefit — decode-stall avoidance (Sarathi). At 128 tokens there
is barely any decode in flight, so "decode never stalls / chunking buys nothing" is nearly
tautological. Does chunking's decode-stall benefit appear under **realistic long generations**?
Sweep `max-tokens ∈ {128, 512, 1024}` on 14B/TP=2, measuring **TPOT** (per-step decode
latency = the stall signal), mono vs chunk.

## Setup
- 8×4090 box, GPUs (0,1), Qwen2.5-Coder-**14B**, TP=2, vLLM 0.23.0, patches gated OFF
  (native static). Cs²=0 (uniform ~1750-tok prefills), single-turn, open-loop.
- **mono** (budget 16384) vs **chunk** (budget 512); **rate calibrated per decode length**
  (capacity drops as decodes lengthen): mt=128→rate 1.0 (knee 1.25), mt=1024→rate 0.64
  (knee 0.8), each ρ≈0.8. NUM=80/point, 3 trials, pooled.

## Result
**TPOT (decode step latency) — the key signal:**
| max-tokens | mono p50/p95 ms | chunk p50/p95 ms | chunk d_p50% | **chunk d_p95%** |
|---|---|---|---|---|
| 128 | 33 / 71 | 35 / 74 | +6.4 | **+4.4** |
| 512 | 32 / 54 | 32 / 45 | +2.3 | **−17.2** |
| 1024 | 29 / 51 | 29 / 40 | +1.9 | **−22.1** |

**TTFT (prefill wait):**
| max-tokens | mono p50/p95 | chunk p50/p95 | chunk d_p50% | chunk d_p95% |
|---|---|---|---|---|
| 128 | 504 / 1615 | 653 / 1432 | +29.6 | −11.4 |
| 512 | 502 / 1324 | 622 / 1208 | +24.0 | −8.8 |
| 1024 | 495 / 1167 | 595 / 1259 | +20.1 | +7.9 |

Saturation sanity: all three sub-saturation (mono TTFT first50→last50 ratio 1.0×).

## Interpretation — the stall-bound lever appears
- **Chunk's TPOT p95 advantage grows monotonically with decode length:** +4.4 (128) →
  **−17.2 (512) → −22.1 (1024)**. At the short-decode setting the benefit is absent (the
  prior null); with realistic long generations chunk clearly **bounds the decode tail**.
  TPOT *p50* is flat (~0) — the median decode step is unaffected; the benefit is in the
  **tail**, where a prefill lands and stalls the batch's decodes. That is exactly the
  decode-stall mechanism chunked prefill exists to prevent.
- **TTFT** still shows chunk's median/throughput cost (+20–30% p50) — the tail-vs-throughput
  tradeoff, now visible on the decode side too.

## Decode-length distribution (why max-tokens=1024 is the right cap)
We log `output_tokens` per request, so the realized decode-length distribution is measurable
(not assumed). Pooled over the 3 trials, mono+chunk (n=480/setting):

| max-tokens | min | p50 | p90 | p95 | max | mean | **% hitting cap (censored)** |
|---|---|---|---|---|---|---|---|
| 128 | 14 | 128 | 128 | 128 | 128 | 112 | **76.5%** |
| 512 | 14 | 315 | 512 | 512 | 512 | 298 | 24.4% |
| **1024** | 14 | 314 | 797 | 927 | 1024 | 358 | **3.5%** |

- **mt=128 is pathological**: the median *equals* the cap and 76.5% of requests are truncated —
  the "short decode" was imposed by the cap, not the workload. This is why the mt=128 TPOT
  null is an artifact.
- **mt=1024 recovers the natural distribution**: only 3.5% censored, so the aggregate is the
  model's real EOS behavior — median ~314, mean ~358, p95 ~927 tokens (a realistic
  ShareGPT-style response-length distribution). Raising the cap to 2048 would move little
  (only that 3.5% tail is clipped). **1024 is the smallest cap at which natural decode
  dominates**, so the TPOT-tail effect is genuine generation, not cap-clipping.

Padding note: the uncacheable prefill pad is deterministic filler (a unique per-request tag
`[req ci.turn]` + repeated "quick brown fox" text, identical across arms), **not** random
tokens — chosen to defeat prefix caching without differentially confounding the arms. Prefill
cost is token-*count*-bound, not content-bound; and the healthy decode distribution above
confirms the pad prefix does not induce degenerate generation.

## Consequence for the paper
The §6 "decode-bound, chunking buys nothing" claim is **scoped to short decodes**. Under
realistic generation lengths a genuine **stall-bound regime exists on 14B/TP=2**, and chunking
delivers a real TPOT-tail benefit (consistent with Sarathi). This:
1. means a **controller lever exists** → the adaptive-controller (slocvar) comparison is now
   meaningful (see the 3-arm follow-up);
2. requires **rescoping** the paper's decode-bound claim to short-generation workloads;
3. reinforces the "**report the right metric**" thesis — the effect is invisible in TTFT-mean
   and only shows in the TPOT tail at realistic decode lengths.

## Reproduction
`orchestrate_decode_sweep.sh` on the box: per mt ∈ {128,512,1024}, calibrate knee
(`calib_cs2b.sh` now `MAX_TOKENS`-aware) → `run_cs2_repl.sh` mono vs chunk at 0.8×knee,
Cs²=0, 3 trials → `analyze_decode_sweep.py` (TTFT + TPOT p50/p95). Raw:
`logs/2026-07-20-cs2repl_mt{128,512,1024}-{mono,chunk}-cv0-t{1,2,3}.jsonl`.

## Next
3-arm controller run (`mono vs chunk(512) vs slocvar`) at mt=1024, the strongest lever point:
does adaptive control capture chunk's TPOT protection without its TTFT cost?
