# 7B single-GPU vs TP=2 control — host-staged NCCL does NOT tax chunk; the 14B null is not a comms artifact

**Date:** 2026-07-20
**Question:** The 14B mono-vs-chunk crossover ran TP=2 on 4090s with **host-staged NCCL**
(no NVLink, P2P disabled), which we flagged as biasing *against* chunk (chunk issues more,
smaller steps → pays the inflated fixed all-reduce latency more times). If so, the 14B
null could be a comms artifact rather than a property of the regime — most damaging for
the multi-chip scope claim. Control: run the **same padded Cs² sweep on 7B**, once
**single-GPU (TP=1, no NCCL)** and once **TP=2 (host-staged NCCL)**, at **matched ρ**, and
compare chunk Δ%. See `docs/2026-07-20-tp-nccl-comms-caveat.md`, memory `project-tp-nccl-caveat`.

## Setup
- 8×4090 box, GPUs (0,1). Qwen2.5-Coder-**7B**, vLLM 0.23.0, patches present but **gated
  off** (`DYNAMIC_CHUNK=0 PREFIX_REORDER=0` at server launch — verified: native static).
- Two configs: **SINGLE** (TP=1, GPU 0) and **TP2** (TP=2, GPUs 0,1).
- **Matched ρ:** each config's capacity knee found by calibration, then run at **0.8×knee**.
  Both landed at **knee≈2.5 → rate 2.0** (TP=2 barely raised 7B capacity — host-staged NCCL
  eats most of the 2-GPU benefit on a small model). Saturation-sanity confirms both sub-sat.
- Sweep: mono(16384) vs chunk(512), Cs²∈{0,2,4}, 3 trials, NUM=150, single-turn open-loop,
  pad-mean 8000c (~1750 tok). Paired seeds within a trial.

**Note on the first attempt (discarded):** calibrating each to its *highest-bounded* rate
(not 0.8×knee) put SINGLE at rate 2.5 ≈ ρ0.9 (near-saturation, ±58–82 noise) and TP2 at a
lower ρ — a ρ mismatch. The matched-ρ redo below fixes it.

## Result (matched ρ≈0.8; NEG = chunk wins; NCCL effect = TP2 − SINGLE)
| metric | Cs² | SINGLE Δ% | TP2 Δ% | NCCL effect |
|---|---|---|---|---|
| mean | 0 | +11.0±18.2 | +9.1±8.4 | **−1.9** |
|      | 2 | +47.9±58.4 | −7.9±15.8 | −55.7 |
|      | 4 | +58.6±60.7 | +6.6±6.4 | −52.0 |
| p50  | 0 | +16.4±17.5 | +12.8±9.0 | **−3.6** |
|      | 2 | +91.4±109.4 | −0.8±1.7 | −92.2 |
|      | 4 | +112.6±151.4 | +3.4±3.9 | −109.2 |
| p95  | 0 | +6.6±26.9 | +6.6±11.5 | **+0.0** |
|      | 2 | +20.6±43.4 | −17.0±23.1 | −37.6 |
|      | 4 | +58.8±56.2 | +11.0±8.0 | −47.8 |

Saturation sanity (cv0 mono, first50→last50): SINGLE 324→370 (1.1×), TP2 271→200 (0.7×) —
**both sub-saturation, ρ matched.**

Raw per-step overhead (cv0, pooled): SINGLE mono p50=353 / chunk p50=408 (**+55 ms**) |
TP2 mono p50=224 / chunk p50=242 (**+18 ms**).

## Conclusion — NCCL is not the villain
- **NCCL effect is never meaningfully positive** (the hypothesized "NCCL taxes chunk"
  direction). At the **cleanest point (Cs²=0, no whale noise) it is ≈0** on every metric
  (mean −1.9 / p50 −3.6 / p95 +0.0) — SINGLE and TP2 chunk-Δ% are essentially identical.
- **Raw overhead confirms it:** chunk's per-request overhead is *smaller* under TP=2
  (+18 ms) than single-GPU (+55 ms). If NCCL penalized chunk's extra steps, TP2's chunk
  overhead would be *larger*. It isn't.
- **Generalizes to 14B:** on a bigger model the all-reduce is a *smaller* fraction of each
  step (compute ~hidden², comms ~hidden), so NCCL taxes chunk even *less* on 14B.

**→ The 14B mono-vs-chunk null is a real result, not a host-staged-NCCL artifact.** This
resolves the threat-to-validity and strengthens the multi-chip scope claim.

## Caveats
- The large −55/−92/−109 "NCCL effect" values at Cs²=2/4 are **inflated by SINGLE-side
  noise** (±100+, whale-driven — single-GPU whales are pricier → noisier), NOT a real chunk
  benefit. The reliable evidence is **cv0 (≈0)** + the **raw per-step overhead**.
- Tests **host-staged** NCCL specifically; NVLink would have even less overhead → same
  conclusion, more so.

## Reproduction
Orchestrated by `orchestrate_7b_nccl.sh` (calibrate each config's knee → run at 0.8×knee →
tag outputs `cs2replSINGLE-*` / `cs2replTP2-*` → `analyze_7b_nccl.py`). Manual equivalent:
```bash
source scripts/env.sh
# SINGLE: CUDA_VISIBLE_DEVICES=0 TP=1 ; TP2: CUDA_VISIBLE_DEVICES=0,1 TP=2
CUDA_VISIBLE_DEVICES=<dev> PYTHON=$(command -v python) \
  MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct \
  TP=<1|2> NUM=250 RATES="2.0 2.5 3.0" bash scripts/calib_cs2b.sh   # -> knee; run at 0.8*knee
CUDA_VISIBLE_DEVICES=<dev> PYTHON=$(command -v python) MODEL=<7b> \
  TP=<1|2> RATE=<0.8*knee> CV2_GRID="0 2 4" NUM_CONVS=150 TRIALS="1 2 3" \
  bash scripts/run_cs2_repl.sh
```
Raw logs (remote): `logs/2026-07-20-cs2repl{SINGLE,TP2}-{mono,chunk}-cv{0,2,4}-t{1,2,3}.jsonl`.
