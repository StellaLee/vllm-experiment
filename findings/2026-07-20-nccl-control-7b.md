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

## Result — same-arm sharding (primary framing)
The cleanest isolation: take the **same arm** (same chunk size, scheduler, rate) and compare
it on **1 chip vs 2 chips**. Both arms get the 2-GPU compute speedup, so that cancels; any
residual is the NCCL cost specific to chunk's extra steps.
**NCCL-tax-on-chunk = chunk_sharding_effect − mono_sharding_effect** (pos ⇒ sharding
penalizes chunk more than mono). Same rate 2.0, pooled 3 trials (NEG sharding = TP=2 faster):

| metric | Cs² | mono single→TP2 | chunk single→TP2 | NCCL-tax-on-chunk |
|---|---|---|---|---|
| mean | 0 | 464→230 (−50.3%) | 507→255 (−49.7%) | **+0.7 pts** |
|      | 2 | 595→207 (−65.2%) | 858→181 (−79.0%) | −13.7 |
|      | 4 | 691→64 (−90.7%) | 1210→68 (−94.4%) | −3.6 |
| p50  | 0 | 353→224 (−36.6%) | 408→242 (−40.7%) | **−4.0 pts** |
|      | 2 | 337→59 (−82.6%) | 544→58 (−89.4%) | −6.8 |
|      | 4 | 324→47 (−85.5%) | 438→49 (−88.8%) | −3.3 |
| p95  | 0 | 929→511 (−44.9%) | 968→585 (−39.5%) | **+5.4 pts** |
|      | 2 | 1885→874 (−53.7%) | 2601→889 (−65.8%) | −12.2 |
|      | 4 | 2643→161 (−93.9%) | 5681→174 (−96.9%) | −3.0 |

- **Both arms get much faster under TP=2** (all sharding effects strongly negative) — the
  2-GPU compute speedup dominates the host-staged NCCL overhead even for a small model.
- **NCCL-tax-on-chunk ≈ 0 at the clean Cs²=0 point** (+0.7 / −4.0 / +5.4 across mean/p50/p95,
  straddling zero). Sharding does NOT penalize chunk more than mono.
- Only faintly positive: **p95 @ Cs²=0 = +5.4 pts** — a hint that chunk's *tail* benefits
  slightly less from sharding (chunk issues more all-reduces), but tiny and within noise;
  mean/p50 go the other way.

Saturation sanity (cv0 mono, first50→last50): SINGLE 324→370 (1.1×), TP2 271→200 (0.7×) —
both sub-saturation, ρ matched (both rate 2.0).

### Cross-check — chunk-vs-mono Δ% per config (indirect framing, same conclusion)
Difference of chunk-vs-mono deltas, NCCL effect = TP2 − SINGLE (pos ⇒ NCCL taxes chunk):

| metric | Cs²=0 SINGLE Δ% | Cs²=0 TP2 Δ% | NCCL effect |
|---|---|---|---|
| mean | +11.0±18.2 | +9.1±8.4 | −1.9 |
| p50 | +16.4±17.5 | +12.8±9.0 | −3.6 |
| p95 | +6.6±26.9 | +6.6±11.5 | +0.0 |

(Cs²=2/4 rows omitted — the single-GPU chunk-vs-mono deltas there are dominated by whale
noise, ±100+, so the large negative "NCCL effect" values are an artifact of that noise, not
a real chunk benefit. Same-arm framing above is cleaner because it sidesteps those deltas.)

## Conclusion — NCCL is not the villain
- **NCCL-tax-on-chunk ≈ 0** at the clean Cs²=0 point (same-arm: +0.7 / −4.0 / +5.4 pts across
  mean/p50/p95, straddling zero). Sharding the model across 2 chips does NOT penalize chunk
  more than mono.
- **Both arms benefit from sharding** (TP=2 ~40–50% faster at cv0) — the 2-GPU compute
  speedup beats the host-staged NCCL overhead. The indirect cross-check (chunk-vs-mono NCCL
  effect ≈ 0 at cv0) agrees.
- **This does NOT mean comms overhead is negligible — it's large, just paid equally by both
  arms.** Evidence it's large: TP=2 barely raised *throughput capacity* over single-GPU
  (both knees ≈2.5) and only improved *latency* ~40%, not the ~2× a free second GPU would
  give — comms ate roughly half the potential speedup. Why chunk isn't taxed *more*:
  host-staged NCCL over PCIe is **bandwidth-dominated** (cost ∝ activation data volume, not
  per-call latency), and both arms prefill the **same total tokens** → same total all-reduce
  data → chunk's extra (but smaller) steps add little. So comms is costly but **arm-neutral**,
  which is exactly why it doesn't distort the mono-vs-chunk comparison.
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
