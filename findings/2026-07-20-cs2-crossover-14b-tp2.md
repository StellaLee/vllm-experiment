# Cs² crossover on 14B / TP=2 — replicates the single-GPU null; not a clean crossover (and confounded by host-staged NCCL)

**Date:** 2026-07-20
**Question:** On multi-chip (14B, tensor-parallel 2), does chunked prefill (PS) beat
run-to-completion (mono/FCFS) on mean TTFT when prefill-size dispersion Cs²>1 — the
crossover `T_FCFS − T_PS = E[S]·(ρ/(1−ρ))·(Cs²−1)/2` that single-GPU 7B could not
robustly show (see `2026-07-13-cs2-crossover.md`)? The premise: 14B/TP=2 raises
E[S_prefill] to seconds, so the head-of-line benefit should clear chunk's overhead.

## Setup
- **8×4090 box** (`ssh 183.147.142.123`), GPUs **(0,1)** — same PCIe switch (`PIX`).
  **No NVLink; PCIe P2P disabled → NCCL all-reduces are host-staged** (see the threat
  in `docs/2026-07-20-tp-nccl-comms-caveat.md`).
- Qwen2.5-Coder-**14B**-Instruct (bf16, ~28 GB), vLLM 0.23.0 venv, TP=2,
  `max-num-seqs 32`, `max-model-len 16384`, `gpu-util 0.90`.
- Client `src/replay_sharegpt.py`: single-turn, open-loop Poisson, pad-mean 8000 chars
  (~1750 tok, uncacheable), pad-cv2 ∈ {0,1,2,4}, paired seeds across arms.
- **mono** = `--max-num-batched-tokens 16384` (run-to-completion), **chunk** = `512` (PS),
  reorder OFF, dynamic OFF. 3 trials, NUM=150/point, rate 1.0.

## Rate calibration was the load-bearing step
14B/TP=2 capacity is only **~1.3–1.7 req/s** (larger E[S] + host-staged TP overhead), so
the 7B rate (2.0) **saturates**. Observed: rate **2.0** → TTFT 2.5–20 s, non-stationary;
rate **1.5** → 7–19× within-run growth (mis-passed a short NUM=100 probe because 1.5 was
only tested on a *warm* server; the real run's cold first point tips it over). Only rate
**1.0** is genuinely sub-saturation (cold-validated: first50→last50 = 532→509 ms, flat).
**Lesson: calibrate at realistic NUM and on a cold server; the sub-saturation band is
narrow and lower than the 7B setup.**

## Result (rate 1.0, sub-saturation confirmed by flat drift)
Paired per-trial `chunk Δ% = (chunk_mean − mono_mean)/mono_mean`, mean ± std over 3 trials:

| Cs²_tot | mono/trial (ms) | chunk/trial (ms) | chunk Δ% | wins |
|---|---|---|---|---|
| ~0.01 | 774/896/1149 | 795/819/853 | **−10.6 ± 11.7** | 2/3 |
| ~1.06 | 710/912/1416 | 965/1773/1038 | **+34.5 ± 49.4** | 1/3 |
| ~2.0  | 959/1199/1197 | 931/1190/910 | **−9.2 ± 10.4** | 3/3 |
| ~4.0  | 695/1102/843 | 775/1206/2060 | **+55.1 ± 63.1** | 0/3 |

Drift sanity (median first50→last50) flat-or-declining in ~23/24 cells (one blip:
chunk cv2 508→1037), absolute TTFT 0.3–2 s → operating point valid.

## Interpretation — no robust crossover
- **Non-monotonic and noise-dominated.** Signs go −,+,−,+ (not the predicted monotonic
  lose-below-1 / win-above-1). Std ≥ mean at three of four points; the large means are
  driven by single-trial outliers (1773, 2060 ms).
- **Only Cs²≈2 behaves as predicted** — chunk wins 3/3 at −9.2% with a tight spread — but
  it is isolated (Cs²=4 reverses). A weak, real signal at moderate variance, swamped by
  noise at the extremes (at Cs²=4 the lognormal is so skewed that whales are rare and the
  many minnows pay chunk overhead for no benefit).
- **Net: 14B/TP=2 replicates the 7B null pattern** — chunk within noise, no clean sign flip.

## The null is confounded — cannot yet claim "no genuine win on multi-chip"
Host-staged NCCL (no NVLink) inflates chunk's per-step overhead (chunk issues more,
smaller steps → pays the fixed all-reduce latency more times), **biasing the comparison
against chunk**. So this null could be a comms artifact rather than a property of the
regime. **Required control: 7B single-GPU (no NCCL) vs 7B TP=2 (NCCL) on the identical
padded sweep** to bound the penalty. See `docs/2026-07-20-tp-nccl-comms-caveat.md` and
memory `project-tp-nccl-caveat`. Until then: report as suggestive, not conclusive.

## Env fixes required to get vLLM 0.23.0 serving on this box (for repro)
- `apt-get install python3.10-dev` (inductor needs `Python.h`).
- `CUDA_HOME` → the venv's bundled cu13 toolkit (system nvcc 12.9 mismatches torch cu130).
- `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer 0.6.12 cccl headers reject nvcc 13.2).
- **Real patch bug fixed:** `scheduler.py` needs a module-level `import os` (0.23.0 lacks
  it; the `__init__` wiring calls `os.getenv` at Scheduler scope → NameError). Now Patch 0
  in `scripts/patch_scheduler.py`. All three are folded into `scripts/env.sh`.

## Reproduction
```bash
# on 183.147.142.123, repo /root/pli/vllm-experiment
source scripts/env.sh                       # venv + CUDA_HOME + flashinfer-off; --check to verify
PYTHON=$(command -v python) bash scripts/apply_patches.sh   # idempotent; installs Patch 0..3b

# 1) calibrate the sub-saturation rate (cold, realistic NUM, cv2=0 = heaviest)
CUDA_VISIBLE_DEVICES=0,1 PYTHON=$(command -v python) \
  MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct \
  TP=2 NUM=300 RATES="1.0 1.25 1.5" bash scripts/calib_cs2b.sh
#   -> take the highest rate that is BOUNDED on a COLD server (here 1.0)

# 2) the paired crossover sweep (mono vs chunk x Cs2 x 3 trials)
CUDA_VISIBLE_DEVICES=0,1 PYTHON=$(command -v python) \
  MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct \
  TP=2 RATE=1.0 CV2_GRID="0 1 2 4" NUM_CONVS=150 TRIALS="1 2 3" \
  bash scripts/run_cs2_repl.sh

# 3) analyze (paired ±std, wins/N)
python scripts/analyze_cs2_repl.py            # canonical repo analyzer
```
Raw logs (remote): `logs/2026-07-20-cs2repl-{mono,chunk}-cv{0,1,2,4}-t{1,2,3}.jsonl`,
server logs `logs/2026-07-20-cs2repl-{mono,chunk}-server.log`.

## Next
1. **7B single-GPU vs 7B TP=2 NCCL control** (the fork — interprets this null).
2. Dynamic controller (`run_cs2_ours.sh`) only if a stall-bound lever is confirmed.
3. If the null survives the control: it strengthens the paper's artifact thesis (even
   large-model multi-chip can't manufacture a robust chunking win). If not: need NVLink.
