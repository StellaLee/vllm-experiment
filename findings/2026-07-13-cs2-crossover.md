# Cs² crossover probe — genuine head-of-line term is not robustly detectable on single-GPU 7B

**Date:** 2026-07-13
**Question:** `eq:genuine` predicts chunked prefill (processor-sharing) beats run-to-completion
(mono) on mean TTFT only when prefill-size dispersion Cs² > 1, with a sign flip at Cs²=1:
`T_FCFS − T_PS = E[S_prefill]·(ρ/(1−ρ))·(Cs²−1)/2`. Can we demonstrate that crossover on a
single 4090 by manufacturing expensive, variable prefill with unique (uncacheable) padding?

## Setup
- Single RTX 4090, Qwen2.5-Coder-7B, vLLM 0.23.0, max-num-seqs 32, max-model-len 16384.
- Client: `src/replay_sharegpt.py` with new **variable padding** — per-request pad length
  ~ lognormal(mean 8000 chars ≈ 1750 tok, target cv2), deterministic per (conv,turn) so
  arms see identical prompts (paired). Uncacheable → real computed prefill.
- Two arms: **mono** (budget 16384 = run-to-completion) vs **chunk** (budget 512 = PS),
  reorder OFF. Open-loop Poisson, **single-turn** requests, rate 2.0 (calibrated
  sub-saturation; multi-turn/rate-3 saturated hard — TTFT grew 0.9s→16s within a run).
- x-axis = Cs² of the **total** prompt (pad + variable base), not pad alone (base adds a
  small offset: at cv2=0, Cs²_total ≈ 0.014, still ≈ uniform).

## Result 1 — single trial (grid 0..8), the crossover APPEARED
chunk Δ% vs mono (mean TTFT): +4 (Cs²0.01), +23 (0.72), **−6 (1.09), −18 (1.52)**, +50 (1.88).
Chunk flips from losing (<1) to winning (~1.1–1.5), reversing at the noisy extreme. The
short-request ("minnow") median metric agreed on the 1.09/1.52 wins. Looked like the
predicted sign flip.

## Result 2 — 3-trial replication (grid {0,2,4}), the crossover DID NOT HOLD
Paired per-trial chunk Δ% (mean ± std, wins/3):
| Cs²_tot | mono/trial (ms) | chunk/trial (ms) | chunk Δ% | wins |
|---|---|---|---|---|
| 0.01 | 11010/448/601 | 562/555/589 | −24±63 | 2/3* (cold-start outlier — junk) |
| 1.06 | 408/610/507 | 475/906/650 | **+31±16** | **0/3** |
| 1.50 | 418/619/578 | 490/845/610 | **+20±16** | **0/3** |

At Cs²>1 chunk **consistently loses** (0/3). The single-trial "win" was noise. **No robust
crossover on this hardware.**

## Diagnosis — why single-GPU is the wrong venue
Chunk overhead (chopping every prefill into ≤512-tok steps) is paid on **every** request;
the head-of-line **benefit** accrues only to the minority of short requests that queue
behind a whale. At the sub-saturation load we need (rate 2.0), whales are rare and their
prefill window short, so few minnows get blocked → benefit rare, overhead universal → net
negative. Raising ρ to make the benefit common tips a single 4090 into saturation (the
noise that wrecked the first attempt). **The window where the effect is both present and
un-saturated is too narrow on single-GPU 7B** because E[S_prefill] is too small (whales
block for ~1s, not seconds).

## Conclusion
- The genuine term is **negligible / within-noise on single-GPU 7B even under adversarial
  high-Cs² expensive-prefill conditions.** This *strengthens* the paper's "single-GPU
  prefix-aware wins are all artifact" claim: we tried hard to manufacture a genuine
  chunking win and could not get one to replicate.
- We therefore **do not** demonstrate the positive crossover here; the genuine term stays
  theoretical (`eq:genuine`) + cited (Sarathi A100). Positive demonstration deferred to a
  **2×4090 / 14B** run (E[S_prefill] 5–10× larger → whales block for seconds → benefit
  clears overhead at moderate load). See [[project_cs2_2x4090_plan]].

## Assets built (reusable next week)
- `src/replay_sharegpt.py`: `--pad-mean-chars --pad-cv2 --pad-seed --pad-min --pad-max`.
- `scripts/run_cs2_repl.sh` (paired mono/chunk, trials×grid), `scripts/run_cs2_ours.sh`
  (adaptive-controller third arm, reorder off), `scripts/analyze_cs2_repl.py` (paired ±std),
  `scripts/analyze_cs2_sweep.py` (three-way), `scripts/calib_cs2b.sh` (rate calibration).
- Raw logs: `logs/cs2sweep/2026-07-13-cs2sweep-*` (3-arm single trial) and `-cs2repl-*`
  (replication).
