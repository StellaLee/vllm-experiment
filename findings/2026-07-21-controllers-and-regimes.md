# Adaptive Chunk-Size Controllers & the Regimes Where Chunking Matters — 2026-07-21

**One-line result:** Across BurstGPT real-trace replay and closed-loop multi-turn ShareGPT at three
concurrency/batch settings, **no adaptive chunk-size controller (bang-bang, feedforward-v1,
feedforward-v2) beats the best static arm** — every apparent win is single-trial noise or the
controller degenerating to a static arm. The deeper reason: the regimes we can easily reach
(short prompts, or overload) don't sustain the *decode-stall* condition chunking exists to manage.
Chunking's genuine TPOT win requires **long prefills + a decode batch + service variance + sub-
saturation load** simultaneously — a narrow target that short-prompt or overloaded benchmarks miss.

> Companion to `findings/2026-07-21-cs2-crossover-sweep.md` (the synthetic open-loop Cs² sweep).
> This file covers the follow-on real-trace + closed-loop + controller-design work.

---

## 1. BurstGPT real-trace 3-arm replay

Integrated the real BurstGPT trace (`HPMLL/BurstGPT`, `BurstGPT_1.csv`, 1.43M rows) to drive the
mono/chunk/ours arms with **real arrival burstiness + trace-derived token lengths**, via BurstGPT's
own `burstgpt-bench` client against vLLM's native `/generate` api_server.

**Four bugs in BurstGPT's own harness had to be fixed to get any data** (each is a small artifact
worth noting — the tooling the community uses to benchmark bursty serving is itself flawed):

| bug | symptom | fix |
|-----|---------|-----|
| stream parser expects `\0`-delimited | 0 records logged (v0.23 api_server sends `\n`) | patched `backends.py` |
| zero-response trace rows → `max_tokens=0` | HTTP 500 per empty-response row | filter `resp≥1` in slicer |
| **arrival bug**: `arrival_time` = per-req *delta* but runner sleeps it from t=0 | all requests fire at once → saturation (23s TTFT) | patched `workload.py` → cumulative offsets |
| **prompt length hard-capped at 1024** (fixed 1024×1024 matcher table, 300-prompt pool) | service-size distribution censored at 1023 tok (Cs² 1.48→0.89) | *unfixed — fundamental to the tool* |

**The 1024-token cap is notable**: BurstGPT's benchmark harness censors prompt length at 1024 — the
exact decode/prefill-censoring artifact our paper is about, in a widely-used bursty-serving benchmark.

**Result** (conv window, gap-capped arrivals, realized service Cs²=0.89, sub-saturation ~0.7 rps):

| arm | TTFT-mean | ΔTTFT | TPOT-p95 | ΔTPOT-p95 |
|-----|-----------|-------|----------|-----------|
| mono | 61 ms | +0.0% | 19.5 | +0.0% |
| chunk | 62 ms | +2.4% | 19.4 | −0.2% |
| ours | 59 ms | −2.8% | 19.2 | −1.5% |

All arms within ±3% (tied). Light load + capped-short prompts → nothing stresses the scheduler.
Controller dwelled **99% at the ceiling** (≡ mono); the −2.8% is noise between identical configs.
Budget trace: grows 512→16384 in the first ~150 steps, then pinned (31 transitions total).

---

## 2. Closed-loop multi-turn ShareGPT: concurrency & batch-size sweep

Real ShareGPT prompts, closed-loop (`--concurrency`), multi-turn (4 turns), mt=1024, prefix-caching
on. Swept load to find where the arms separate.

### 2.1 TPOT rises with the DECODE BATCH (max-num-seqs), not concurrency

| config | mono TPOT-p50 | mono TPOT-p95 | mono TTFT-mean |
|--------|--------------|--------------|---------------|
| conc=16, max-seqs=32 | 22.7 | 23.5 | 95 ms |
| conc=96, max-seqs=32 | 28.0 | 28.9 | 4522 ms |
| conc=128, max-seqs=128 | **48.1** | **55.4** | 506 ms |

**Key control-knob finding:** raising concurrency at fixed `max-seqs=32` exploded TTFT (queue) but
barely moved TPOT (+23%) — the decode batch is capped at 32, so extra load queues rather than
enlarging the decode batch. Raising **`max-num-seqs` 32→128 roughly doubled TPOT** (28→48-55ms): the
decode batch is what drives per-step decode cost. This matters for the controller: only at
max-seqs=128 does per-step latency (~48-55ms) approach the 50ms SLO, so only there does the
controller ever engage.

### 2.2 The arms do NOT separate — differences are noise

Two **identical-config** runs (conc=128, max-seqs=128, Cs²=1.064) gave chunk ΔTTFT = **+10.3%** and
**−10.2%** — a 20-point sign flip on an unchanged setup. Single-trial TTFT is dominated by the turn-1
cold-start burst (128 convs firing at once), which dwarfs steady-state (turns 2–4 ≈ 150ms vs turn-1
≈ 1700ms) and controls the mean. **Steady-state (turns 2–4): all arms within ±4% of mono on TTFT and
±2% on TPOT.** Any apparent controller "win" is a turn-1 coin-flip.

---

## 3. Adaptive controllers: three designs, three ways to fail

All three were tested as a 4th (or 3rd) arm in the max-seqs=128 regime — the only regime where the
controller engages at all. Budget trajectories (`logs/2026-07-21-controller-compare.png`):

| controller | mechanism | budget behavior | verdict |
|-----------|-----------|-----------------|---------|
| **bang-bang** (slocvar) | AIMD servo of CVaR(step-latency) to 50ms SLO | ~1% interior; floor 512 then hard jump to ceiling 16384, pinned | limit cycle / rails |
| **feedforward v1** | τ = EMA(dt / *granted budget*); B=SLO/τ | inflates to ceiling (81% dwell); decode-only steps have granted≫actual → τ underestimated | rail (up) |
| **feedforward v2** | τ = EMA(dt / *actual tokens*); B=SLO/τ | collapses to floor (100% dwell); see §3.1 | rail (down) |

### 3.1 Why feedforward can't find an interior point (the real finding)

Feedforward assumes `step_time ≈ τ · tokens` (one parameter). But at high concurrency the true model is

```
step_time ≈ β(decode_batch) + α · prefill_tokens
```

with a **large fixed decode-batch cost β**. At max-seqs=128, decoding the 128-seq batch alone costs
~50ms — nearly the whole SLO — almost independent of prefill tokens. v2's `τ = dt / total_tokens`
folds β into τ, inflating it → `SLO/τ` collapses to the floor. v1's `dt / granted_budget` had the
opposite-signed error → ceiling. **A single-parameter cost model cannot yield an interior chunk size
at high concurrency; it always rails.** A correct controller would need to fit both α and β (decode
fixed cost + prefill marginal cost) and solve `chunk = (SLO − β)/α`. Even then it only matters when
there is prefill worth sizing — i.e., long prefills, which these workloads lack.

### 3.2 Controller scorecard (max-seqs=128, Cs²=1.064, single trial each)

| arm | ΔTTFT | ΔTPOT-p95 |
|-----|-------|-----------|
| chunk (static 512) | −10 to −11% (noise) | +1.4 to +1.8% |
| ours (bang-bang) | −10 to −15% (noise) | +1.7 to +1.8% |
| feed (v1) | −3.5% | −0.6% |
| feed (v2) | −9.1% | +5.3% |

TPOT differences are all within ±2% (noise) except v2's degenerate +5.3%. **No controller shows a
gain over the best static arm that survives the turn-1-noise / single-trial caveat.**

---

## 4. Long-prompt + high-concurrency: an overload, not a clean test

Attempt to reach the decode-stall regime directly: single-turn, **~5k-token prompts** (20k-char
uniform pads), conc=128, max-seqs=128, mono vs chunk.

**Result was invalid (overload):** TTFT = **84 seconds**, only **107/200 requests completed**,
realized Cs²=0.003 (uniform pads → no variance). `conc=128 × ~5k-tok prompts` ≈ 640k tokens of KV,
vastly exceeding capacity → preemption thrash. **This is not "high concurrency," it is overload** —
KV memory caps the *achievable* concurrency far below 128 when prompts are long (a bounded-resource
echo of the crossover paper's context-limit finding).

Notably chunk came out **+22% WORSE on TPOT-p95** here, not better. Mechanism: with **uniform** long
prefills under chunking, *every* step carries a 512-tok prefill chunk → every decode token waits
behind prefill → TPOT uniformly elevated (p50 +13%, p95 +22%). Mono concentrates prefill into fewer
long steps → most decode steps pure/fast, rare huge stalls that land at **p99+** (past what we
measured). So chunk's "always-slightly-slower" loses to mono's "usually-fast-rarely-terrible" at
p50–p95 — **but only because (a) it is overloaded and (b) pads were uniform (no whales to protect
against).** Not a valid win for either side.

---

## 5. Synthesis: where chunking actually wins (and why we kept missing it)

Chunked prefill's genuine benefit (validated by vLLM/Sarathi-Serve adoption, and by our own
**decode-length sweep: chunk TPOT-p95 −22% vs mono at mt=1024**) requires **four conditions at once**:

1. **Long prefills** (thousands of tokens) — a big prefill to slice.
2. **A decode batch to protect** (moderate–high concurrency, large max-num-seqs).
3. **Service variance** (whales relative to the rest — Cs²>0) — otherwise chunking elevates *every*
   step instead of shielding rare stalls.
4. **Sub-saturation load** — not overload; at overload everything is queue-bound and chunk's per-step
   overhead dominates.

Today's runs each hit *some* subset, never all four:

| run | long prefill | decode batch | variance | sub-saturation | → |
|-----|:---:|:---:|:---:|:---:|---|
| BurstGPT | ✗ (capped 1024) | ✓ | ✓ | ✓ | tie |
| ShareGPT closed-loop | ✗ (short) | ✓ | ~ | ✓ | tie |
| long-prompt conc=128 | ✓ | ✓ | ✗ (uniform) | ✗ (overload) | chunk worse |
| **decode-length sweep (07-20)** | ✓ | ✓ | ✓ | ✓ | **chunk −22% (wins)** |

**The winning regime is narrow**, which is the paper's point: benchmarks reporting chunking as a
general win are implicitly sitting in it; much real serving (short prompts, or bounded concurrency
from long-prompt KV pressure) is not, where chunking is neutral-to-harmful and adaptive control buys
nothing.

---

## 6. Caveats

- **Single trial** on every run in §1–4. TTFT is turn-1-cold-start-noise-dominated; the "no gain"
  conclusion is directionally strong (sign-flips across identical runs) but should be nailed with n≥3
  before publication.
- BurstGPT prompt lengths are **censored at 1024** by its harness → its service Cs² understated.
- Long-prompt run **overloaded**; use lower concurrency or open-loop calibrated rate for a clean read.
- Controllers only meaningfully tested at max-seqs=128 (the only engaging regime); their behavior at
  a correctly-sized α/β model is untested.

---

## 7. Artifacts

**Code (new/changed this session):**
- `orchestrate_burstgpt_one.sh`, `scripts/analyze_burstgpt.py` — BurstGPT real-trace 3-arm runner
- BurstGPT patches (in `/root/pli/BurstGPT/example/src/burstgpt/`): `backends.py` (newline stream),
  `workload.py` (cumulative arrivals) — **local-only, not upstreamed**
- `orchestrate_sharegpt_cl.sh`, `scripts/analyze_sharegpt_cl.py` — closed-loop multi-turn 3/4-arm
  (arm-selectable via `RUN_ARMS`, pad via `PAD_MEAN`/`PAD_CV2`, `MAX_SEQS` knob)
- `scripts/hotpatch_slo_tail.py` — added `CHUNK_MODE=feedforward` (v2: τ from actual tokens)
- `scripts/hotpatch_ff_tokens.py` — wires actual scheduled-token count into the controller

**Data/figures (`logs/`):**
- `2026-07-21-controller-compare.png` — bang-bang vs feedforward budget trajectories (the key figure)
- `burstgpt_one-*-detail.jsonl`, `*-sharecl_{c16,c96,c128,ffv1,ffv2}-*` — per-run records
- `2026-07-21-sharecl-{ours,feed}-chunktrace.csv` — controller budget traces

---

## 8. Next steps

1. **Clean decode-stall test** — long prefills (~5k) at **lower concurrency (24–32)** or open-loop
   calibrated rate, **with `pad_cv2≈1`** (variance) — the regime where chunk should clearly win TPOT
   without overloading. Confirms the §5 four-condition thesis.
2. **Replicate at n≥3** whichever regime we report, to convert "no controller gain" from directional
   to error-barred.
3. **Correct controller** (if pursued): fit α/β step-cost model, solve `chunk=(SLO−β)/α`. But per §3
   its value is bounded to the narrow winning regime; likely a footnote, not a headline.
4. Decide framing: the accumulated evidence supports a **regime-map** paper — "chunked prefill wins
   in a narrow long-prompt/high-batch/variance/sub-saturation regime; elsewhere neutral-to-harmful,
   and no adaptive controller helps" — pairing the crossover sweep (§ companion file) with this
   controller/regime characterization.
