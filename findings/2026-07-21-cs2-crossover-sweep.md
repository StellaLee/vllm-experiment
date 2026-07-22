# Cs² Crossover Sweep: Chunked Prefill vs Run-to-Completion — 2026-07-21

**One-line result:** On a realistic bounded-context server (16k) at moderate open-loop load,
chunked prefill (small token budget) **loses on TTFT across the entire *attainable* service-time
variance range** (realized Cs² up to ~1.08), and its TPOT-tail benefit is marginal. The regime
where chunking is predicted to win (Cs² ≫ 1) is **architecturally hard to reach** because the
context-length cap bounds achievable prefill-size dispersion. An adaptive controller that tries to
interpolate between the two static budgets **never beats the better static arm at any Cs²**.

> **Correction of note:** an underpowered pilot (n=3) appeared to show a clean TTFT/TPOT crossover
> at Cs²≈1.1 (chunk −6.7% TTFT, −38% TPOT-p95). The n=8 confirmation shows those "flips" were a
> whale-timing outlier in a single mono trial. **The crossover did not reproduce.** This flip-that-
> wasn't is itself a clean example of the benchmarking fragility the work is about.

---

## 1. Experimental setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-Coder-14B-Instruct |
| Hardware | 8× RTX 4090, TP=2 per server (host-staged NCCL, no NVLink) |
| Engine | vLLM 0.23.0, **V1 scheduler** (patched: `v1/core/sched/scheduler.py`) |
| max-model-len | 16384 |
| max-num-seqs | 32 |
| max-tokens (decode cap) | **1024** (≈3.5% censored — natural distribution; see §2) |
| Arrivals | **open-loop Poisson**, rate 0.64 req/s, single-turn |
| Dataset | ShareGPT v3, per-request unique pad tag (defeats prefix caching) |
| Prefix caching | on, but neutralized by unique pad tag (zero cross-request hits) |
| Trials | n=3 (Cs²<1 points), **n=8 (Cs²≈1 points)** |

**Three arms** (per-server config byte-identical except `--max-num-batched-tokens` and controller env):

| Arm | token budget | discipline (analogy) | mechanism |
|-----|-------------|----------------------|-----------|
| **mono** | 16384 (static) | FCFS / run-to-completion | whole prefill in one step (no prompt reaches 16k) |
| **chunk** | 512 (static) | Processor Sharing | prefill sliced to 512-tok pieces, interleaved with decode |
| **ours** | 512→16384 (adaptive) | — | slocvar controller: AIMD on CVaR of per-step latency vs 50ms SLO, starts at 512 |

**Chunked-prefill mechanism (verified in server logs):** vLLM V1 has `enable_chunked_prefill=True`
by default; the arm distinction is *purely* `max_num_batched_tokens`. Log confirms
`Chunked prefill is enabled with max_num_batched_tokens=512` (chunk) vs `=16384` (mono). Because
`max_model_len=16384` and pads cap at ~12.5k tokens, **no prompt ever reaches the 16384 budget, so
mono never actually chunks** = genuine run-to-completion. The contrast is not confounded by a flag.

---

## 2. Decode-length censoring (why max-tokens=1024)

The decode-length cap silently truncates the output distribution. Choosing it wrong manufactures
results:

| max-tokens | p50 out | % censored (hit cap) |
|-----------|---------|----------------------|
| 128 | 128 | **76.5%** (artifact) |
| 512 | 315 | 24.4% |
| **1024** | 314 (mean 358, p95 927) | **3.5%** (natural) |

At mt=128, three-quarters of requests are truncated — any policy comparison is run on a decode
distribution the benchmark invented. All results here use mt=1024 (3.5% censored).

---

## 3. Cs² calibration: nominal knob vs realized dispersion

Service "size" ≈ prefill work ∝ prompt-token count. Variance is injected via a lognormal pad-length
distribution (`--pad-cv2`), clipped to [100, 50000] chars. **The knob is compressed ~0.6× by
clipping**, so the nominal cv2 is *not* the realized Cs²:

| nominal pad-cv2 | realized Cs² (prompt) | mean prompt tok | max/mean |
|----------------|----------------------|-----------------|----------|
| 0.5 | **0.370** | 1509 | 3.9× |
| 1.0 | **0.663** | 1451 | 5.1× |
| 1.5 | **0.911** | 1409 | 6.2× |
| 1.75 | **0.970** (n=8) | ~1400 | — |
| 2.25 | **1.075** (n=8) | ~1400 | — |

**Bounded-context reachability limit (novel):** high Cs² comes from the lognormal's right tail
(whales), but `pad_max` (~12.5k tok) is capped by `max_model_len=16384`. Clipping those whales is
exactly what kills realized dispersion. **A 16k-context server can barely push realized Cs² past 1.**
This is a structural limit, not a tuning oversight: the regime where chunking wins may be
architecturally out of reach on bounded-context serving.

---

## 4. Results — the crossover sweep

Deltas are vs the **mono** arm at the same point. Crossover signal = chunk ΔTTFT-mean (M/G/1
predicts >0 below Cs²=1, <0 above).

### 4.1 TTFT-mean (chunk & ours vs mono)

| realized Cs² | n | chunk ΔTTFT | ours ΔTTFT |
|---|---|---|---|
| 0.370 | 3 | +22.9% | +9.1% |
| 0.663 | 3 | +18.9% | +24.0% |
| 0.911 | 3 | +15.8% | +1.6% |
| **0.970** | **8** | **+15.0%** | +3.7% |
| **1.075** | **8** | **+13.4%** | +1.9% |

Chunk is worse at **every** attainable Cs², declining monotonically (+22.9 → +13.4%) but **never
crossing zero**. At the highest reachable point (1.075), chunk still costs +13.4% TTFT.

### 4.2 TPOT-p95 (decode-tail protection — chunking's raison d'être)

| realized Cs² | n | chunk ΔTPOT-p95 | ours ΔTPOT-p95 |
|---|---|---|---|
| 0.370 | 3 | +23.4% | +27.3% |
| 0.663 | 3 | +4.0% | +26.7% |
| 0.911 | 3 | +5.5% | +15.3% |
| **0.970** | **8** | **−10.4%** | +5.2% |
| **1.075** | **8** | **−1.2%** | −1.4% |

TPOT-tail protection is **marginal and non-monotone**: chunk is *worse* on the decode tail below
Cs²≈0.9 (protecting a non-problem — at open-loop rates decode rarely stalls), shows a hint of
protection near Cs²≈0.97 (−10%), and is essentially tied by 1.075. Not the dramatic flip the pilot
suggested.

### 4.3 The n=3 → n=8 correction (the artifact)

The pilot (n=3) at the two ≥1 points claimed a clean crossover:

| point | pilot n=3 | confirmed n=8 |
|---|---|---|
| chunk ΔTTFT @ Cs²≈1.1 | **−6.7%** (chunk wins) | **+13.4%** (chunk loses) |
| chunk ΔTPOT-p95 @ Cs²≈1.1 | **−38.2%** | **−1.2%** |

**Root cause:** at high Cs² the per-trial TTFT-mean is dominated by rare whale-timing. The n=3 mono
baseline had one unlucky trial (per-trial mono TTFT-mean = 631 / 494 / **807** ms); that 807 outlier
inflated mono's mean and made both chunk and ours look better. Paired per-trial, chunk was actually
worse in 2 of 3 trials. **n=3 is under-powered above Cs²=1; n≥8 is required there.** Below Cs²=1 the
variance is tight and n=3 is fine.

---

## 5. The adaptive controller (ours)

The slocvar controller **never beats the better static arm at any Cs²**, but its failure mode
changes with variance:

| regime | behavior | verdict |
|---|---|---|
| Cs² < 0.9 | dominated middle: pays TTFT cost *and* worst TPOT-p95 (+27%) | strictly dominated |
| Cs² ≈ 1 | ≈ mono on all metrics (TTFT +2–4% vs chunk's +13–15%); best TTFT-p95 at 1.075 | harmless tracker, no net win |

**It is a limit cycle, not a regulator.** Budget dwell is ~90%+ at the two rails (512 and 16384),
near-zero interior. Transitions *grow* with Cs² (645 → 663 → 1230 → 2438 → **2646**): higher variance
= more prefill spikes = more rail-to-rail thrashing. At high Cs² its ceiling dwell rises to 55% — it
is effectively *learning that mono is the right static arm and degenerating toward it*, the hard way.

**Why no interior operating point exists (structural):** the controlled signal (CVaR of per-step
latency) is **bimodal and coupled to the action** — ~27ms at the floor (decode-only steps), 100–140ms
at the ceiling (a whale prefill in one step). The 50ms SLO sits in the empty gap between, so no single
budget parks the signal at the setpoint. AIMD-to-a-setpoint assumes a monotone continuous plant; this
plant is bistable → limit cycle. No SLO fixes it — extreme SLOs merely collapse the controller to one
static arm (floor-pinned = chunk, ceiling-pinned = mono). See budget-trajectory PNGs (§9).

**Takeaway:** "adaptive buys nothing over picking the right static budget." At high Cs² the right
choice is mono, and the controller's best outcome is to imitate mono — an argument for *just picking
mono*, not for running the controller.

---

## 6. Theory — M/G/1 FCFS vs PS (directional, not literal)

The arms map onto M/G/1 service disciplines:

- **mono ≈ FCFS/run-to-completion:** a whale's prefill occupies one long uninterrupted step, freezing
  everyone else → head-of-line blocking. P-K wait ∝ (1+Cs²), grows with variance.
- **chunk ≈ Processor Sharing:** the whale is time-sliced across steps, interleaved with all decodes
  → no monopolization. PS mean response E[S]/(1−ρ), **insensitive to Cs²**.
- Difference: `W_FCFS − W_PS ∝ (ρ/(1−ρ))·E[S]·(Cs²−1)/2` → sign set by (Cs²−1); crosses at Cs²=1.

**Why we saw no crossover even at Cs²=1.075:** the benefit term `(Cs²−1)/2 ≈ 0.04` at Cs²=1.075, and
at our low load `ρ/(1−ρ)` is small — so the *predicted* queueing benefit near Cs²=1 is nearly zero and
is **swamped by chunking's fixed overhead** (the residual +13% is chunk's structural per-step cost).

**Where the model is loose (all push the empirical crossover right of 1):**
1. **Batching/parallelism.** vLLM runs the whole batch (≤32 seqs) in one forward pass; the GPU's
   effective rate *rises* with batch size rather than being split 1/n as PS assumes. mono is therefore
   *already partly PS*, muting FCFS's disadvantage.
2. **Two-phase service** (prefill then N decode steps) vs M/G/1's scalar service.
3. **State-dependent service time** (depends on batch composition) vs i.i.d. assumption.
4. **Finite MPL** (max-num-seqs + KV cache) → really Limited Processor Sharing, not PS∞.

Recommended use: M/G/1 FCFS-vs-PS as a **directional scaffold** that correctly identifies Cs² as the
axis and predicts the sign — not as a quantitative model. The rightward crossover shift is itself
evidence the baseline already self-shares (a systems insight).

---

## 7. What this says about the literature (deflationary framing)

- Reported chunked-prefill / prefix-aware "wins" depend on undisclosed benchmarking choices —
  aggressive decode caps (§2) and low/absent static-small baselines.
- In the realistic regime (Cs² < 1, open-loop, natural decode cap), chunking is **strictly harmful**
  on TTFT and does **not** protect the TPOT tail (nothing to protect — decode doesn't stall).
- The regime where chunking wins (Cs² ≫ 1) is **architecturally hard to reach** on a bounded-context
  server (§3).
- An adaptive controller cannot rescue this — the mono↔chunk tradeoff has **no interior operating
  point** (§5).

*(Cross-reference: `findings/2026-07-06-chunk-size-experiment.md` reported a dynamic-chunk "win" that,
under this tighter design — static-small arm + open-loop + fixed decode cap — does not survive; that
win rested on a static-2048 baseline, rate=inf saturation, and uncontrolled output length.)*

---

## 8. Caveats & threats to validity

1. **Contention confound (n=8 points).** The n=8 ≥1 points ran all 3 arms *concurrently* on separate
   GPU pairs (for within-point load symmetry). Chunk does more scheduler steps and may be penalized
   more by host/PCIe contention → chunk's +13% could be modestly inflated. The *sign* (no flip) is
   robust (the pilot flip was a demonstrable mono outlier), but the *magnitude* is not contention-clean.
   → **Next: isolated n=8 rerun** to lock magnitudes.
2. **Mixed n.** Cs²<1 points are n=3 (tight variance, acceptable); Cs²≈1 points are n=8.
3. **Single load point** (rate 0.64, low ρ). The queueing term is small here; the crossover, if
   reachable, needs higher ρ to surface. → **Next: load sweep.**
4. **TP=2 = host-staged NCCL** (no NVLink) — biases prefill cost; conservative for chunk.
5. **BurstGPT context:** real-workload service-side Cs² ≈ 0.46 (variance is on *arrivals*, not prefill
   size) — i.e., production workloads sit well below the crossover.

---

## 9. Artifacts (files)

**Data (remote `/root/pli/vllm-experiment/logs/`, mirror to local before commit):**
- `2026-07-21-cross{05,10,15,175,225}-{mono,chunk,ours}-cv0-t*.jsonl` — per-request TTFT/TPOT/tokens
- `2026-07-21-cross{05,10,15,175,225}-ours-chunktrace.csv` — per-step controller budget traces
- `logs/crossoverhi_ANALYSIS.txt`, `logs/crossoverconf_ANALYSIS.txt` — analyzer outputs

**Figures (local `logs/`):**
- `2026-07-21-cross05-ours-chunktrace.png`, `...cross10-...png` — budget trajectories (bang-bang)
- (regenerate cross175/225 n=8 traces for the intensified-thrashing figure)

**Code:**
- `scripts/analyze_crossover.py` — 5-point analyzer (realized Cs², TTFT + TPOT deltas, crossover summary)
- `scripts/run_cs2_3arm.sh`, `scripts/run_arm_unit.sh` — single-point / single-arm runners
- `orchestrate_crossover.sh`, `orchestrate_crossover_hi.sh`, `orchestrate_crossover_confirm.sh`
- `scripts/plot_chunk_trace.py` — trajectory summarizer/plotter
- `scripts/hotpatch_slo_tail.py` — slocvar controller patch (modes: depth/slo/slotail/slocvar)

---

## 10. Reproduction

```bash
# n=8 confirmation of the two above-crossover points (all 3 arms concurrent, one point per wave)
bash orchestrate_crossover_confirm.sh        # -> logs/crossoverconf_ANALYSIS.txt, marker crossoverconf_ALLDONE
# full 5-point analysis (realized Cs², TTFT + TPOT deltas)
python scripts/analyze_crossover.py
# controller budget trajectory
python scripts/plot_chunk_trace.py logs/2026-07-21-cross225-ours-chunktrace.csv
```

Per-server config: 14B, TP=2, mt=1024, rate=0.64, mono=16384 / chunk=512 / ours=slocvar-start-512,
pad-mean=8000, pad-cv2 ∈ {0.5,1.0,1.5,1.75,2.25}, pad-min/max=100/50000, seeds 1000+trial.

---

## 11. Next steps

1. **Isolated n=8 rerun** of the two ≥1 points (remove contention confound; lock magnitudes).
2. **Load sweep (ρ ∈ {0.4,0.6,0.75,0.9})** at fixed above-1 Cs² — tests whether higher load surfaces
   the crossover by amplifying the queueing term, and tests the PS insensitivity signature (chunk flat
   in Cs², mono growing).
3. **Reordering (SJF/SRPT-on-prefill) arm + aging** — the constructive lever: size-aware sequencing
   should beat chunk on TTFT at *every* Cs² (prefill length is known exactly). Completes the scheduling
   design space (granularity × sequencing) and extends the model to a 3-discipline M/G/1 framework
   (FCFS / PS / SRPT).
4. **Optional modeling contribution:** batch-service / state-dependent-rate queue to quantify the
   rightward crossover shift.
