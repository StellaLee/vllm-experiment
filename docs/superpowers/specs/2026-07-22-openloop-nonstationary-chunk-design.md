# Open-loop regime check + non-stationary dynamic-chunking — design

**Date:** 2026-07-22
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0 (V1), Qwen2.5-Coder-14B-Instruct, TP=2, GPUs 0,1
**Builds on:** [2026-07-21-longprompt-tbt-win](../../../findings/2026-07-21-longprompt-tbt-win.md),
[2026-07-22-hslo-controller](../../../findings/2026-07-22-hslo-controller.md)

Two experiments, run in sequence:

- **E1 — Open-loop Poisson regime check.** Does chunk's decode-protection win survive Poisson
  arrivals, or was it an artifact of closed-loop concurrency-20? (No harness change.)
- **E2 — Non-stationary concurrency phases.** Show the optimal prefill budget *moves* with decode
  pressure, and `hslo` tracks it while a well-tuned static budget is wrong in half the phases — the
  case that converts hslo's +11% stationary-TTFT cost into a win. (Two small harness additions.)

Both reuse the established bimodal-whale workload verbatim: single-turn, real ShareGPT first-turns,
whale-frac 0.15 (~12k-token whales) + short majority, pad-seed 1001 (whale positions **paired** across
arms), MODEL Qwen2.5-Coder-14B-Instruct, `--max-num-batched-tokens 16384 --max-model-len 16384
--gpu-memory-utilization 0.90 --max-num-seqs 48`, arms **isolated sequential** (each alone on GPUs 0,1).
Metric: pooled **P99 / max TBT** + TTFT, reconstructed per-token from the existing log fields.

---

## The load-bearing constraint (applies to both, gates E1)

The whale-freeze mechanism only produces a signal when a **live decode batch exists during the whale's
prefill iteration** — there must be decoders to freeze. This is the same failure mode as the early
≤4.5k-token runs (mono never stalled → clean tail → chunk only added overhead), but here the cause is
**too-sparse arrivals**, not too-short prompts.

By Little's law the mean in-flight population is `L = λ · W` (arrival rate × mean end-to-end latency).
Set the rate too low → `L` small → whales often prefill with no co-resident decoders → mono's TBT tail
stays clean → **a null that means "the rate was wrong," NOT "chunk doesn't win."**

**Discipline for E1:**
- Anchor the rate to reproduce the closed-loop mean concurrency, not guessed.
- Treat a clean-mono result (P99 TBT ≈ the decode baseline, no multi-second tail) as **"raise the
  rate and re-run,"** never as a negative result. The failure mode is one-directional: too-low kills
  the signal; too-high only saturates harder (whales freeze *more*) at the cost of unbounded queueing.
- **Validity gate, reported by the analyzer:** (a) realized mean/max in-flight concurrency, computed
  by sweeping the `[start, ts]` request intervals; (b) mono's P99 TBT must be ≫ its decode baseline
  (i.e. the freeze is present). A run failing (b) is marked INVALID and is not compared.

---

## Per-token wall-clock reconstruction (no logging change)

Every record already stores `ts` (completion wall-clock, `time.time()`), `latency`, `ttft`, and the
full `tbt_ms` array. Therefore for any request:

```
start_wall   = ts - latency
token[1]_at  = start_wall + ttft
token[k]_at  = start_wall + ttft + sum(tbt_ms[:k-1]) / 1000     # k >= 2
```

This gives each token's absolute emission time, so E2 can bucket pooled TBT (and each request's TTFT
at `start_wall`) into concurrency phases with **no change to the replay's logging**. The same interval
sweep `[start_wall, ts]` yields realized concurrency for E1's validity gate.

---

## E1 — Open-loop Poisson regime check

**Hypothesis.** The per-iteration whale-freeze is arrival-pattern-independent, so chunk still wins P99
TBT under Poisson arrivals — and open-loop may *amplify* it: mono's multi-second whale iteration also
stalls the scheduler, so Poisson arrivals queue behind it → mono's TTFT degrades more than in
closed-loop.

**Harness change:** none. `src/replay_sharegpt.py` already supports `--rate R` (open-loop Poisson,
`--concurrency` ignored). Reuses `analyze_longprompt.py` unchanged for the core metric; the concurrency
validity numbers come from a new small analyzer helper (shared with E2, see below).

**Rate selection (grounded, not guessed).**
1. Compute the **realized throughput** of the 07-21 closed-loop conc-20 run:
   `R_match = completions / (max(ts) − min(start_wall))` from `logs/2026-07-21-longp-b2048-t1.jsonl`
   (use the chunk-2048 arm as the reference latency regime). By Little's law, driving open-loop at
   `R_match` reproduces the same mean in-flight population that produced the original signal.
2. Run `R_match` first (3 arms). Check the validity gate. If mono's freeze is present and the queue is
   bounded (completions ≈ 200, latencies not diverging), add `R_hi ≈ 1.3 · R_match` to probe toward
   saturation. If mono comes back clean at `R_match`, **raise** the rate (1.5×, 2×) until the freeze
   appears — do not report a clean run as a result.

**Arms:** mono=16384, chunk=2048, chunk=512. Isolated sequential, pad-seed 1001, 200 convs.

**Metric:** pooled P99/max TBT + TTFT (now includes queue wait) + completions + preemptions +
realized mean/max concurrency.

**Orchestrator:** `orchestrate_longprompt_openloop.sh` — clones `orchestrate_longprompt.sh`, swaps
`--concurrency $CONC` → `--rate $RATE`, markers `longpol_ALLDONE/FAILED`, analysis
`logs/longpol_ANALYSIS.txt`.

---

## E2 — Non-stationary concurrency phases (Option A)

**Hypothesis.** Higher decode concurrency → larger decode batch → larger `decode_baseline` → the
SLO-correct prefill budget is *smaller* (`iter_time ≈ db + α·budget ≤ SLO` ⇒ `budget ≤ (SLO−db)/α`).
So the optimal budget moves down as concurrency rises. Under alternating low/high concurrency phases:
static-2048 is clean in the low phase but **violates the SLO in the high phase**; static-512 is clean
on TBT everywhere but **pays TTFT in the low phase**; `hslo` measures `db` rising/falling and **tracks
the moving optimum**, holding ≈SLO in both phases while recovering TTFT in the low phase. That flips
hslo's +11% stationary-TTFT cost into a per-phase win.

### De-risk step (run FIRST, before locking phase params)

The story is only strong if `decode_baseline` moves a **meaningful fraction of the 400ms SLO** across
the concurrency range — otherwise the optimum barely shifts. Probe `db(concurrency)` from the existing
hslo chunktrace (`logs/2026-07-22-longp-bhslo400af-chunktrace.csv`, which logs `signal_ms = db`
vs `decode_depth`) and/or a short pilot at fixed concurrencies {8, 20, 40}. **Gate:** if `db(40)` is
not a substantial fraction of the SLO (target: high-phase budget ≲ 1024 vs low-phase ≳ 2048), push the
high phase toward 48 and/or raise whale-frac in the high phase until the optimum genuinely separates.
Lock the final phase params from this probe, then run the arms.

### Harness addition 1 — phase-aware closed-loop driver

New flags on `replay_sharegpt.py`:
- `--concurrency-schedule "N1@S1,N2@S2,..."` — phase list; `Ni` = target in-flight, `Si` = phase
  seconds. Cycles the list until `--duration` elapses or convs exhausted.
- `--duration SECONDS` — total wall-clock cap for the schedule driver.

Driver: a 0.5s control loop tracks `in_flight` (incremented on conv launch, decremented on completion
via a callback). Each tick reads the current phase's target `N(t)`; while `in_flight < N(t)` and convs
remain, launch the next conv thread. On a phase **rise** this fills immediately (crisp boundary); on a
phase **fall** it simply stops launching and lets running convs drain naturally (no kills, no runaway).
If the distinct-conv pool empties before `--duration`, recycle the list with a **per-cycle uniqueness
salt** folded into the pad tag (`[req {ci}.{cycle}.{turn}]`) so refill prompts stay un-cacheable and
whale-paired. Isolated from the existing closed-loop/open-loop/stagger paths (a fourth branch), so
those are untouched.

### Harness addition 2 — phase-aware analyzer

`scripts/analyze_nonstationary.py` (new; shares the interval/token-time helper with E1): given the
schedule string and a set of arm jsonl files, bucket every output token into its wall-clock phase (via
the reconstruction above) and every request's TTFT by `start_wall`, then report **per-phase** pooled
P99/max TBT + TTFT for each arm, plus overall. For the `hslo` arm, cross-reference the chunktrace to
report per-phase controller dwell (budget median/p90) — the direct evidence that the budget tracks the
phase. Emits a per-phase table: `phase | conc | arm | TTFT | TBT-p99 | TBT-max | (hslo budget median)`.

### Phase params (provisional — finalized by the de-risk step)

Low **8** ↔ high **40** (headroom under `max-num-seqs 48`), ~50s/phase, 3 cycles (`--duration 300`).
Whale-frac 0.15 throughout, pad-seed 1001. Conv budget capped by duration with per-cycle recycling.

### Arms

mono=16384, static-2048 (stationary oracle), static-512, **hslo@400af** (both hotpatches, SLO_MS=400,
START=512, ALPHA_HW_MS=0.18 — identical to `orchestrate_longprompt_hslo_af.sh`). Isolated sequential.

**Orchestrator:** `orchestrate_longprompt_nonstationary.sh` — schedule + duration wired to all four
arms; markers `longpns_ALLDONE/FAILED`; analysis via `analyze_nonstationary.py` →
`logs/longpns_ANALYSIS.txt`.

**Success criterion.** static-2048 shows a **clear per-phase inversion** (wins low-phase TTFT, loses
high-phase TBT-p99 vs its own low-phase value crossing the SLO), while hslo holds P99 TBT ≈ SLO in
*both* phases AND beats static-2048's low-phase TTFT — demonstrated with the per-phase table and the
hslo budget-vs-phase trace.

---

## Sequencing

1. **E1** — no code; compute `R_match`, run 3 arms at `R_match`, validity-gate, add `R_hi` if clean.
   Findings appended to the long-prompt line.
2. **E2 de-risk** — `db(concurrency)` probe; lock phase params.
3. **E2** — build the two harness pieces (driver + analyzer), run 4 arms, per-phase analysis.

## Operational discipline (unchanged)

Shared/rented box: `kill_ours()` only ever kills processes whose cmdline contains `venv-vllm023`.
GitHub blocked from the box → edit locally, `rsync` up, commit/push from the Mac; local is source of
truth. Confirm the box is idle (only venv-vllm023 processes present) before any launch. Results persist
via `*_ALLDONE`/`*_FAILED` markers. Stop and wait for instruction if an auto-chaining pipeline has run
> 6h.

## Risks

- **E1 rate too low → no signal.** Mitigated by the validity gate + one-directional "raise, don't
  conclude" rule above.
- **E1 rate too high → unbounded queue,** run never completes / TTFT meaningless. Mitigated by checking
  completions ≈ 200 and latency non-divergence; back off to `R_match` if `R_hi` diverges.
- **E2 optimum doesn't move enough.** Mitigated by the de-risk `db(concurrency)` probe gating the
  phase params before the arms run.
- **E2 conv-pool exhaustion** under high-phase throughput. Mitigated by per-cycle-salted recycling.
- **Single trial** (inherited caveat). Directional first; replication (2–3 trials) deferred, consistent
  with the 07-21/07-22 findings' stated next steps.
