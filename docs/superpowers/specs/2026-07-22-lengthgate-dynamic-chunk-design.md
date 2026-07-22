# Length-Gated Dynamic Chunking — Design

**Date:** 2026-07-22
**Status:** approved (design), pending implementation plan
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0 (V1), Qwen2.5-Coder-14B-Instruct, TP=2

## Goal

Demonstrate that a **prompt-length-aware dynamic chunker** beats every static chunk size under a
workload whose **whale-fraction varies over time**: it blasts large chunks during all-short phases
(matching mono's prefill throughput) and shrinks to small chunks when a whale prefills (matching
static-512's decoder-tail protection). Static-2048 — the vLLM default and the strongest static arm —
loses in *both* phases, so the dynamic arm strictly dominates it.

## Why this axis (and why it dodges every wall we hit)

Prior work in this project established, four independent ways, that a **decode-pressure-driven**
controller (hslo: `budget = (SLO − db)/α`) cannot win on bounded-memory GPUs: the term it reacts to,
`db` (decode-step time), is pinned near-constant because the decode batch is KV-capped shallow, and no
lever (concurrency, gpu-util, model size, output length) moves it. See
`findings/2026-07-22-*` and the concurrency sweep (`logs/csweep_*`): 512 wins the tail and 2048 wins
TTFT at *every* concurrency — the optimum does **not** move on the concurrency axis.

This experiment targets a **different axis: the prompt-length mix over time.** The mechanism does not
depend on `db` at all:

- **All-short phase (no whales):** no single prefill is large enough to freeze anyone, so chunk size
  only affects prefill *batching throughput* — **larger is strictly better** (mono packs many short
  prefills per iteration; 512 throttles them under a burst). No tail cost.
- **Whale phase:** a large chunk lets the whale prefill in one giant iteration and **freezes active
  decoders** — catastrophic even with *one* decoder present. **Smaller protects.** This benefit exists
  at **any decode depth** — no deep batch required, which is why this axis is reachable on the 4090.

Static-2048 is a compromise: it under-batches in the short phase (mono is faster) and under-protects in
the whale phase (512 is tighter). A controller that goes large when the mix is uniform and small when a
whale is prefilling beats it at both ends. The concurrency sweep's "no moving optimum" is a statement
about the *concurrency* axis and says nothing about the *prompt-mix* axis; this axis is untested.

## Hypothesis / success criteria

Under an alternating **short-burst (S) ↔ whale (W)** open-loop workload, per phase:

- **Phase S (short burst, whale-frac 0):** `lengthgate TTFT ≈ mono ≤ 2048 < 512` (dynamic blasts like
  mono; 512 is throttled and pays TTFT).
- **Phase W (whales present):** `lengthgate P99/max TBT ≈ 512 < 2048 ≪ mono` (dynamic protects like
  512; mono freezes; 2048 partial).
- **Net:** the lengthgate arm beats **static-2048 in both phases simultaneously** — the dynamic win no
  static can match. This is the headline result if it holds.

**Documented failure/null modes** (each has a concrete diagnosis, not a dead end):
- Phase S not prefill-bound → `mono ≈ 2048 ≈ lengthgate` there → the pilot must raise Phase-S rate
  until 512 visibly pays TTFT.
- Over-saturation → latency becomes queue-dominated → the pilot keeps both phases sub-saturation
  (per the standing constraint: reject any operating point where the reading is queue-wait, not
  scheduling).

## Architecture / components

Five components, built on the existing whale-workload + hotpatch + phase-analysis infrastructure.

### 1. Open-loop phase-schedule driver (`src/replay_sharegpt.py`)

New capability: a per-phase schedule of `(rate, whale_frac, duration)`, driven **open-loop** (Poisson
arrivals), analogous to the existing closed-loop `--concurrency-schedule`.

- Flag: `--phase-schedule "R:F@D,R:F@D,..."` where `R`=arrival rate (conv/s), `F`=whale fraction
  (0.0–1.0), `D`=phase duration (s); plus `--duration` for total wall-clock (schedule loops).
- Each phase: draw inter-arrival times `~Exp(R)`; each conversation is a whale with probability `F`
  (whale = pad in `[whale_min, whale_max]` chars), else a short request (normal pad).
- The existing `--rate` path (single-rate open-loop) is preserved; `--phase-schedule` is a new branch
  ordered *before* `--rate` (like the concurrency-schedule branch precedes `--concurrency`).
- Each output record must allow phase reconstruction: the analyzer derives a request's phase from its
  **arrival wall-time** vs the schedule (reuse `scripts/replay_timing.phase_at`-style logic); the whale
  flag comes from the existing `pad_chars` field. No new per-record logging beyond what exists
  (`ttft`, `tbt_ms`, `pad_chars`, arrival time).

### 2. Length-gated controller (`CHUNK_MODE=lengthgate` hotpatch)

New chunk mode in the scheduler-budget hotpatch (same injection point as `hotpatch_hslo.py`). Per
scheduler step, before the prefill budget is applied:

```
if (max prefill-tokens-remaining across scheduled/waiting prefills) > LENGTHGATE_THRESHOLD
   and (number of running decode sequences) > 0:
       token_budget = LENGTHGATE_PROTECT      # default 512
else:
       token_budget = LENGTHGATE_BLAST        # default 16384
```

- Env: `CHUNK_MODE=lengthgate`, `LENGTHGATE_THRESHOLD` (default 4096 tokens — a prompt whose one-shot
  prefill would freeze decoders >~700ms at this α), `LENGTHGATE_PROTECT` (512), `LENGTHGATE_BLAST`
  (16384). `DYNAMIC_CHUNK_TRACE` emits the same `step,wall_s,depth,signal_ms,chunk` CSV for diagnostics
  (here `chunk` = the gated budget; `signal_ms` may be unused/0).
- Gates on the **actual prompt being prefilled**, so it switches **per-whale-event**, not just
  per-phase — it blasts short requests even during the whale phase. Reacts in time (the scheduler sees
  prompt lengths before/at admission), so no controller lag.
- Does **not** read `db` — deliberately, since `db` is the flat term.

### 3. Orchestrator (`orchestrate_lengthgate.sh`)

Four arms, isolated sequential, identical workload (same `--phase-schedule`, same `--pad-seed`):
`16384` (mono) / `512` / `2048` (static) / `lengthgate` (dynamic). Standard server-per-arm pattern with
the project's `kill_ours` (TERM the api_server pattern, then `-9` any GPU compute-app whose
`/proc/PID/cmdline` contains `venv-vllm023`). Markers `logs/lengthgate_ALLDONE` / `_FAILED`.

### 4. Per-phase analyzer (`scripts/analyze_lengthgate.py`)

Buckets each arm's records by phase **type** (S vs W, pooling all cycles of a type), reconstructing
phase from arrival wall-time vs the schedule. Reports:
- **Phase S:** TTFT mean/p95, throughput (completed conv/s).
- **Phase W:** pooled P99/max TBT (per-token ITL, the sharp tail), request-goodput@SLO (fraction of
  requests whose worst inter-token gap ≤ SLO; default 500ms).
- Per-phase verdict table: for each arm, its position vs the others; explicit check of the success
  criteria (lengthgate ≈ mono in S, ≈ 512 in W, both better than 2048).

### 5. Calibration pilot (`pilot_lengthgate_rates.sh`)

Runs **before** the 4-arm comparison, one static-2048 server, sweeping candidate Phase-S and Phase-W
rates, reporting per-phase TTFT/throughput and realized concurrency so we pick rates that (a) make
Phase S **prefill-bound** (512 would visibly pay TTFT — checked by also probing a 512 server at the
candidate S-rate) and (b) keep both phases **sub-saturation** (TTFT mean below a bound, e.g. ~8s).
Output feeds the orchestrator's `--phase-schedule`.

## Setup (original, for comparability)

- Model Qwen2.5-Coder-14B-Instruct, **TP=2**, GPUs 0,1, `--gpu-memory-utilization 0.90`,
  `--max-model-len 16384`, `--max-num-seqs 128` (headroom for the short burst).
- Whales: pad uniform `[44000, 50000]` chars (~12k tokens), `--max-prompt-chars 50000`. Short requests:
  `--pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000`. `--pad-seed 1001`.
  `--max-tokens 256` (whale-phase decoders present; output length is not the variable here).
- Phase schedule (starting point, pilot-tuned): `S = (rate_S, 0.0, 60s)`, `W = (rate_W, 0.20, 60s)`,
  looped over `--duration 240` (2 full S↔W cycles). `rate_S` high (prefill-bound), `rate_W` moderate.

## Risks & mitigations

- **Phase S not prefill-bound** (biggest risk to a positive result): pilot raises `rate_S` until a
  static-512 server visibly pays TTFT vs static-2048 at that rate. If unreachable sub-saturation, note
  it — the short-phase win is then genuinely small on this hardware and the result is honest.
- **Over-saturation**: pilot enforces a TTFT bound; reject rates above it.
- **Whale KV pressure at high combined load**: watch `preempt` count in server logs; a preemption
  confound invalidates the tail reading (recompute spikes) — reduce `rate_W` or whale-frac if it
  appears.
- **Controller reaction lag**: gate on **waiting** prefills too (not only running), so the budget
  shrinks the step *before* a queued whale is admitted.
- **Single trial**: as with prior findings, report as n=1 directional; replicate if the win is real.

## Non-goals

- Not touching the concurrency axis (settled: no moving optimum there).
- Not an SLO-derived interpolating controller (binary blast/protect is the clean demonstration; the
  protect level stays a tunable knob for later).
- Not the 7B / small-model / deep-batch direction (settled: unreachable on bounded GPUs).
