# Sensitive-Regime Chunking Probe — chunking is work-conserving; the SLO controller is blind to the stall

**Date:** 2026-07-10
**Question:** Every prior experiment showed **flat TPOT** (~17 ms), so chunked-prefill
control had no lever and won nothing. Two objections remain: (a) maybe we only tested
decode-cheap regimes and never stalled decode, and (b) maybe the TPOT sensor is simply
blind. This experiment **constructs a regime where decode IS stalled by prefill** — as a
positive control for the sensor, and to test whether static or adaptive chunking has any
lever when the stall is real.

## Why the budget knob alone was not enough

On this workload prefills are tiny for two reasons: ShareGPT prompts are short, and the
replay truncates history to 256 chars (it also uses `/v1/completions` with a plain
`Human:/Assistant:` concat, no chat template — which is why 71% of outputs run to the
128-token cap). So raising `max-num-batched-tokens` alone does nothing: there is no large
prefill to chunk. To force a genuine stall we add `--pad-chars`: prepend N chars of
**unique** filler to each prompt → a large, **un-cacheable** prefill on every request.
(Disabling prefix caching would not suffice — the underlying prompts are still small; the
binding variable is prompt *size*, which padding sets directly.)

## Setup

- **Hardware/model:** single RTX 4090 (24 GB), Qwen2.5-Coder-7B-Instruct, vLLM 0.23.0,
  `max-num-seqs=32`.
- **Regime:** staggered closed-loop, **sub-saturation** (concurrency 10, 30 s stagger
  window). Deliberately closed-loop to hold a steady mixed prefill+decode batch and
  isolate prefill/decode *interference* from admission *queueing*; staggered to avoid the
  thundering-herd artifact. Verified **0 preemptions** in every arm → interference regime,
  not memory-bound (on 24 GB the two are adjacent, so this gate matters).
- **Workload:** ShareGPT, 80 convs × 3 turns = 240 requests, `max_tokens=512` (long enough
  that a decode batch is present when big prefills land), `--pad-chars 12000` (≈ 2300
  unique prefill tokens/request).
- **SLO controller:** `CHUNK_MODE=slo`, floor 512, ceiling 8192, target `SLO_MS=50`,
  EMA 0.3 on measured per-step iteration latency (from `hotpatch_slo_chunk.py`).

## Results

**(1) Baseline budget sweep (staggered c=15, no padding) — the ceiling alone does nothing:**

| arm | TPOT p95 | TTFT p95 |
|---|---|---|
| static-2048 | 17.8 | 70.5 |
| static-8192 | 17.8 | 73.6 |
| SLO 512–8192 (reorder+aging) | 17.8 | 73.8 |

Flat at both budgets; the SLO controller is a no-op (no SLO breach → pins the ceiling).

**(2) Positive control — padding reaches the sensitive region:**

| arm | TPOT mean | TPOT p95 | TPOT max | TTFT p95 | preempt |
|---|---|---|---|---|---|
| base-8192, no pad | 16.9 | 17.8 | 18.6 | 73.6 | 0 |
| base-8192, **padded** | 38.7 | **56.6** | 73.8 | 1194 | 0 |

TPOT p95 jumps 17.8 → 56.6 ms (3.2×), 90% of requests > 25 ms. The sensor is **not blind**:
it reads flat when decode is cheap and elevated when decode is stalled.

**(3) Isolation — reorder OFF, three chunk arms (padded sensitive regime):**

| arm | TPOT mean | p50 | p95 | p99 | max | TTFT p95 |
|---|---|---|---|---|---|---|
| static-8192 (monolithic prefill) | 39.3 | 39.1 | 56.3 | 65.0 | 84.0 | 1213 |
| static-512 (chopped prefill) | 37.5 | 37.4 | 52.3 | 61.2 | **63.3** | **1955** |
| SLO 512–8192 (adaptive) | 41.9 | 38.8 | 53.6 | **144.9** | **657.4** | 1194 |

**SLO budget trajectory** (`chunk=` logged every 50 steps): **143 of 147 samples at 8192**
(the ceiling), one brief dip-and-recover (512→1536→3072→7680→8192). Measured `lat_ms`
stayed **~16–20 ms** the whole run — never near the 50 ms SLO.

## Findings

1. **Positive control succeeds.** Padding stalls decode (TPOT p95 17.8→56.6 ms) with **0
   preemptions**. The flat TPOT in all other experiments is therefore a *real* decode-cheap
   property, not a dead instrument.

2. **Chunking is work-conserving on the mean; it only caps the tail, at a steep TTFT cost.**
   static-512 vs static-8192: **mean TPOT unchanged** (39.3→37.5), **max capped 25%**
   (84→63) — bought with **+61 % TTFT** (1213→1955). Chopping a big prefill spreads the same
   total decode-interference across more steps: it bounds the worst single spike but does not
   reduce the average stall. So chunking's *best case*, even when decode is genuinely
   stalled, is trading TPOT jitter for a large TTFT hit — a bad trade for almost any workload.

3. **The SLO controller's signal is blind to the stall.** It pinned at the 8192 ceiling
   because its control signal — the **EMA-smoothed mean** per-step iteration latency (~18 ms)
   — is dominated by the frequent cheap decode-only steps and **averages away** the rare
   expensive prefill steps that cause the stall. It reads "18 ms ≪ 50 ms → fine," never
   chops, and its one oscillation produced the **worst tail of all** (max 657 ms, p99 145 ms
   — worse than either fixed budget). A correct TPOT controller must key off the *tail* of
   step latency (or actual decode TPOT), not the mean.

4. **It is the controller, not reordering.** The earlier "SLO worse" result used
   reorder+aging; the reorder-**off** SLO arm here is *still* catastrophic (max 657 ms), so
   the pathology is the controller dynamics, not the reordering.

5. **No good operating point exists.** Not in the common regime (flat TPOT, no lever), and
   not even in the regime engineered to give chunking a lever (static-512 is a bad trade;
   the adaptive controller cannot even reach it). Fixing the controller's signal would at
   best recover the static-512 frontier — which is itself a bad trade.

## Conclusion

Chunked-prefill control has **no home**. Where decode is cheap it has no lever; where decode
is stalled it is work-conserving (caps tail TPOT, not the mean) at a steep TTFT cost, and
the adaptive controller's mean-latency signal is structurally blind to the interference it
is meant to relieve. This is a sharper, more complete version of the paper's null and
supplies the positive control that rebuts "you only tested decode-cheap regimes."

**Caveats.** Single trial per arm (directions are large and structural). The unique-padding
workload is a **positive control**, not a production regime — the realistic sensitive region
is genuinely long prompts (long-context/RAG/agent), untested here. On 24 GB the interference
and memory-bound regimes are adjacent; the 0-preemption gate confirms we stayed in the
former.

## Repro

Prereqs: vLLM 0.23.0 with `scripts/patch_scheduler.py` + `scripts/hotpatch_slo_chunk.py`
applied, `src/replay_sharegpt.py` at this commit (has `--pad-chars`), model + dataset in
place. Runners self-apply the patches (idempotent).

```bash
# (1) baseline budget sweep — no stall (staggered c=15)
MAX_BUDGET=2048 STAGGER=30 TRIAL=slo2048 bash scripts/run_staggered_slo.sh
MAX_BUDGET=8192 STAGGER=30 TRIAL=slo8192 bash scripts/run_staggered_slo.sh

# (2) positive control — reach the sensitive region (padded, staggered c=10, out=512)
MAX_BUDGET=8192 PAD_CHARS=12000 MAX_TOKENS=512 CONC=10 STAGGER=30 \
  NUM_CONVS=80 MAX_TURNS=3 TRIAL=sens bash scripts/run_staggered_slo.sh

# (3) isolation — reorder OFF, 3 chunk arms {static-8192, static-512, slo} + budget logging
python3 scripts/patch_chunklog_info.py      # flip ChunkCtrl[slo] log debug -> info
bash scripts/run_sensitive_probe.sh         # defaults: pad=12000 c=10 out=512 staggered

# (4) analyze
python3 scripts/analyze_sprobe.py           # TPOT/TTFT percentiles per arm
DATE=$(date +%Y-%m-%d)
grep -oE "chunk=[0-9]+" logs/${DATE}-sprobe-slo-server.log | sort | uniq -c   # budget trajectory
grep -icE preempt logs/${DATE}-sprobe-*-server.log                            # validity gate (must be 0)
```

Key knobs: `--pad-chars` (unique filler → un-cacheable large prefill), `CONC` (hold
sub-saturation / 0 preemption), `MAX_TOKENS` (decode presence), `DYNAMIC_CHUNK_SLO_MS`
(controller target). Log tags: `logs/<date>-sprobe-{static8192,static512,slo}_tsprobe.jsonl`.
