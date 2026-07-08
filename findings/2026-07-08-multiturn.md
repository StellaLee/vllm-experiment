# Multi-Turn Sequential Replay Experiment — 2026-07-08

Tests whether prefix-aware reordering + dynamic chunking + aging improve tail
latency in the multi-turn conversation regime, where each request includes the
full accumulated conversation history and KV prefix hit rates are genuinely high.

**Model:** Qwen2.5-Coder-7B-Instruct  
**Hardware:** NVIDIA RTX 4090 (24 GB)  
**Workload:** ShareGPT sequential replay (200 conversations, up to 4 turns each)  
**Script:** `scripts/run_multiturn_bench.sh`  
**Analysis:** `python3 src/analyze_multiturn.py --concurrency 15`  
**Conditions:** baseline (REORDER=0, CHUNK=0) · combined (REORDER=1, CHUNK=1) · aging T=2s (REORDER=1, CHUNK=1, AGING_THRESHOLD_MS=2000)

---

## Motivation

The BurstGPT and shuffled-ShareGPT near-saturation experiments showed ~0.6%
KV hit rates. The mechanism produced large E2EL improvements (−47%) despite no
cache exploitation, which means dynamic chunking is the primary driver. This
experiment tests the complementary regime: **what happens when KV hit rates are
genuinely high (56%)** via sequential multi-turn conversation replay?

Prior vLLM benchclient ShareGPT runs shuffled conversations randomly, destroying
cache locality. `replay_sharegpt.py` replays conversations in order: turns within
each conversation are sent sequentially, with full accumulated history as the
prompt. N conversations run in parallel (concurrency=N) via ThreadPoolExecutor.

---

## Regime Characterization

| Concurrency | GPU util | KV hit rate | Regime              |
|-------------|----------|-------------|---------------------|
| 20          | 94.6%    | ~56%        | oversaturated       |
| 15          | 95.5%    | 56.3%       | oversaturated       |

Both concurrency levels produce ~95% GPU utilization, above the 85% near-
saturation target. With multi-turn conversations, accumulated prompts grow
longer each turn, so even lower concurrency generates heavy compute load. This
experiment is therefore in the **high-load (saturated) regime**, not near-
saturation. Future work: find concurrency that targets 85%.

Despite operating above the near-saturation target, the mechanism still produces
large tail latency improvements — showing benefit even in a harder regime.

---

## Results (concurrency=15, 200 conversations × 4 turns)

### TTFT Median — flat across conditions

| Turn | n   | Baseline | Combined | Aging (T=2s) |
|------|-----|----------|----------|--------------|
| 1    | 199 | 53.5 ms  | 53.4 ms  | 54.1 ms      |
| 2    | 200 | 54.2 ms  | 54.5 ms  | 54.6 ms      |
| 3    | 200 | 54.5 ms  | 54.5 ms  | 54.8 ms      |
| 4    | 200 | 54.6 ms  | 54.6 ms  | 54.9 ms      |

Median is essentially unchanged (<1% difference). Typical requests are served
quickly regardless of condition.

### TTFT P95 — improvement grows with conversation depth

| Turn | n   | Baseline p95 | Combined p95 | Combined Δ  | Aging p95 | Aging Δ     |
|------|-----|-------------|-------------|-------------|-----------|-------------|
| 1    | 199 | 234.6 ms    | 219.1 ms    | −6.6%       | 225.2 ms  | −4.0%       |
| 2    | 200 | 125.0 ms    | 108.1 ms    | −13.5%      | 98.9 ms   | −20.9%      |
| 3    | 200 | 147.8 ms    | 87.8 ms     | **−40.6%**  | 83.8 ms   | **−43.3%**  |
| 4    | 200 | 154.5 ms    | 81.5 ms     | **−47.2%**  | 80.6 ms   | **−47.8%**  |

### KV Cache Hit Rate

| Condition    | KV hit rate | Hits    | Queries |
|--------------|-------------|---------|---------|
| baseline     | 56.3%       | 101,008 | 179,311 |
| combined     | 56.3%       | 203,264 | 361,212 |
| aging (T=2s) | 56.1%       | 203,984 | 363,381 |

Hit rate is identical across all conditions (as in all prior experiments). The
scheduler does not change how many blocks land in cache — it changes which
requests the GPU works on and when.

---

## Key Findings

### 1. P95 improvement scales with conversation depth

Turn 1 has minimal shared prefix (just system boilerplate), so reordering has
little to work with (−7%). By turn 4, each conversation's prompt includes 3
full prior exchange turns — long shared prefixes that the warm-first scheduler
can exploit aggressively. P95 drops ~47%.

This is the cleanest demonstration that **the mechanism's benefit scales with
the degree of prefix sharing**, not just GPU utilization level.

### 2. Median is unchanged — benefit is purely tail reduction

The median improvement is negligible (<1%). All requests continue to be served
quickly (54 ms); the mechanism eliminates the tail without harming typical
latency. This is the behavior of a scheduling policy that reduces stalls, not
one that trades latency type for another.

### 3. KV hit rate invariance confirmed at 56% — mechanism is scheduling, not caching

Even with 56% hit rate (vs 0.6% in BurstGPT), the combined and aging conditions
do not change the hit rate. The benefit is purely from how requests are ordered
and chunked, not from increased cache utilization.

### 4. Aging (T=2s) slightly outperforms combined at turns 2–4

| Turn | Combined Δ | Aging Δ |
|------|-----------|---------|
| 2    | −13.5%    | −20.9%  |
| 3    | −40.6%    | −43.3%  |
| 4    | −47.2%    | −47.8%  |

The gap is small (0.6–7 pp) but consistent: aging outperforms at every turn.
With 200 conversations per turn this is a plausible real effect, though
statistical confirmation requires repeated trials.

**Interpretation:** In the high-hit-rate regime, some cold requests (turn 1 of
new conversations) are deprived of GPU time by the warm-first scheduler. Aging
promotes those after 2 s, preventing starvation while preserving warm-first
benefits for conversations that are already active.

### 5. Mechanism works in the saturated regime — not just near-saturation

The BurstGPT near-sat experiment demonstrated benefit at 85% GPU util. This
experiment shows comparable benefit (~47% p95 reduction) at 95% GPU util. The
mechanism is robust across load levels when KV hit rates are genuine.

---

## Mechanism Synthesis Across All Experiments

| Experiment        | KV hit rate | GPU util | p95 improvement (combined) |
|-------------------|-------------|----------|---------------------------|
| BurstGPT near-sat | 0.6%        | 85%      | −47% (E2EL)               |
| Multi-turn c=15   | 56.3%       | 95%      | −47% (TTFT, turn 4)       |

Nearly identical p95 improvement from different mechanisms:
- **Low hit rate (BurstGPT):** Dynamic chunking prevents decode stalls → latency reduction
- **High hit rate (multi-turn):** Warm-first reordering schedules cached requests together → latency reduction

Both paths converge on ~47% tail improvement. This gives us two distinct claims:
1. Dynamic chunking alone produces large improvements regardless of cache hit rate
2. Warm-first reordering compounds this when genuine prefix sharing exists

---

## Open Items

- **Find near-saturation concurrency for multi-turn.** Concurrency=15 gives 95.5%
  GPU util. Try concurrency=8–10 to target ~85%.
- **Statistical validity.** Single trial. Need ≥3 replications to confirm
  turn-level p95 numbers.
- **Aging T sweep for multi-turn.** T=2s was chosen by analogy with BurstGPT.
  A turn-2 p95 curve over T=1s,2s,5s would reveal optimal T in this regime.
- **Decompose chunking vs reordering.** Run `REORDER=0, CHUNK=1` condition to
  isolate how much of the 47% comes from chunking vs warm-first ordering.

---

## Repro

```bash
# on remote server
ssh -p 23 root@117.50.214.139
cd /root/vllm-experiment && git pull origin main

# run all 3 conditions at concurrency=15, 200 conversations
env CONCURRENCY=15 NUM_CONVS=200 bash scripts/run_multiturn_bench.sh \
  2>&1 | tee /tmp/multiturn_c15.log

# analyze
python3 src/analyze_multiturn.py --concurrency 15
```

Tags produced:

| Tag               | Condition              |
|-------------------|------------------------|
| `mt_base_c15`     | baseline, concurrency=15 |
| `mt_comb_c15`     | combined, concurrency=15 |
| `mt_aging_c15`    | aging T=2s, concurrency=15 |
