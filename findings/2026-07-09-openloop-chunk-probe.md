# Open-Loop Chunking Probe — does any chunk-budget policy beat a static budget?

**Date:** 2026-07-09
**Question:** In the open-loop (Poisson) regime, does *any* prefill-chunk-budget
policy beat a static high budget? This decides whether we iterate the SLO
controller or pivot to a characterization paper.

## Setup

- Workload: ShareGPT multi-turn replay, 200 convs × 4 turns, max_tokens=128,
  `--rate` Poisson arrivals, `max-num-seqs=32`. Model: Qwen2.5-Coder-7B-Instruct.
- **Reorder OFF** (`PREFIX_REORDER=0`) in all arms — chunk-only, to isolate
  chunking from the soft-aging starvation. Verified: no "reordering enabled" log
  line fired; the reorder/aging block is gated by a single `if self._prefix_reorder`.
- Three arms:
  - `base2048`  — static high budget (`DYNAMIC_CHUNK=0`, `--max-num-batched-tokens 2048`)
  - `sarathi512` — static stall-free budget (`DYNAMIC_CHUNK=0`, `--max-num-batched-tokens 512`)
  - `slo`       — SLO-feedback adaptive budget 512..2048 (`CHUNK_MODE=slo`,
                  `DYNAMIC_CHUNK_MIN=512`, `DYNAMIC_CHUNK_SLO_MS=50`). New controller,
                  see `scripts/hotpatch_slo_chunk.py`.
- Rates: 3 (1 trial), 4 (3 trials), 5 (3 trials). Trials differ only in Poisson
  arrival jitter (workload otherwise deterministic, temperature 0).

## Results (p95, means across trials)

```
TTFT p95 (ms)        base2048   sarathi512        slo       between-trial spread
 rate=3                 83.7    185.9(+122%)   87.3(+4%)
 rate=4                 2025    2208 (+9%)     1852(-9%)     base2048 973–3130 (3x)
 rate=5                 4320    4232 (-2%)     4973(+15%)    slo 3080–6001 (2x)

E2EL p95 (s)
 rate=4                  4.5     4.7 (+5%)      4.4 (-3%)
 rate=5                  6.8     6.7 (-2%)      7.5 (+10%)

TPOT p95 (ms)         ~39–42 FLAT across every rate, trial, and policy
```

## Findings

1. **No open-loop chunk win — robust, replicated.** `base2048` is best-or-tied on
   the means at every rate. Every apparent chunk-policy advantage sits inside a
   ±2–3× between-trial variance and disappears on replication. The eye-catching
   rate=5 trial-1 `slo` result (−28% TTFT) did **not** replicate: trials 2–3 were
   +40%/+58%, mean **+15% worse**. It was arrival-jitter noise.

2. **Mechanistic why: decode never stalls in open-loop.** TPOT p95 is flat at
   ~41 ms across all rates/trials/policies (±0.9 ms). NOTE: these ~41 ms figures
   use a word-count token proxy; real per-token latency is ~20 ms (server
   `inter_token_latency`, see 2026-07-09-measurement-validation.md). The absolute
   value is off but the *flatness* — the actual claim — holds either way, and is
   server-confirmed via `request_decode_time`. Chunking exists to protect
   decode from prefill stalls; with no stall, it has no lever. Under overload the
   tail is dominated by *queueing*, which is highly sensitive to arrival jitter —
   hence the large trial variance and the noisy sign flips.

3. **The SLO controller is validated as correct, not as a winner.** It never
   collapses (contrast the old depth controller: T1 1452 ms at rate=3). At rate=3
   it stays within 4% of baseline while dominating fixed-Sarathi (which pays
   +122% TTFT for chopping long multi-turn prompts). It correctly grows to the
   ceiling when decode has latency headroom. The redesign works; there's just
   nothing to win in this regime. (Minor: under deep overload the wall-clock SLO
   signal picks up queueing noise and the controller can oscillate — a limitation
   to note, not a headline.)

## Decision

**Pivot to the characterization framing.** The chunking benefit measured earlier
(closed-loop c=15, GPU ~95%: T3 −45%, T4 −47% TTFT) is real but **specific to
synchronized closed-loop saturation**. It does not transfer to open-loop, because
open-loop at rate 3–5 is decode-bound, not stall-bound. The paper's defensible
contribution is the *regime characterization* — when static chunk budgets and
warmth-priority scheduling help vs. break, with the mechanism-level explanation —
which none of PRISM / Sarathi / tail-aware provide.

## Repro

```
# on server (117.50.214.139:23, /root/vllm-experiment)
bash scripts/run_probe_openloop.sh              # rate=3
RATE=4 TRIAL=2 bash scripts/run_probe_openloop.sh
RATE=5 TRIAL=3 bash scripts/run_probe_openloop.sh
python3 scripts/probe_agg.py r4                  # aggregate across trials
```

Logs: `logs/2026-07-09-probe-{arm}_{r3,r4,r5}_{t1,t2,t3}.jsonl`.
