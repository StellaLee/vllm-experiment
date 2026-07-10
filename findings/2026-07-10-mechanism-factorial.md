# Mechanism Factorial — the closed-loop herd win is CHUNKING, not reordering

**Date:** 2026-07-10
**Question:** In every closed-loop experiment so far, reordering and chunking were run
**bundled** as `combined`; we never isolated `reorder-only` vs `chunk-only`. The paper
(§3/§4) attributes the herd TTFT win to reordering ("warm continuations jump the queue").
Is that true? Decompose the herd win into its parts, and test whether each part's win
collapses under staggering.

## Setup

Real ShareGPT multi-turn (prefix caching ON, **no padding** — so prefix warmth actually
exists, unlike the sensitive-regime probe), original §3/§4 config: single RTX 4090,
Qwen2.5-Coder-7B, vLLM 0.23.0, c=15, 120 convs × 4 turns, `max_tokens=128`,
`max-num-batched-tokens=2048`, depth-mode chunk controller (so `combined` reproduces the
published win). Factorial: **{baseline, reorder, chunk, combined} × {herd, staggered-30s}**,
8 cells, 1 trial each. **0 preemptions in all 8 cells.**

- baseline = `PREFIX_REORDER=0 DYNAMIC_CHUNK=0`
- reorder  = `PREFIX_REORDER=1 DYNAMIC_CHUNK=0 AGING_ALPHA=0.3`
- chunk    = `PREFIX_REORDER=0 DYNAMIC_CHUNK=1 DYNAMIC_CHUNK_HOLD=3` (depth-mode)
- combined = reorder + chunk

## Results — per-turn TTFT p95 (ms, % vs baseline)

**HERD (synchronized t=0 start):**

| Turn | baseline | reorder | chunk | combined |
|------|----------|---------|-------|----------|
| T1 | 185.2 | 251.6 (+36%) | 208.4 (+13%) | 191.1 (+3%) |
| T2 | 145.8 | 136.7 (−6%) | 108.7 (−25%) | 109.9 (−25%) |
| T3 | 159.2 | 167.5 (+5%) | **87.1 (−45%)** | 98.4 (−38%) |
| T4 | 118.2 | 116.8 (−1%) | **87.5 (−26%)** | 81.4 (−31%) |

**STAGGERED (starts spread over 30 s):**

| Turn | baseline | reorder | chunk | combined |
|------|----------|---------|-------|----------|
| T1 | 93.8 | 92.9 (−1%) | 106.1 (+13%) | 109.1 (+16%) |
| T2 | 71.9 | 68.4 (−5%) | 82.0 (+14%) | 80.9 (+13%) |
| T3 | 74.4 | 74.3 (−0%) | 76.2 (+2%) | 75.2 (+1%) |
| T4 | 88.0 | **69.7 (−21%)** | 80.4 (−9%) | 71.6 (−19%) |

## Findings

1. **The herd win is chunking, not reordering.** `chunk-only` reproduces the headline
   later-turn win (T3 −45%, T4 −26%); `reorder-only` is neutral-to-harmful (T3 +5%,
   T4 −1%, and T1 **+36% worse**). `combined ≈ chunk` under herd — reordering adds
   essentially nothing. The paper's "warm continuations jump the queue" mechanism is
   **wrong**; the driver is the depth controller shrinking the prefill budget under the
   herd's deep queue, which interleaves small prefill chunks across the 15 piled-up
   first-turns and spreads first-token latency more evenly.

2. **The chunk win is an arrival artifact — it collapses *and reverses* under staggering.**
   Staggered `chunk`: T3 +2% (win gone) and it *hurts* T1/T2 (+13%/+14%) by over-chopping
   when there is no pile to spread. So chunking's entire TTFT value is recovering the
   synchronized-burst artifact — the coordinated-omission thesis, re-attributed from
   reorder to chunk. (Consistent with chunk being null in open-loop and work-conserving in
   the sensitive regime; see `2026-07-10-sensitive-regime.md`.)

3. **Reordering has no herd win, only a modest staggered-T4 win.** reorder-only gives
   **T4 −21%** under staggering (corroborated by combined's T4 −19%) — genuine
   warm-continuation priority on the last turn in a non-synchronized regime — but it is one
   turn, ~20%, single-trial. Its real effect is small and lives in the *staggered* regime,
   not the herd.

4. **Under staggering, adding chunk to reorder HURTS.** combined inherits chunk's early-turn
   harm (T1 +16%, T2 +13%) while keeping reorder's T4 win — i.e., in the realistic regime
   the chunk controller is a net liability.

## Conclusion

The closed-loop headline win is a **chunking** artifact of the synchronized start, not a
reordering effect. The paper's arrival-model / coordinated-omission thesis **survives and
gets cleaner** (a single mechanism — prefill chunking spreading a synchronized first-token
pile — that vanishes and backfires once starts are de-synchronized), but the **mechanism
attribution in §3/§4/§6 must change from reordering to chunking**, and reordering is
demoted from "the win" to "a minor staggered-T4 effect."

**Caveats.** Single trial per cell. The chunk-drives-the-herd-win result is large and
unambiguous (−45% vs +5%); the reorder staggered-T4 effect (−21%) is smaller and should be
replicated (2–3×) before the paper leans on it. Depth-mode chunk controller used to match
the original `combined`; the SLO controller was not run here.

## Repro

```bash
# 8-cell factorial: {baseline,reorder,chunk,combined} x {herd,staggered}
bash scripts/run_mechanism_factorial.sh          # defaults: c=15 convs=120 turns=4 budget=2048

# analyze per-turn TTFT p95 (herd, then staggered)
P=logs/$(date +%Y-%m-%d)-mfact
python3 src/analyze_ablation.py "baseline:${P}-baseline_herd_t1.jsonl" \
  "reorder:${P}-reorder_herd_t1.jsonl" "chunk:${P}-chunk_herd_t1.jsonl" \
  "combined:${P}-combined_herd_t1.jsonl"
python3 src/analyze_ablation.py "baseline:${P}-baseline_stag_t1.jsonl" \
  "reorder:${P}-reorder_stag_t1.jsonl" "chunk:${P}-chunk_stag_t1.jsonl" \
  "combined:${P}-combined_stag_t1.jsonl"
grep -icE preempt ${P}-*-server.log     # validity gate: all 0
```

Raw logs for this run: `logs/2026-07-10-mfact-*.jsonl` (8 cells, 480 records each).
