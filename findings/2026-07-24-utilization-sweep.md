# Utilization sweep: the admission-side benefit's *absolute* reduction grows with load, as Eq. genuine predicts

**Date:** 2026-07-24
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0 (V1), Qwen2.5-Coder-14B-Instruct, TP=2
**Workload:** `orchestrate_cs2_whale_threshold_sweep.sh`, `WHALE_FRACS='0.05' CONCS='26 32 38'
THRESHOLDS='0 512'` — same whale/short bimodal mix as the whale-threshold-concurrency work
(whale 44-50k chars, short mean 800 chars, pad-cv2=0.5, pad-seed=1001, max-num-seqs=48,
single-turn, max-tokens=256, NCONV=200, budget fixed 16384). Concurrency swept 26/32/38,
reusing existing conc=20 data (`2026-07-23-cs2wt-*-w05-c20-t1.jsonl`) as the low end of the
range. wf held fixed at 5% (Cs² roughly constant) so concurrency is the only thing varying —
a utilization ($\rho$) proxy, replacing the abandoned direct Cs² sweep (which was confounded
by low utilization at the open-loop rates needed to avoid overload).

**Headline: Eq. genuine predicts an *absolute* reduction term
($\mathbb{E}[S_{\text{prefill}}]\cdot\rho/(1-\rho)\cdot(C_s^2-1)/2$), not a percentage — and
once read that way, the prediction holds cleanly.** A first look at *relative* deltas
(threshold=512 vs. mono, percent change) looked exactly like the earlier abandoned Cs² sweeps:
noisy, non-monotonic, no clean trend (-60%, -48%, -50%, -64% across conc 20→38). But the
*absolute* reduction in S:TTFT mean grows monotonically and cleanly as concurrency rises:

| conc | mono S:TTFT mean | thr=512 S:TTFT mean | **absolute reduction** |
|---|---|---|---|
| 20 | 1086ms | 432ms | **654ms** |
| 26 | 1702ms | 887ms | **814ms** |
| 32 | 2495ms | 1249ms | **1247ms** |
| 38 | 3558ms | 1278ms | **2280ms** |

Mono's own baseline TTFT climbs steadily across the sweep (1086→1702→2495→3558ms), confirming
utilization is genuinely rising, not just noise. The absolute admission-side benefit tracks
that rise almost 1:1 in magnitude by the end (654ms → 2280ms, a 3.5× increase over a load
range where mono's own queue-wait grew about 3.3×) — exactly the qualitative shape
$\rho/(1-\rho)$ predicts: the benefit should diverge as load approaches saturation, not stay a
fixed fraction of a baseline that's itself growing.

**Tail percentiles are noisier but trend the same way.** p95 absolute delta: 4677ms → 6957ms →
5426ms → 11507ms — a dip at conc=32 before a large jump at conc=38, consistent with thinner
percentile statistics (n=189 short population per point, single trial) rather than a
contradicting signal; the endpoints (conc=20 vs. conc=38) still show a clear ~2.5× growth.

**Why this succeeds where the earlier Cs² sweeps failed:** those sweeps varied $C_s^2$ at a
*fixed, low* open-loop rate (0.05, calibrated to avoid overload) — meaning $\rho/(1-\rho)$
stayed pinned near zero throughout, so there was rarely more than one competing request
regardless of dispersion, and the genuine term had nothing to scale with. This sweep instead
holds $C_s^2$ roughly fixed and varies $\rho$ directly via closed-loop concurrency, which is
exactly the free variable Eq. genuine's gating structure is actually sensitive to at
accessible load levels.

## Caveats

- Single trial per concurrency point. The percentile-level noise (the conc=32 p95 dip) is a
  reminder that a repeat-trial pass would help confirm the mean-level trend is not itself a
  single-run artifact, even though it looks clean.
- Concurrency is a proxy for utilization, not a direct measurement of $\rho$ — no attempt was
  made here to compute a numeric $\rho$ per point (e.g., via sustained token throughput over
  the budget, the operational definition used elsewhere in the queuing-lens section). A tighter
  version of this result would fit the growth curve against an estimated $\rho$ rather than
  raw concurrency.
- conc=38 is close to but still below the conc=40 point already confirmed overloaded in
  earlier work — worth flagging that this sweep may be running close to the edge of the
  sub-saturation regime at its top end, not comfortably inside it.
- Whale-only TTFT was not reanalyzed here (only short-population S:TTFT); the whale's own
  cost at rising concurrency is a natural follow-up given the whale/threshold/concurrency
  findings already establish it degrades with load too.

## Next steps

1. Repeat-trial the mean-level absolute-reduction trend to confirm it's stable, not a
   single-run artifact (the percentile noise suggests real run-to-run variance exists here).
2. Fit against an estimated $\rho$ rather than raw concurrency, to make the quantitative tie
   to Eq. genuine explicit rather than qualitative.
3. Feeds `docs/2026-07-24-mlsys-main-track-plan.md` Phase 2 — this is meaningfully stronger
   evidence for the queuing-lens account than anything the Cs² sweeps produced, and is worth
   promoting into the paper (likely §5.5 or a new subsection) once repeated.
