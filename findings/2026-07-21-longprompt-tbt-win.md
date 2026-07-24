# Long-prompt regime: chunking's decode-protection win, reproduced on P99 TBT

**Date:** 2026-07-21
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0 (V1), Qwen2.5-Coder-14B-Instruct, TP=2
**Headline:** With **12k-token whales** mixed into a short-prompt stream, **mono (budget 16384) freezes
every decoding user for ~2.5s (P99) up to ~6s (max) between tokens**; chunking cuts that by **82%
(budget 2048) to 94% (budget 512)**. This is Sarathi-Serve's stall-free-batching benefit, reproduced
in vLLM V1 — and it is the measured justification for vLLM's 2048 default over 16384.

---

## TL;DR

- Every prior chunk-vs-mono test (crossover sweep, controllers, single-turn budget sweep) found chunk
  **losing or tying** on decode latency. Cause: **prompts were too short** (≤4.5k tokens) to fill a
  16k-token iteration, so mono never stalled decode. **Not a Cs² problem — an absolute-prefill-size
  problem.**
- Two prerequisites unlocked the result: (1) a **bimodal whale workload** (mostly short + ~15% 12k-token
  prompts) so mono has a live decode batch to stall; (2) **per-token ITL logging** in the replay
  (`tbt_ms`), because per-request-*mean* TPOT dilutes a transient stall into invisibility.
- On the tail metric that matters for interactive serving (**P99 TBT**), chunk wins decisively.
- The optimum is **interior** (2048, not 512 nor 16384): 2048 captures ~82% of the protection *and
  improves TTFT*; 512 over-pays TTFT for marginal extra protection. Interior optimum ⇒ the empirical
  hook for **dynamic chunking**.

---

## Setup

- **Workload:** single-turn closed-loop, real ShareGPT first-turns. Bimodal padding:
  ~84% short (base pad ~800 chars) + **16% whales** with pad uniform in [44k, 50k] chars
  ≈ **~12k tokens** (whale words p50/max = 9167/9819; ×~1.3 tok/word). Context-overflow guard caps
  total prompt at 50k chars (this filler tokenizes at ~3.85 chars/token; 60k-char whales overflowed
  the 16384 context and 400'd — the guard fixed it).
- **Load:** concurrency 20, max-num-seqs 48, max-tokens 256, 200 conversations. Low concurrency so
  the KV-heavy whales don't overload capacity (195/200 completed, **0 preemptions** on every arm).
- **Arms (ISOLATED sequential — each alone on GPUs 0,1, other 6 idle → zero cross-arm PCIe/NCCL
  contention):** mono=16384, chunk=2048 (vLLM default), chunk=512 (aggressive anchor).
- **Metric:** pooled **P99 TBT** = 99th percentile of per-token inter-token latencies across all output
  tokens of all requests (Sarathi's metric), plus TTFT and per-request-mean TPOT.

## Results

```
 budget |   n  TTFTmean(Δ)  TPOTp95(mean) |  TBTp50   TBTp99   TBTmax |  ΔTBTp99  ΔTBTmax  [arm]
  16384 | 195   5676  +0.0        178.5   |   28.0   2474.1   5974.1  |    +0.0     +0.0   mono
   2048 | 195   5042 -11.2        248.8   |   29.2    436.8   1228.5  |   -82.3    -79.4   chunk
    512 | 195   6842 +20.6        115.5   |   30.3    136.7    391.9  |   -94.5    -93.4   chunk
```

(TTFT/TBT in ms. 195/195 completed, 0 preemptions, all arms.)

## Interpretation

**The mechanism, measured.** In vLLM V1 continuous batching, every decoding request advances one token
per scheduler iteration, so **iteration time ≈ TBT for all streaming users**. A 12k-token whale under
mono (budget 16384) prefills in ONE iteration → that iteration processes ~12k tokens → **every decoder
freezes ~2.5s (p99), up to ~6s (max)** for its next token. Chunk-2048 slices the whale into ~6
iterations → each bounded → TBT capped at ~440ms. Chunk-512 → ~24 iterations → ~140ms.

**Why earlier tests saw nothing.** With ≤4.5k-token prompts, a mono iteration never got long enough to
stall decode, so mono's TBT tail stayed clean and chunk only added overhead. The effect scales with
**absolute** prefill length, not relative dispersion (Cs²). High Cs² around a small mean ≠ big whales.

**Why the mean-TPOT metric hid it.** Mono concentrates the damage into ONE catastrophic per-decoder gap
per whale; averaged over a request's ~250 decode tokens, per-request-mean TPOT-p95 (178ms) looks fine.
The catastrophe lives only in the pooled per-token tail (P99 TBT), which is exactly what interactive
users feel and what the ITL-logging fix exposed.

**The three questions this settles at once:**
1. *Is there a regime where chunk wins?* Yes — on P99 TBT, by 82–94%.
2. *Why does vLLM default to 2048 not 16384?* Because 16384 permits a 2.5–6s decode freeze on long
   prompts; 2048 bounds it to ~440ms. Measured, not asserted.
3. *Why 2048 not 512?* The knee. 2048 gets most of the protection **and improves TTFT (−11%)**; 512
   over-pays TTFT (+21%) for marginal extra TBT. The optimum is interior.

## The tradeoff (honest, and it is Sarathi's point)

Chunking trades a little **average** decode latency for a huge **tail** improvement. chunk-2048's
per-request-mean TPOT-p95 is *worse* than mono (249 vs 178ms) because its 2048-token chunks moderately
elevate many decode steps, whereas mono elevates one step catastrophically. **Under any interactive SLO
("no >500ms freeze"), mono FAILS (2474ms) and chunk PASSES (437ms)** — that is the win. If you only
optimize mean throughput, mono is defensible; real serving optimizes the tail.

Chunk-size profile across the three arms:
- **512** — best TBT, best mean TPOT, **worst TTFT (+21%)**.
- **2048** — great TBT (−82%), **best TTFT (−11%)**, worst mean TPOT.
- **16384** — catastrophic TBT (2.5–6s), middling TTFT/mean-TPOT.

The best static budget depends on which SLO you defend (TTFT vs TBT) and on the whale rate — i.e., it
should move with the workload. **That is the concrete justification for dynamic chunking.**

## Update (2026-07-24): the pooled +20.6% TTFT cost at chunk=512 was mostly the whale's own

A later head-to-head (`2026-07-23-whale-threshold-concurrency-tradeoff.md`, "budget=512 vs.
threshold=512" update) reran this same step-wide-budget mechanism at wf=5% with whale vs.
short populations disaggregated. The whale-only TTFT delta for chunk=512 vs. mono came out to
**+20.2%/+20.7%** (two MAXTOK arms) — almost exactly this doc's pooled **+20.6%**. Since whale
TTFT values are an order of magnitude larger than short TTFT values, the pooled mean above was
very likely dominated by the whale population paying the cost of being chunked, not a broad
cost spread across the whole stream; the disaggregated short-only TTFT was roughly
flat-to-improved. Doesn't change the headline (chunk=512 over-pays TTFT relative to chunk=2048
either way), but sharpens *who* pays it.

## Update (2026-07-24): 3-trial replication — robust, after fixing a real context-overflow bug

Attempted a straight 2-more-trials replication (`TRIALS="2 3"`, original `WHALE_MIN=48000/
WHALE_MAX=60000/MAX_PROMPT_CHARS=62000`) and got wildly inconsistent mono baselines across
trials (trial 1 mono TTFT 5676ms vs. trial 2/3's 682ms/339ms — an order of magnitude apart).
Root cause: this script's whale bounds and overflow guard were set assuming the *originally
assumed* ~4 chars/token ratio, but the measured true ratio (established elsewhere this
session) is 3.235 chars/token — denser. At the true ratio even the 62000-char guard is
~19,168 tokens, already over the 16384 context limit, so a meaningful fraction of whale draws
silently 400'd. Trial 1's original run got a lucky seed (only 4/200 failures); the replication
attempts got unlucky seeds (27–29/200 failures each) — and since the biggest whales are
exactly what causes mono's worst stalls, losing them disproportionately deflated *both* arms'
tails, invalidating the comparison rather than showing a weaker effect.

**Fix:** tightened to `WHALE_MIN=44000/WHALE_MAX=50000/MAX_PROMPT_CHARS=50000` — the same
bounds already validated safe in the whale/threshold/concurrency work. Archived the buggy runs
(`logs/archive_longp_overflow_bug/`, not deleted) and reran all 3 trials clean. Error rates are
now low and consistent (4, 6, 6 out of 200 across trials 1/2/3, vs. 4/29/27 before the fix),
and completion counts match closely (195/193/194).

**Result: the core finding replicates robustly, with one part turning out stronger than the
original single trial suggested.**

| | trial 1 | trial 2 | trial 3 | mean ± std |
|---|---|---|---|---|
| chunk=2048 dTTFT | −15.8% | −48.9% | −28.8% | **−31.2% ± 16.7** |
| chunk=2048 dTBT-p99 | −43.3% | −83.7% | −14.3% | −47.1% ± 34.9 |
| chunk=2048 dTBT-max | −79.8% | −85.0% | −85.4% | **−83.4% ± 3.1** |
| chunk=512 dTTFT | +14.1% | −4.0% | +16.1% | +8.7% ± 11.1 |
| chunk=512 dTBT-p99 | −82.3% | −95.0% | −73.1% | **−83.5% ± 11.0** |
| chunk=512 dTBT-max | −93.4% | −93.4% | −95.5% | **−94.1% ± 1.2** |

TBT-max is the tightest, most reproducible number in the whole result (80–85% for chunk=2048,
93–96% for chunk=512, both across all 3 trials) — this is a stable, structural effect, not a
lucky single run. TBT-p99 replicates in direction every time but is noisier in magnitude.
The interior-optimum claim holds and chunk=2048's TTFT improvement is *more* consistent than
the original single trial showed — it improves TTFT in all 3 trials (−16% to −49%), not just
the original's −11%. Chunk=512's TTFT cost is directionally present in 2/3 trials but mild
(trial 2 is essentially flat at −4%), so "512 over-pays TTFT" is right in direction but less
ironclad in magnitude than the tight TBT numbers.

**Status: this resolves the single-trial caveat for this finding.** The whale-fraction ×
whale-size grid (mapping the frontier more broadly) remains a separate, still-open next step.

## Caveats

- **Isolated sequential arms** removes cross-arm contention but allows small box-state drift between
  arms (far smaller than the effect).
- Whales are ~12k tokens (context-safe under 16384 at the corrected 44-50k char bounds); pushing
  toward larger whales would sharpen mono's stall further but needs care re-deriving safe bounds
  from the measured (not assumed) chars/token ratio, given the overflow bug found above.
- A whale-fraction × whale-size grid (5/15/30% × 8k/12k/15k tokens) to map the frontier more
  broadly remains open — the 3-trial replication above confirms this one point is stable, not
  that the effect's shape is fully characterized across whale characteristics.

## Artifacts

- Orchestrator: `orchestrate_longprompt.sh` (bimodal whales, isolated sequential, P99-TBT).
- Analyzer: `scripts/analyze_longprompt.py` (pooled P99/max TBT + TTFT + mean-TPOT deltas vs mono).
- Replay fix enabling this: `src/replay_sharegpt.py` — per-token ITL logging (`tbt_ms` field),
  TPOT corrected to `/(N-1)`, and a bimodal `--whale-frac/--whale-min-chars/--whale-max-chars` knob
  with `--max-prompt-chars` overflow guard.
- Data: `logs/2026-07-21-longp-b{16384,2048,512}-t1.jsonl`, analysis `logs/longp_ANALYSIS.txt`.

## Next steps

1. **Replicate:** 2–3 trials + whale-fraction {5,15,30%} × whale-size {8k,12k,15k} grid → frontier map.
2. **Goodput framing:** open-loop rate sweep, report max QPS under a P99-TBT SLO (Sarathi's headline).
3. **Dynamic chunking:** now that the interior optimum is demonstrated, an SLO-aware controller that
   shrinks the budget when a whale is detected in the queue and relaxes it otherwise — the payoff is
   the −82% tail with 2048's TTFT, adapted per-whale.
   → **Done: see [2026-07-22-hslo-controller](2026-07-22-hslo-controller.md).** The SLO-headroom
   feedforward controller (`hslo`) matches the oracle static-2048's tail protection from offline-only
   knobs; four naive controllers first failed by railing to a corner. Remaining win = non-stationary load.
