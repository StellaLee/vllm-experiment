# Measurement Validation — client-side counters vs vLLM /metrics

**Date:** 2026-07-09
**Question:** Are the custom client-side measurements in `replay_sharegpt.py`
(TTFT / TPOT / E2EL) trustworthy? Cross-check them against vLLM's server-side
`/metrics` histograms.

## Setup

Fresh server (so `/metrics` counters reflect only this run), ShareGPT 100 convs
× 4 turns, `--rate 4` open-loop (chosen to force nonzero queue time so all
components are exercised), `max-num-seqs 32`. Snapshot `/metrics`, compare with
`scripts/compare_metrics.py`. Means are exact (`sum/count`); server percentiles
are coarse histogram-bucket upper bounds. Scripts: `scripts/run_metrics_validation.sh`,
`scripts/compare_metrics.py`.

## Results (means, 400 requests)

| Metric | client | server (`/metrics`) | Δ |
|--------|--------|---------------------|---|
| TTFT   | 344.9 ms | 340.8 ms (`time_to_first_token`) | +1.2% |
| E2EL   | 2482 ms | 2369 ms (queue+prefill+decode; no e2e family in 0.23.0) | +0.9% |
| decode (E2EL−TTFT) | 2138 ms | 2136 ms (`request_decode_time`) | +0.1% |
| TPOT (word-proxy)  | 49.8 ms | 19.6 ms (`inter_token_latency`) | **+154%** |
| TPOT (real tokens, after fix) | 22.1 ms | 19.7 ms | +12.5% |

## Findings

1. **Latency measurements are faithful.** TTFT, E2EL, decode all match server
   truth within ~2%. The client-side counter (including the queue-wait component
   folded into TTFT, and the decode/queue split) is validated. Safe to keep for
   the per-turn / per-conversation analysis that `/metrics` cannot produce.

2. **TPOT word-count proxy was badly wrong and is now fixed.** The original
   denominator was whitespace word count; on Qwen-*Coder* (code-dense output,
   few words per token) this was **2.5× off and right-skewed** (client mean 49.8
   > client p95 42.5). Fixed by requesting `stream_options.include_usage` and
   dividing by real `completion_tokens` (`tokens_exact=true` on 400/400 records).
   Residual +12.5% is client-side observation overhead (Python threads timing
   token *arrival* under concurrent SSE reads, vs server token *emission*) —
   expected, small, acceptable.

3. **`/metrics` histogram percentiles are unreliable for tails.** Decode p95:
   true (raw) = 2.68s, but the histogram buckets jump 2.5s → 5.0s and the p95
   crossing lands in that bucket, so both upper-bound (5.0s) and linear
   interpolation (~4.6s) overestimate by ~1.7–1.9×. vLLM buckets are fine below
   2.5s but coarse above. **Accurate p95/p99 requires per-request records**, not
   histogram buckets — a second, independent reason to keep the custom counter.

4. **Server gives a decomposition the client can't.** TTFT (341 ms) splits into
   `queue_time` 281 ms + `prefill_time` 44 ms — i.e. TTFT is ~80% queue wait,
   not prefill. The client only sees their sum.

## Conclusion

Custom per-request counter is the **system of record** (now accurate on all four
metrics, with `tokens_exact` for auditability); `/metrics` is the **auditor** and
the source for the queue/prefill split, real-token TPOT, throughput, and KV hit
rate. Absolute per-token claims should cite server `inter_token_latency` (~20 ms);
the client is authoritative for TTFT/E2EL and all tail percentiles.

## Repro

```
bash scripts/run_metrics_validation.sh          # rate=4, 100 convs, fresh server
# -> logs/<date>-metricsval-client.jsonl, -metrics.txt; prints the comparison
```
