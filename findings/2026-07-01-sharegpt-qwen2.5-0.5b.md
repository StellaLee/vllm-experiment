# Baseline Profile - Qwen2.5-0.5B-Instruct (ShareGPT multi-turn)
**Date:** 2026-07-01 13:19:48
**Model:** /model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct
**GPU:** NVIDIA GeForce RTX 4090 (450W TDP)
**vLLM:** 0.23.0  |  **Trace:** ShareGPT multi-turn  |  **Requests:** 27

## GPU Metrics
| Metric | Value |
|--------|-------|
| Duration | 10.0 s |
| Total Energy | 470.8 mWh (0.4708 Wh) |
| Avg Power | 169.5 W |
| Peak Power | 252.9 W |
| Idle Power (first 5s) | 90.1 W |
| Avg GPU Util | 55.1% |
| Peak GPU Util | 100% |
| Avg Mem Used | 22750 MiB |
| Peak Mem Used | 22750 MiB |
| Avg Temp | 42.9 C |
| Peak Temp | 48 C |

## Request Latency & TTFT
| Metric | P50 | P95 | P99 | Min | Max |
|--------|-----|-----|-----|-----|-----|
| Latency (s) | 0.230 | 0.233 | 0.239 | 0.227 | 0.242 |
| TTFT (s)    | 0.010 | 0.013 | 0.014 | 0.009 | 0.015 |

## Per-Turn Breakdown (KV Cache Signal)
_Lower latency on later turns indicates KV prefix cache hits._

| Turn | Requests | Avg Latency (s) | Avg TTFT (s) | Avg Prompt Words |
|------|----------|-----------------|--------------|------------------|
| 1 | 10 | 0.231 | 0.010 | 33 |
| 2 | 7 | 0.230 | 0.011 | 98 |
| 3 | 6 | 0.230 | 0.011 | 188 |
| 4 | 4 | 0.231 | 0.011 | 186 |

**Conversations replayed:** 10  |  **Total requests:** 27

## Raw Log Paths
- GPU timeseries: `/root/vllm-experiment/logs/2026-07-01-sharegpt-gpu.json`
- Trace detail: `/root/vllm-experiment/logs/2026-07-01-sharegpt-detail.jsonl`
