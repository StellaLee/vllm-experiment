# Baseline Profile - Qwen2.5-0.5B-Instruct
**Date:** 2026-06-30 18:15:57
**Model:** /model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct
**GPU:** NVIDIA GeForce RTX 4090 (450W TDP)
**vLLM:** 0.23.0  |  **Requests:** 45

## GPU Metrics
| Metric | Value |
|--------|-------|
| Duration | 396.5 s |
| Total Energy | 6068.7 mWh (6.0687 Wh) |
| Avg Power | 55.1 W |
| Peak Power | 240.4 W |
| Idle Power (first 5s) | 48.4 W |
| Avg GPU Util | 5.4% |
| Peak GPU Util | 100% |
| Avg Mem Used | 22752 MiB |
| Peak Mem Used | 22752 MiB |
| Avg Temp | 37.3 C |
| Peak Temp | 48 C |

## Request Latency & TTFT
| Metric | P50 | P95 | P99 | Min | Max |
|--------|-----|-----|-----|-----|-----|
| Latency (s) | 0.561 | 1.128 | 1.177 | 0.029 | 1.185 |
| TTFT (s)    | 0.019 | 0.032 | 0.097 | 0.013 | 0.141 |

## Raw Log Paths
- GPU timeseries: `/root/vllm-experiment/logs/2026-06-30-gpu.json`
- BurstGPT detail: `/root/vllm-experiment/logs/2026-06-30-burstgpt-detail.jsonl`
