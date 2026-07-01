# BurstGPT QPS Sweep Summary — 2026-07-01

**Model:** Qwen2.5-0.5B-Instruct  |  **gpu-memory-utilization:** 0.9
**Note:** 0.5× = lower QPS than baseline; 2×/4× = higher QPS

| Scale | N | P50 TTFT | P95 TTFT | P50 Lat | P95 Lat | Energy |
|-------|---|----------|----------|---------|---------|--------|
| 0.5x | 45 | 0.020s | 0.037s | 0.647s | 1.304s | 15.235 Wh |
| 1x (baseline 06-30) | 45 | 0.019s | 0.032s | 0.561s | 1.128s | 6.069 Wh |
| 2x | 45 | 0.020s | 0.035s | 0.656s | 1.314s | 4.272 Wh |
| 4x | 45 | 0.013s | 0.025s | 0.579s | 1.296s | 2.767 Wh |

## Per-scale findings
- [0.5x](2026-07-01-qps-0.5x.md)
- [2x](2026-07-01-qps-2x.md)
- [4x](2026-07-01-qps-4x.md)
