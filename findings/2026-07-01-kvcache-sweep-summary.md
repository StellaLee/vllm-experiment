# ShareGPT KV Cache Size Sweep Summary — 2026-07-01

**Model:** Qwen2.5-0.5B-Instruct  |  **Trace:** ShareGPT multi-turn
**Signal:** rising P95 TTFT on later turns = KV cache pressure forcing evictions.

| gpu-memory-utilization | N | P50 TTFT | P95 TTFT | P50 Lat | P95 Lat | Energy |
|------------------------|---|----------|----------|---------|---------|--------|
| gpu-util=0.3 | 256 | 0.011s | 0.022s | 0.231s | 0.243s | 3.831 Wh |
| gpu-util=0.5 | 256 | 0.011s | 0.023s | 0.231s | 0.244s | 3.811 Wh |
| gpu-util=0.7 | 256 | 0.012s | 0.023s | 0.232s | 0.243s | 3.780 Wh |
| gpu-util=0.9 | 256 | 0.012s | 0.023s | 0.231s | 0.243s | 3.755 Wh |

## Per-utilization findings
- [gpu-util=0.3](2026-07-01-kvcache-0.3.md)
- [gpu-util=0.5](2026-07-01-kvcache-0.5.md)
- [gpu-util=0.7](2026-07-01-kvcache-0.7.md)
- [gpu-util=0.9](2026-07-01-kvcache-0.9.md)

## Reproduction

**Prerequisites (one-time):**
```bash
bash scripts/setup.sh   # fetches ShareGPT V3 dataset
```

**Run the full sweep (restarts vLLM for each utilization level):**
```bash
bash scripts/run_kvcache_sweep.sh
```

Per-utilization output: `findings/YYYY-MM-DD-kvcache-{0.3,0.5,0.7,0.9}.md`  
Summary output: `findings/YYYY-MM-DD-kvcache-sweep-summary.md`

**Key parameters (override via env):**
```bash
NUM_CONVS=100 MAX_TURNS=4 MAX_TOKENS=128 bash scripts/run_kvcache_sweep.sh
```
