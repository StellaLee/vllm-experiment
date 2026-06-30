# BurstGPT Baseline Profile — Design Doc
Date: 2026-06-30

## Goal
Run a small-scale baseline experiment using BurstGPT real-world traces against a vLLM server
serving Qwen2.5-0.5B-Instruct. Observe GPU latency, energy consumption, and TTFT under
realistic bursty LLM traffic.

## Model
- **Model**: Qwen2.5-0.5B-Instruct
- **Path**: /model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct
- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM, 450W TDP)
- **vLLM version**: 0.23.0

## Dataset
- **Source**: BurstGPT traces (BurstGPT_1.csv) + ShareGPT prompts
- **Scale**: ~100–200 requests (small baseline run)
- **Traffic pattern**: Poisson-distributed at QPS=1.0, with BurstGPT trace timestamps
- **Streaming**: enabled (required for accurate TTFT measurement)

## Directory Layout
```
/root/experiment/
├── setup.sh              # one-time: clone BurstGPT, install deps, fetch data
├── start_server.sh       # launch vLLM with Qwen2.5-0.5B-Instruct on port 8000
├── monitor_gpu.py        # polls nvidia-smi every 0.5s → JSON timeseries
├── run_baseline.sh       # orchestrates full run
├── analyze.py            # computes metrics → dated markdown report
├── docs/
│   └── 2026-06-30-baseline-design.md   ← this file
├── logs/
│   ├── YYYY-MM-DD-gpu.json             # raw GPU timeseries
│   └── YYYY-MM-DD-burstgpt.json        # raw BurstGPT output
└── findings/
    └── YYYY-MM-DD-baseline-qwen2.5-0.5b.md
```

## Metrics
### From BurstGPT (streaming)
- TTFT per request (time from send to first token received)
- End-to-end latency per request

### From nvidia-smi (0.5s poll)
- power.draw (W)
- memory.used (MiB)
- utilization.gpu (%)
- temperature.gpu (°C)

### Derived
- Total energy = Σ(power_draw × 0.5s), reported in Wh
- P50 / P95 / P99 for TTFT and latency

## Execution Flow
```
Terminal 1: bash start_server.sh        # blocks; wait for "Application startup complete"
Terminal 2: bash run_baseline.sh        # waits for /health, then runs full pipeline
```

run_baseline.sh steps:
1. Wait for vLLM /health endpoint
2. Start monitor_gpu.py in background
3. Run BurstGPT profile_vllm_server.py (200 requests, streaming, QPS=1)
4. Stop monitor_gpu.py
5. Run analyze.py → findings/YYYY-MM-DD-baseline-qwen2.5-0.5b.md

## Resumption Notes
- All scripts are idempotent; rerun run_baseline.sh for a fresh dated run
- Findings are dated so multiple runs accumulate without overwriting
- To resume: start server, then run_baseline.sh
