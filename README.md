# vllm-experiment

Baseline profiling experiments for vLLM serving — measuring GPU power draw,
energy consumption, TTFT, and request latency under realistic bursty LLM
traffic using [BurstGPT](https://github.com/HPMLL/BurstGPT) traces.

## Setup

**Hardware:** NVIDIA GeForce RTX 4090 (24 GB, 450 W TDP)  
**Software:** vLLM 0.23.0, Python 3 (`/root/miniconda3/bin/python3`), CUDA 13.2

```bash
# One-time setup: clone BurstGPT, install deps, fetch BurstGPT_1.csv trace
bash setup.sh
```

## Running an Experiment

Open two terminals on the server:

```bash
# Terminal 1 — start the vLLM server (blocks until killed)
bash start_server.sh
# Wait for: "Application startup complete"

# Terminal 2 — run the full baseline pipeline
bash run_baseline.sh
```

`run_baseline.sh` will:
1. Wait for vLLM to be healthy on port 8000
2. Start `monitor_gpu.py` in the background (polls nvidia-smi every 0.5 s)
3. Run BurstGPT `profile_vllm_server.py` (50 requests, streaming enabled)
4. Stop the GPU monitor
5. Run `analyze.py` → write a dated findings report

## Scripts

| Script | Purpose |
|--------|---------|
| `setup.sh` | Clone BurstGPT, install pip deps, fetch BurstGPT_1.csv |
| `start_server.sh` | Launch vLLM with Qwen2.5-0.5B-Instruct on port 8000 |
| `monitor_gpu.py` | Poll nvidia-smi every 0.5 s → `logs/YYYY-MM-DD-gpu.json` |
| `run_baseline.sh` | Orchestrate full experiment run |
| `analyze.py` | Merge GPU + BurstGPT logs → dated markdown report |

## Output

```
logs/
  YYYY-MM-DD-gpu.json            # raw GPU timeseries (gitignored)
  YYYY-MM-DD-burstgpt-detail.json

findings/
  YYYY-MM-DD-baseline-qwen2.5-0.5b.md   # human-readable summary
```

**Metrics reported:**
- GPU: avg/peak power (W), total energy (Wh), utilization (%), memory (MiB), temperature (°C)
- Requests: P50/P95/P99 latency and TTFT

## Model

Currently configured for `Qwen2.5-0.5B-Instruct` at:
```
/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct
```

To switch models, edit the `MODEL` variable in `start_server.sh` and
`--model_path` in `run_baseline.sh`.

## Committing Results

After each run, commit the findings to track results over time:

```bash
git add findings/
git commit -m "results: baseline run $(date +%Y-%m-%d)"
git push
```

## Docs

- [`docs/2026-06-30-baseline-design.md`](docs/2026-06-30-baseline-design.md) — experiment design
- [`docs/2026-06-30-baseline-plan.md`](docs/2026-06-30-baseline-plan.md) — implementation plan
