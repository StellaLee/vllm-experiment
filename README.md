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

## Eviction Policy Comparison

KV cache eviction policy comparison using vLLM 0.23.0 with Qwen2.5-Coder-7B-Instruct on
ShareGPT multi-turn conversations. Three policies patched into vLLM's `BlockPool`:

| Policy | Score formula | Description |
|--------|---------------|-------------|
| LRU | — | vLLM default; recency-only |
| TDF | `(hit_count+1)·exp(−λ·age)` | Time-Decayed Frequency; penalizes old blocks |
| **CF** | `(hit_count+1)/(prefix_depth+1)` | Cascaded Frequency; preserves structurally critical prefix roots |

**Result (200 conversations, ≥4 turns, concurrency=20):**

| Turn | LRU P95 TTFT | CF P95 TTFT | Improvement |
|------|-------------|-------------|-------------|
| 1 | 166.7 ms | 127.9 ms | +23.3% |
| 2 | 149.8 ms | 122.7 ms | +18.1% |
| 3 | 102.6 ms | 97.5 ms | +5.0% |
| 4 | 108.1 ms | 99.8 ms | +7.7% |

TPOT (decode phase) is identical across policies (within noise) — eviction affects prefix cache
hit rate only, not decode arithmetic.

Full results: [`findings/2026-07-03-full-eviction-comparison.md`](findings/2026-07-03-full-eviction-comparison.md)

### Applying the patches

```bash
# Patches in patches/ apply to vLLM 0.23.0 in-place
bash patches/apply_patches.sh

# Select policy via env var before starting vLLM
EVICTION_POLICY=cf   vllm serve ...   # CF (recommended)
EVICTION_POLICY=tdf  vllm serve ...   # TDF (λ=0.1 default, tune via TDF_LAMBDA)
EVICTION_POLICY=lru  vllm serve ...   # LRU (default, no patch needed)
```

See [`patches/README.md`](patches/README.md) for prerequisites and revert instructions.

## Docs

- [`docs/2026-06-30-baseline-design.md`](docs/2026-06-30-baseline-design.md) — experiment design
- [`docs/2026-06-30-baseline-plan.md`](docs/2026-06-30-baseline-plan.md) — implementation plan
