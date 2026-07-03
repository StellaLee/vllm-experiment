# vllm-experiment

Profiling and KV-cache eviction experiments for vLLM — measuring GPU power draw,
energy consumption, TTFT, and request latency under realistic bursty LLM traffic
using [BurstGPT](https://github.com/HPMLL/BurstGPT) traces and ShareGPT multi-turn
conversations.

## Hardware & Software

**Hardware:** NVIDIA GeForce RTX 4090 (24 GB, 450 W TDP)  
**Software:** vLLM 0.23.0, Python 3 (`/root/miniconda3/bin/python3`), CUDA 13.2

## Codebase Structure

```
vllm-experiment/
├── README.md
├── requirements.txt
│
├── scripts/                         # experiment orchestration (run on server)
│   ├── setup.sh                     # one-time: clone BurstGPT, install deps, fetch datasets
│   ├── start_server.sh              # start vLLM (Qwen2.5-0.5B, port 8000)
│   │
│   ├── run_baseline.sh              # BurstGPT single-run baseline
│   ├── run_qps_sweep.sh             # BurstGPT QPS sweep (0.5×/2×/4×, 0.5B)
│   ├── run_2x_4x_sweep.sh           # BurstGPT 2×/4× partial sweep
│   ├── run_qps7b_sweep.sh           # BurstGPT QPS sweep (1×/2×/4×/8×, 7B)
│   │
│   ├── run_sharegpt.sh              # ShareGPT multi-turn replay (0.5B)
│   ├── run_kvcache_sweep.sh         # KV cache utilization sweep (0.3–0.9, 0.5B)
│   ├── run_kvcache_7b_sweep.sh      # KV cache utilization sweep (0.7–0.9, 7B)
│   │
│   ├── run_eviction_comparison.sh   # LRU vs TDF eviction (50 convs, legacy)
│   └── run_full_comparison.sh       # LRU vs TDF vs CF eviction (200 convs, canonical)
│
├── src/                             # Python analysis and replay scripts
│   ├── replay_sharegpt.py           # stream ShareGPT conversations against vLLM /generate
│   ├── monitor_gpu.py               # poll nvidia-smi every 0.5 s → JSON timeseries
│   ├── analyze.py                   # merge GPU + request logs → dated markdown report
│   ├── summarize.py                 # aggregate sweep runs → comparison table
│   └── compare_eviction.py          # diff LRU/TDF/CF JSONL logs → TTFT/TPOT tables
│
├── patches/                         # vLLM 0.23.0 in-place patches (eviction policies)
│   ├── apply_patches.sh             # auto-detect vLLM install and apply both patches
│   ├── kv_cache_utils.patch         # adds hit_count / prefix_depth fields to KVCacheBlock
│   ├── block_pool.patch             # routes EVICTION_POLICY env var → BlockPool
│   └── README.md                    # patch prerequisites, quick start, revert instructions
│
├── docs/                            # design documents
│   ├── 2026-06-30-baseline-design.md
│   └── 2026-06-30-baseline-plan.md
│
└── findings/                        # dated experiment results (one file per run)
    ├── 2026-06-30-baseline-qwen2.5-0.5b.md
    ├── 2026-07-01-qps-sweep-summary.md
    ├── 2026-07-01-kvcache-sweep-summary.md
    ├── 2026-07-01-kvcache7b-sweep-summary.md
    ├── 2026-07-02-qps7b-sweep-summary.md
    ├── 2026-07-03-eviction-comparison.md     # 50-conv pilot (n=19 turn-4 samples)
    └── 2026-07-03-full-eviction-comparison.md # 200-conv canonical run
```

## Quick Start

```bash
# One-time: clone BurstGPT, install deps, fetch BurstGPT_1.csv and ShareGPT V3
bash scripts/setup.sh

# Terminal 1 — start vLLM
bash scripts/start_server.sh

# Terminal 2 — run an experiment
bash scripts/run_baseline.sh
```

## Experiments

### BurstGPT Baseline

Sends 50 requests from a BurstGPT trace at realistic inter-arrival times.
Measures TTFT, latency, GPU power, and energy per request.

```bash
bash scripts/run_baseline.sh
bash scripts/run_qps_sweep.sh       # 0.5×, 2×, 4× BurstGPT QPS
bash scripts/run_qps7b_sweep.sh     # same sweep with Coder-7B
```

### KV Cache Utilization Sweep

Runs ShareGPT multi-turn replay at different `gpu-memory-utilization` settings
to measure how cache capacity affects TTFT at different turns.

```bash
bash scripts/run_kvcache_sweep.sh       # 0.3, 0.5, 0.7, 0.9 (0.5B)
bash scripts/run_kvcache_7b_sweep.sh    # 0.7, 0.8, 0.9 (Coder-7B)
```

### Eviction Policy Comparison

Three-way comparison of KV cache eviction policies.
Requires applying the patches in `patches/` first.

| Policy | Score formula | Result |
|--------|---------------|--------|
| LRU | least-recently-used (default) | baseline |
| TDF | `(hit_count+1)·exp(−λ·age)` | worse than LRU on multi-turn |
| **CF** | `(hit_count+1)/(prefix_depth+1)` | **+5–23% P95 TTFT** vs LRU |

```bash
# 1. Apply vLLM patches
bash patches/apply_patches.sh

# 2. Run the canonical 3-way comparison (200 conversations, ≥4 turns)
bash scripts/run_full_comparison.sh

# Or run with a custom policy:
EVICTION_POLICY=cf bash scripts/start_server.sh
```

Full results: [`findings/2026-07-03-full-eviction-comparison.md`](findings/2026-07-03-full-eviction-comparison.md)  
See [`patches/README.md`](patches/README.md) for patch details and revert instructions.

## Output

```
logs/                              # raw timeseries (gitignored)
  YYYY-MM-DD-*-gpu.json
  YYYY-MM-DD-*-detail.jsonl

findings/                          # committed human-readable summaries
  YYYY-MM-DD-*.md
```

**Metrics reported:**
- GPU: avg/peak power (W), total energy (Wh), utilization (%), memory (MiB), temperature (°C)
- Requests: P50/P95 TTFT and total latency, TPOT (decode phase only)

## Model Paths

| Model | Path on server |
|-------|---------------|
| Qwen2.5-0.5B-Instruct | `/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct` |
| Qwen2.5-Coder-7B-Instruct | `/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct` |

## Docs

- [`docs/2026-06-30-baseline-design.md`](docs/2026-06-30-baseline-design.md) — experiment design
- [`docs/2026-06-30-baseline-plan.md`](docs/2026-06-30-baseline-plan.md) — implementation plan
