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

## Installation

### Prerequisites

- Linux with an NVIDIA GPU (experiments run on RTX 4090, 24 GB)
- CUDA 12.x or 13.x
- Python 3.10

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

All subsequent commands assume the venv is active. Scripts resolve
`python3` and `pip` from the active environment automatically.

### 2. Install vLLM 0.23.0

```bash
pip install vllm==0.23.0
```

> **Note:** vLLM requires a CUDA-capable GPU. Install takes ~5 min including
> torch and flash-attention wheels.

### 3. Install experiment dependencies

```bash
pip install -r requirements.txt
```

### 4. Apply vLLM patches

The patches in `patches/` modify vLLM in-place to add pluggable eviction
policies and a dynamic chunk size controller:

```bash
# Apply both patches (recommended)
bash patches/apply_patches.sh

# Or apply individually
bash patches/apply_patches.sh --eviction     # KV-cache eviction policy only
bash patches/apply_patches.sh --chunk-size   # dynamic chunk size only
```

The script auto-detects your vLLM install location and is idempotent —
safe to run again if already applied.

### 5. Fetch datasets

```bash
bash scripts/setup.sh
```

This clones BurstGPT and downloads `BurstGPT_1.csv` and `ShareGPT_v3.json`
into the repo. Requires ~1 GB of disk space.


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

## How Monitoring and Recording Work

Each experiment script runs two processes in parallel:

```
shell script
 ├── src/monitor_gpu.py &    ← background: polls nvidia-smi every 0.5 s
 └── src/replay_sharegpt.py  ← foreground: streams requests, blocks until done
      ↓ (replay finishes)
 kill monitor → monitor flushes JSON → analyze.py joins both files
```

**GPU monitor (`src/monitor_gpu.py`):** Calls `nvidia-smi` every 0.5 s and appends
`{ts, power_w, mem_mib, util_pct, temp_c}` to an in-memory list. On `SIGTERM` (sent
by the shell after replay finishes) it exits the loop and writes the full timeseries
to `logs/YYYY-MM-DD-*-gpu.json`. Nothing is written during the run.

**Request recorder (`src/replay_sharegpt.py`):** N conversations run concurrently via
`ThreadPoolExecutor`. Each completed request appends `{conv_id, turn, ttft, tpot,
latency, …}` to a shared list (protected by a `threading.Lock`). After all threads
finish, records are sorted and written to `logs/YYYY-MM-DD-*-detail.jsonl`.

**Analysis (`src/analyze.py`):** Joins the two files after the fact using wall-clock
timestamps — GPU telemetry is trimmed to the replay window and averaged/summed to
produce the per-experiment energy, power, and latency tables in `findings/`.

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

---

## Dynamic Chunk Size Experiment

Tests whether a feedback controller for chunked-prefill token budget reduces
decode starvation under mixed prefill/decode workloads.

**Patch:** `patches/scheduler.patch` adds `ChunkSizeController` to
`vllm/v1/core/sched/scheduler.py`. Enabled via `DYNAMIC_CHUNK=1`.

**Controller logic (bang-bang):**
- decode depth > target x 1.5 → halve token budget (floor: DYNAMIC_CHUNK_MIN)
- decode depth < target x 0.5 → double token budget (ceiling: static max)

**Results (Qwen2.5-Coder-7B, 150 prompts, rate=inf):**

| Dataset | Metric | Baseline | Dynamic | Delta |
|---------|--------|----------|---------|-------|
| ShareGPT | TTFT p50 | 7083 ms | 5633 ms | -20.5% |
| ShareGPT | TTFT p95 | 14346 ms | 12378 ms | -13.7% |
| ShareGPT | TPOT p95 | 29.5 ms | 25.3 ms | -14.2% |
| ShareGPT | Throughput | 5.61 req/s | 6.01 req/s | +7.2% |
| BurstGPT | TPOT p95 | 127.0 ms | 37.6 ms | -70.4% |
| BurstGPT | TTFT p50 | 4818 ms | 5111 ms | +6.1% |

ShareGPT (mixed arrival): clean win across TTFT, TPOT, and throughput.
BurstGPT (thundering herd): TPOT improves dramatically but TTFT/E2EL
tail rises -- controller over-shrinks chunk under pure burst load.

Full results: `findings/chunk_20260706_093754/`

### Running the chunk experiment

```bash
# Apply the scheduler patch
bash patches/apply_patches.sh --chunk-only

# Run all 4 benchmarks (baseline + dynamic x BurstGPT + ShareGPT)
bash run_chunk_experiment.sh

# Tune via env vars
NUM_PROMPTS=200 DYNAMIC_CHUNK_TARGET=12 bash run_chunk_experiment.sh
```

### Env vars

| Variable | Default | Description |
|----------|---------|-------------|
| DYNAMIC_CHUNK | 0 | Set to 1 to enable |
| DYNAMIC_CHUNK_TARGET | 8 | Target decode-queue depth |
| DYNAMIC_CHUNK_MIN | 256 | Minimum token budget per scheduling step |
