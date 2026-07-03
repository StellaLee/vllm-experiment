# Session Notes — 2026-06-30

## Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4090 (24 GB VRAM, 450 W TDP) |
| vLLM | 0.23.0 (native API server — `vllm.entrypoints.api_server`) |
| Model | Qwen2.5-0.5B-Instruct (`/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct`) |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.2 |
| Workload | BurstGPT_1.csv — 45 requests, `--scale=1.2344`, `--conv_or_api=conv` |

---

## Key Findings

### Latency & TTFT (45 requests)

| Metric | P50 | P95 | P99 | Min | Max |
|--------|-----|-----|-----|-----|-----|
| TTFT (s) | 0.019 | 0.032 | 0.097 | 0.013 | 0.141 |
| Latency (s) | 0.561 | 1.128 | 1.177 | 0.029 | 1.185 |

### GPU Power & Energy (397 s window)

| Metric | Value |
|--------|-------|
| Avg Power | 55.1 W |
| Peak Power | 240.4 W |
| Idle Power (first 5s) | 48.4 W |
| Total Energy | 6.07 Wh (6068.7 mWh) |
| Avg GPU Util | 5.4% |
| Peak GPU Util | 100% |
| Avg Mem Used | 22752 MiB |
| Peak Temp | 48 °C |

### Observations

- **Bursty utilization**: avg GPU util is only 5.4% with 100% spikes — matches the
  real Azure bursty trace pattern where requests cluster then idle.
- **Fast TTFT**: P50 of 19 ms reflects the small model size (0.5B) and cached
  torch.compile graphs (warm second run).
- **Power headroom**: peak 240 W vs 450 W TDP — the 0.5B model barely stresses the GPU.
- **Memory**: 22752 MiB used — nearly the full 24 GB — due to vLLM pre-allocating
  KV cache pages at startup.

---

## Issues Encountered & Fixes

### 1. `ninja` not installed — vLLM FlashInfer JIT fails
**Symptom:** `FileNotFoundError: ninja` during engine core init.  
**Fix:** `apt-get install -y ninja-build`

### 2. Health check using HEAD — vLLM `/health` returns 405
**Symptom:** `wget --spider` sends HEAD; vLLM `/health` only accepts GET.  
**Fix:** Switch to `wget -q -O /dev/null` (GET).

### 3. BurstGPT repo refactored — `profile_vllm_server.py` removed
**Symptom:** Script not found at `example/profile_vllm_server.py`.  
**Fix:** Install as package and use `burstgpt-bench` CLI:
```bash
cd /root/vllm-experiment/BurstGPT/example && pip install -e .
```

### 4. vLLM `/generate` endpoint requires native API server
**Symptom:** BurstGPT posts to `/generate` but OpenAI-compat server returns 404.  
**Fix:** Use `vllm.entrypoints.api_server` (not `openai.api_server`).

### 5. BurstGPT streaming format mismatch
**Symptom:** `json.JSONDecodeError` — BurstGPT expected null-byte delimited stream;
vLLM 0.23.0 sends newline-delimited JSON.  
**Fix:** Patched `BurstGPT/example/src/burstgpt/backends.py` —
replaced `_read_null_delimited_stream` to split on `\n` instead of `\0`.

### 6. `analyze.py` used `json.load()` on JSONL output
**Symptom:** Parse error — BurstGPT now writes one JSON object per line.  
**Fix:** Rewrote `load_burstgpt()` to read line-by-line with fallback to single-JSON.

---

## Reproduction Steps

### Prerequisites (one-time)
```bash
apt-get install -y ninja-build
cd /root/vllm-experiment
bash setup.sh          # clones BurstGPT, installs deps, fetches BurstGPT_1.csv
cd BurstGPT/example && pip install -e . && cd /root/vllm-experiment
```

### Run experiment
```bash
# Terminal 1 — start vLLM (blocks; wait for "Application startup complete")
bash scripts/start_server.sh

# Terminal 2 — run baseline
bash scripts/run_baseline.sh
```

Results appear in `findings/YYYY-MM-DD-baseline-qwen2.5-0.5b.md`.

### Re-run analysis on existing logs
```bash
/root/miniconda3/bin/python3 src/analyze.py \
  --gpu-log  logs/2026-06-30-gpu.json \
  --burstgpt-log logs/2026-06-30-burstgpt-detail.jsonl \
  --output   findings/2026-06-30-baseline-qwen2.5-0.5b.md
```

### Commit results after each run
```bash
git add findings/
git commit -m "results: baseline run $(date +%Y-%m-%d)"
git push
```

---

## Raw Log Paths

| Log | Path |
|-----|------|
| GPU timeseries | `/root/vllm-experiment/logs/2026-06-30-gpu.json` |
| BurstGPT detail | `/root/vllm-experiment/logs/2026-06-30-burstgpt-detail.jsonl` |
| vLLM server | `/root/vllm-experiment/logs/vllm-server.log` |
| Findings | `/root/vllm-experiment/findings/2026-06-30-baseline-qwen2.5-0.5b.md` |

## Next Steps

- Run full 50-request experiment to completion (was stopped at 45/50)
- QPS sweep: repeat at scale factors 0.5×, 2×, 4× to see power/latency curve
- Try larger model (Qwen2.5-Coder-7B-Instruct) to observe higher power draw
