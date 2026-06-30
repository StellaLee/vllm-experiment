# BurstGPT Baseline Profile — Implementation Plan
Date: 2026-06-30

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a 50-request BurstGPT baseline against vLLM serving Qwen2.5-0.5B-Instruct,
collecting GPU power/utilization timeseries and per-request TTFT/latency, then produce a
dated findings report.

**Architecture:** Five scripts under /root/experiment/. start_server.sh runs vLLM in one
terminal; run_baseline.sh orchestrates the rest — it waits for vLLM health, starts a
background GPU monitor, runs BurstGPT's profiling script with streaming, then invokes
analyze.py which merges the two JSON outputs into a findings markdown.

**Tech Stack:** Python 3 (/root/miniconda3/bin/python3), vLLM 0.23.0, BurstGPT
profile_vllm_server.py, nvidia-smi, aiohttp, pandas, numpy, scipy, transformers

## Global Constraints

- Python interpreter: /root/miniconda3/bin/python3 (has vLLM + deps installed)
- Model path: /model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct
- vLLM API port: 8000
- BurstGPT clone dir: /root/experiment/BurstGPT
- All logs dated YYYY-MM-DD; findings accumulate, never overwrite
- 50 prompts / 50 surplus_prompts for small baseline run
- nvidia-smi poll interval: 0.5 seconds

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| /root/experiment/setup.sh | Create | Clone BurstGPT, install deps, fetch BurstGPT_1.csv |
| /root/experiment/start_server.sh | Create | Launch vLLM OpenAI-compat server on port 8000 |
| /root/experiment/monitor_gpu.py | Create | Poll nvidia-smi every 0.5s → JSON timeseries |
| /root/experiment/run_baseline.sh | Create | Orchestrate: health-check → monitor → profiler → analyze |
| /root/experiment/analyze.py | Create | Merge GPU + BurstGPT JSONs → P50/P95/P99 + energy report |

---

### Task 1: setup.sh — clone BurstGPT, install deps, fetch trace data

**Files:**
- Create: /root/experiment/setup.sh

- [ ] **Step 1: Write setup.sh**

```bash
cat > /root/experiment/setup.sh << 'EOF'
#!/bin/bash
set -e
EXPERIMENT_DIR=/root/experiment
BURSTGPT_DIR=$EXPERIMENT_DIR/BurstGPT
PYTHON=/root/miniconda3/bin/python3
PIP=/root/miniconda3/bin/pip

echo "=== [1/4] Cloning BurstGPT ==="
if [ ! -d "$BURSTGPT_DIR" ]; then
  git clone https://github.com/HPMLL/BurstGPT "$BURSTGPT_DIR"
else
  echo "Already cloned, skipping."
fi

echo "=== [2/4] Installing Python deps ==="
$PIP install --quiet aiohttp>=3.8.6 numpy>=1.25.1 pandas>=2.2.2 scipy>=1.14.0 transformers>=4.41.1

echo "=== [3/4] Checking shareGPT.json ==="
SHAREGPT=$BURSTGPT_DIR/example/preprocess_data/shareGPT.json
if [ ! -f "$SHAREGPT" ]; then
  echo "ERROR: shareGPT.json not found at $SHAREGPT"
  exit 1
fi
echo "Found shareGPT.json ($(wc -l < $SHAREGPT) lines)"

echo "=== [4/4] Fetching BurstGPT_1.csv ==="
DATA_DIR=$BURSTGPT_DIR/data
mkdir -p "$DATA_DIR"
CSV=$DATA_DIR/BurstGPT_1.csv
if [ ! -f "$CSV" ]; then
  echo "Attempting git-lfs pull..."
  cd "$BURSTGPT_DIR" && git lfs pull 2>/dev/null && cd - || true
  if [ ! -f "$CSV" ]; then
    echo "LFS pull failed or not installed. Downloading from GitHub releases..."
    wget -q --show-progress \
      "https://github.com/HPMLL/BurstGPT/releases/download/v1.0/BurstGPT_1.csv" \
      -O "$CSV" 2>/dev/null || \
    wget -q --show-progress \
      "https://github.com/HPMLL/BurstGPT/raw/main/data/BurstGPT_1.csv" \
      -O "$CSV" 2>/dev/null || true
  fi
fi
if [ -f "$CSV" ]; then
  echo "BurstGPT_1.csv ready ($(wc -l < $CSV) rows)"
  export USE_BURSTGPT=1
else
  echo "WARNING: BurstGPT_1.csv not found. Will use Poisson distribution instead."
  export USE_BURSTGPT=0
fi

echo "=== Setup complete ==="
EOF
chmod +x /root/experiment/setup.sh
```

- [ ] **Step 2: Run setup.sh and verify**

```bash
bash /root/experiment/setup.sh
```

Expected output ends with "Setup complete". Verify:
```bash
ls /root/experiment/BurstGPT/example/preprocess_data/shareGPT.json
```

---

### Task 2: start_server.sh — vLLM OpenAI API server

**Files:**
- Create: /root/experiment/start_server.sh

- [ ] **Step 1: Check vLLM entrypoint**

```bash
/root/miniconda3/bin/python3 -c "from vllm.entrypoints.openai import api_server; print('entrypoint ok')" 2>/dev/null \
  || /root/miniconda3/bin/python3 -c "import vllm.entrypoints.openai.api_server; print('entrypoint ok')"
```

Expected: "entrypoint ok"

- [ ] **Step 2: Write start_server.sh**

```bash
cat > /root/experiment/start_server.sh << 'EOF'
#!/bin/bash
PYTHON=/root/miniconda3/bin/python3
MODEL=/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct
PORT=8000

echo "Starting vLLM server: model=$MODEL port=$PORT"
echo "Logs will appear below. Ready when you see 'Application startup complete'."

exec $PYTHON -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --dtype auto \
  --max-model-len 4096
EOF
chmod +x /root/experiment/start_server.sh
```

- [ ] **Step 3: Verify script syntax**

```bash
bash -n /root/experiment/start_server.sh && echo "syntax ok"
```

Expected: "syntax ok"

---

### Task 3: monitor_gpu.py — nvidia-smi polling loop

**Files:**
- Create: /root/experiment/monitor_gpu.py

**Interfaces:**
- Produces: JSON file at path given by --output arg
  Schema: `{"interval_s": 0.5, "records": [{"ts": float, "power_w": float, "mem_mib": int, "util_pct": int, "temp_c": int}, ...]}`

- [ ] **Step 1: Write monitor_gpu.py**

```bash
cat > /root/experiment/monitor_gpu.py << 'EOF'
#!/usr/bin/env python3
"""Poll nvidia-smi every 0.5s and write a JSON timeseries. Run with --output path."""
import argparse, json, signal, subprocess, sys, time

INTERVAL = 0.5
QUERY = "power.draw,memory.used,utilization.gpu,temperature.gpu"
FMT = "csv,noheader,nounits"

records = []
running = True

def _stop(sig, frame):
    global running
    running = False

def poll():
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={QUERY}", f"--format={FMT}"],
        text=True
    ).strip()
    parts = [p.strip() for p in out.split(",")]
    return {
        "ts": time.time(),
        "power_w": float(parts[0]),
        "mem_mib": int(parts[1]),
        "util_pct": int(parts[2]),
        "temp_c": int(parts[3]),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path to write JSON timeseries")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    print(f"[monitor_gpu] polling every {INTERVAL}s → {args.output}", flush=True)
    while running:
        try:
            records.append(poll())
        except Exception as e:
            print(f"[monitor_gpu] poll error: {e}", file=sys.stderr)
        time.sleep(INTERVAL)

    with open(args.output, "w") as f:
        json.dump({"interval_s": INTERVAL, "records": records}, f, indent=2)
    print(f"[monitor_gpu] saved {len(records)} records to {args.output}", flush=True)

if __name__ == "__main__":
    main()
EOF
chmod +x /root/experiment/monitor_gpu.py
```

- [ ] **Step 2: Smoke-test monitor_gpu.py**

```bash
PYTHON=/root/miniconda3/bin/python3
$PYTHON /root/experiment/monitor_gpu.py --output /tmp/gpu_test.json &
MON_PID=$!
sleep 3
kill $MON_PID
sleep 1
$PYTHON -c "import json; d=json.load(open('/tmp/gpu_test.json')); print(f'{len(d[\"records\"])} records, first={d[\"records\"][0]}')"
```

Expected: "5 records" (approx), with power_w, mem_mib, util_pct, temp_c fields visible.

---

### Task 4: run_baseline.sh — orchestration

**Files:**
- Create: /root/experiment/run_baseline.sh

**Interfaces:**
- Consumes:
  - monitor_gpu.py --output path → JSON schema from Task 3
  - BurstGPT: /root/experiment/BurstGPT/example/profile_vllm_server.py
  - analyze.py --gpu-log --burstgpt-log --output (Task 5)
- Produces:
  - /root/experiment/logs/YYYY-MM-DD-gpu.json
  - /root/experiment/logs/YYYY-MM-DD-burstgpt-detail.json

- [ ] **Step 1: Write run_baseline.sh**

```bash
cat > /root/experiment/run_baseline.sh << 'EOF'
#!/bin/bash
set -e
PYTHON=/root/miniconda3/bin/python3
DATE=$(date +%Y-%m-%d)
EXPERIMENT_DIR=/root/experiment
BURSTGPT_DIR=$EXPERIMENT_DIR/BurstGPT
LOG_DIR=$EXPERIMENT_DIR/logs
FINDINGS_DIR=$EXPERIMENT_DIR/findings
PORT=8000
HOST=localhost

mkdir -p "$LOG_DIR" "$FINDINGS_DIR"

GPU_LOG=$LOG_DIR/${DATE}-gpu.json
BURSTGPT_LOG=$LOG_DIR/${DATE}-burstgpt.json
BURSTGPT_DETAIL=$LOG_DIR/${DATE}-burstgpt-detail.json
FINDINGS=$FINDINGS_DIR/${DATE}-baseline-qwen2.5-0.5b.md

echo "=== [1/5] Waiting for vLLM on $HOST:$PORT ==="
for i in $(seq 1 60); do
  if wget -q --spider "http://$HOST:$PORT/health" 2>/dev/null; then
    echo "vLLM is up!"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "ERROR: vLLM not ready after 60s. Run start_server.sh in another terminal."
    exit 1
  fi
  printf "."
  sleep 2
done

echo "=== [2/5] Starting GPU monitor ==="
$PYTHON /root/experiment/monitor_gpu.py --output "$GPU_LOG" &
MON_PID=$!
echo "Monitor PID: $MON_PID"
sleep 1

echo "=== [3/5] Running BurstGPT profiler (50 requests, streaming) ==="
SHAREGPT=$BURSTGPT_DIR/example/preprocess_data/shareGPT.json
BURSTGPT_CSV=$BURSTGPT_DIR/data/BurstGPT_1.csv

# Build flags
FLAGS="--port=$PORT --host=$HOST --temperature=0 --stream"
FLAGS="$FLAGS --data_path=$SHAREGPT"
FLAGS="$FLAGS --model_path=/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct"
FLAGS="$FLAGS --surplus_prompts_num=50 --prompt_num=50"
FLAGS="$FLAGS --max_tokens=128"
FLAGS="$FLAGS --log_path=$BURSTGPT_LOG"
FLAGS="$FLAGS --detail_log_path=$BURSTGPT_DETAIL"

if [ -f "$BURSTGPT_CSV" ]; then
  echo "Using BurstGPT trace: $BURSTGPT_CSV"
  FLAGS="$FLAGS --use_burstgpt --burstgpt_path=$BURSTGPT_CSV --scale=1.2344107085"
else
  echo "BurstGPT_1.csv not found. Using Poisson distribution at QPS=1."
  FLAGS="$FLAGS --qps=1.0"
fi

cd $BURSTGPT_DIR/example
$PYTHON profile_vllm_server.py $FLAGS
cd -

echo "=== [4/5] Stopping GPU monitor ==="
kill $MON_PID 2>/dev/null || true
wait $MON_PID 2>/dev/null || true

echo "=== [5/5] Running analysis ==="
$PYTHON /root/experiment/analyze.py \
  --gpu-log "$GPU_LOG" \
  --burstgpt-log "$BURSTGPT_DETAIL" \
  --output "$FINDINGS"

echo ""
echo "=== Done! Findings: $FINDINGS ==="
cat "$FINDINGS"
EOF
chmod +x /root/experiment/run_baseline.sh
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n /root/experiment/run_baseline.sh && echo "syntax ok"
```

Expected: "syntax ok"

---

### Task 5: analyze.py — compute metrics and write dated findings

**Files:**
- Create: /root/experiment/analyze.py

**Interfaces:**
- Consumes:
  - --gpu-log: JSON with schema `{"interval_s": float, "records": [{"ts", "power_w", "mem_mib", "util_pct", "temp_c"}]}`
  - --burstgpt-log: JSON list of per-request records (field names auto-detected)
- Produces: dated markdown at --output path

- [ ] **Step 1: Write analyze.py**

```bash
cat > /root/experiment/analyze.py << 'EOF'
#!/usr/bin/env python3
"""Merge GPU timeseries + BurstGPT detail log → dated markdown findings report."""
import argparse, json, math, os
from datetime import datetime

def percentile(data, p):
    if not data:
        return float("nan")
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])

def find_field(record, candidates):
    for c in candidates:
        if c in record:
            return c
    return None

def load_burstgpt(path):
    if not os.path.exists(path):
        return [], {}
    with open(path) as f:
        raw = json.load(f)
    if not raw:
        return [], {}
    # Auto-detect field names from first record
    if isinstance(raw, list):
        sample = raw[0]
    elif isinstance(raw, dict):
        # might be {"results": [...]} or similar
        for v in raw.values():
            if isinstance(v, list) and v:
                raw = v
                sample = v[0]
                break
        else:
            return [], {}
    else:
        return [], {}
    print(f"[analyze] BurstGPT record keys: {list(sample.keys())}")
    ttft_field = find_field(sample, ["ttft", "first_token_latency", "time_to_first_token",
                                      "first_token_time", "TTFT"])
    lat_field  = find_field(sample, ["latency", "elapsed_time", "e2e_latency",
                                      "total_time", "end2end_latency", "response_time"])
    return raw, {"ttft": ttft_field, "lat": lat_field}

def load_gpu(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def compute_energy_wh(records, interval_s):
    total_ws = sum(r["power_w"] for r in records) * interval_s  # watt-seconds
    return total_ws / 3600

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-log", required=True)
    parser.add_argument("--burstgpt-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gpu = load_gpu(args.gpu_log)
    requests, fields = load_burstgpt(args.burstgpt_log)

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Baseline Profile — Qwen2.5-0.5B-Instruct",
        f"**Date:** {date_str}",
        f"**Model:** /model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct",
        f"**GPU:** NVIDIA GeForce RTX 4090 (450W TDP)",
        f"**vLLM:** 0.23.0  |  **Requests:** {len(requests)}",
        "",
    ]

    # --- GPU metrics ---
    lines.append("## GPU Metrics")
    if gpu and gpu.get("records"):
        recs = gpu["records"]
        powers = [r["power_w"] for r in recs]
        utils  = [r["util_pct"] for r in recs]
        mems   = [r["mem_mib"] for r in recs]
        temps  = [r["temp_c"] for r in recs]
        energy = compute_energy_wh(recs, gpu["interval_s"])
        duration_s = len(recs) * gpu["interval_s"]
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Duration | {duration_s:.1f} s |",
            f"| Total Energy | {energy*1000:.1f} mWh ({energy:.4f} Wh) |",
            f"| Avg Power | {sum(powers)/len(powers):.1f} W |",
            f"| Peak Power | {max(powers):.1f} W |",
            f"| Idle Power (first 5s) | {sum(powers[:10])/max(1,len(powers[:10])):.1f} W |",
            f"| Avg GPU Util | {sum(utils)/len(utils):.1f}% |",
            f"| Peak GPU Util | {max(utils)}% |",
            f"| Avg Mem Used | {sum(mems)/len(mems):.0f} MiB |",
            f"| Peak Mem Used | {max(mems)} MiB |",
            f"| Avg Temp | {sum(temps)/len(temps):.1f}°C |",
            f"| Peak Temp | {max(temps)}°C |",
        ]
    else:
        lines.append("_GPU log not found or empty._")
    lines.append("")

    # --- Latency / TTFT metrics ---
    lines.append("## Request Latency & TTFT")
    if requests and fields["lat"]:
        lats = [r[fields["lat"]] for r in requests if fields["lat"] in r]
        lines += [
            f"| Metric | P50 | P95 | P99 | Min | Max |",
            f"|--------|-----|-----|-----|-----|-----|",
            f"| Latency (s) | {percentile(lats,50):.3f} | {percentile(lats,95):.3f} | {percentile(lats,99):.3f} | {min(lats):.3f} | {max(lats):.3f} |",
        ]
        if fields["ttft"]:
            ttfts = [r[fields["ttft"]] for r in requests if fields["ttft"] in r]
            lines.append(
                f"| TTFT (s)    | {percentile(ttfts,50):.3f} | {percentile(ttfts,95):.3f} | {percentile(ttfts,99):.3f} | {min(ttfts):.3f} | {max(ttfts):.3f} |"
            )
        else:
            lines.append(f"_TTFT field not found in log. Available keys: {list(requests[0].keys())}_")
    elif requests:
        lines.append(f"_Latency field not detected. Record keys: {list(requests[0].keys())}_")
        lines.append(f"_Raw first record: {json.dumps(requests[0], indent=2)}_")
    else:
        lines.append("_BurstGPT detail log not found or empty._")
    lines.append("")

    lines.append("## Raw Log Paths")
    lines += [
        f"- GPU timeseries: `{args.gpu_log}`",
        f"- BurstGPT detail: `{args.burstgpt_log}`",
    ]

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"[analyze] Report written to {args.output}")
    print(report)

if __name__ == "__main__":
    main()
EOF
chmod +x /root/experiment/analyze.py
```

- [ ] **Step 2: Smoke-test analyze.py with mock data**

```bash
PYTHON=/root/miniconda3/bin/python3
# Create mock GPU log
python3 - << 'PYEOF'
import json, time, random
records = [{"ts": time.time()+i*0.5, "power_w": 80+random.random()*20,
            "mem_mib": 1200, "util_pct": 45, "temp_c": 35} for i in range(20)]
json.dump({"interval_s": 0.5, "records": records}, open("/tmp/mock_gpu.json","w"))
# Create mock BurstGPT detail log
reqs = [{"latency": 0.5+random.random()*0.3, "ttft": 0.05+random.random()*0.02,
         "prompt_tokens": 100, "output_tokens": 50} for _ in range(50)]
json.dump(reqs, open("/tmp/mock_detail.json","w"))
PYEOF
$PYTHON /root/experiment/analyze.py \
  --gpu-log /tmp/mock_gpu.json \
  --burstgpt-log /tmp/mock_detail.json \
  --output /tmp/mock_findings.md
cat /tmp/mock_findings.md
```

Expected: markdown table with GPU metrics and latency P50/P95/P99 values printed.

---

## Running the Full Experiment

Once all tasks complete:

```bash
# Terminal 1 — start vLLM (blocks)
bash /root/experiment/start_server.sh

# Terminal 2 — wait for server, then run everything
bash /root/experiment/setup.sh   # one-time
bash /root/experiment/run_baseline.sh
```

Findings appear in `/root/experiment/findings/YYYY-MM-DD-baseline-qwen2.5-0.5b.md`.
