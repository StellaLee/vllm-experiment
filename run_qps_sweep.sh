#!/bin/bash
set -e

PYTHON=/root/miniconda3/bin/python3
BURSTGPT_BENCH=/root/miniconda3/bin/burstgpt-bench
DATE=$(date +%Y-%m-%d)
EXPERIMENT_DIR=/root/vllm-experiment
LOG_DIR=$EXPERIMENT_DIR/logs
FINDINGS_DIR=$EXPERIMENT_DIR/findings
PORT=8000
HOST=localhost
MODEL=/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct
SHAREGPT=$EXPERIMENT_DIR/BurstGPT/example/preprocess_data/shareGPT.json
BURSTGPT_CSV=$EXPERIMENT_DIR/BurstGPT/data/BurstGPT_1.csv
BASE_SCALE=1.2344107085

mkdir -p "$LOG_DIR" "$FINDINGS_DIR"

# ===== Start vLLM once for all sweep runs =====
echo "=== Starting vLLM server (gpu-memory-utilization=0.9) ==="
$PYTHON -m vllm.entrypoints.api_server \
  --model $MODEL --port $PORT --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  > $LOG_DIR/${DATE}-qps-sweep-vllm.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

MON_PID=""
cleanup() {
  [ -n "$MON_PID" ] && { kill "$MON_PID" 2>/dev/null || true; wait "$MON_PID" 2>/dev/null || true; }
  kill $VLLM_PID 2>/dev/null || true
  wait $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM..."
for i in $(seq 1 90); do
  if $PYTHON -c "
import urllib.request, json
urllib.request.urlopen(urllib.request.Request(
    'http://${HOST}:${PORT}/generate',
    data=json.dumps({'prompt':'hi','max_tokens':1,'stream':False,'temperature':0}).encode(),
    headers={'Content-Type':'application/json'}), timeout=3)
" 2>/dev/null; then
    echo " vLLM ready."
    break
  fi
  [ $i -eq 90 ] && { echo "ERROR: vLLM not ready after 180s"; exit 1; }
  printf "."; sleep 2
done
sleep 2  # let KV cache settle

# ===== Sweep: 0.5x, 2x, 4x relative to base scale =====
for MULT in 0.5 2.0 4.0; do
  SCALE=$($PYTHON -c "print('{:.6f}'.format($BASE_SCALE * $MULT))")
  # "2.0" → "2x", "0.5" → "0.5x", "4.0" → "4x"
  LABEL=$(echo "$MULT" | awk '{v=$1+0; if(v==int(v)) printf "%dx",int(v); else printf "%sx",v}')

  echo ""
  echo "========================================================"
  echo " QPS ${LABEL}  |  --scale=${SCALE}"
  echo "========================================================"

  GPU_LOG=$LOG_DIR/${DATE}-qps-${LABEL}-gpu.json
  DETAIL=$LOG_DIR/${DATE}-qps-${LABEL}-burstgpt-detail.jsonl
  BURSLOG=$LOG_DIR/${DATE}-qps-${LABEL}-burstgpt.jsonl
  FINDINGS=$FINDINGS_DIR/${DATE}-qps-${LABEL}.md

  echo "--- Starting GPU monitor ---"
  $PYTHON $EXPERIMENT_DIR/monitor_gpu.py --output "$GPU_LOG" &
  MON_PID=$!
  sleep 1

  echo "--- Running BurstGPT (scale=${SCALE}) ---"
  $BURSTGPT_BENCH \
    --port=$PORT --host=$HOST \
    --temperature=0 --stream \
    --data_path=$SHAREGPT \
    --model_path=$MODEL \
    --surplus_prompts_num=50 --prompt_num=50 \
    --max_tokens=128 \
    --log_path=$BURSLOG \
    --detail_log_path=$DETAIL \
    --use_burstgpt --burstgpt_path=$BURSTGPT_CSV \
    --scale=$SCALE

  echo "--- Stopping GPU monitor ---"
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  MON_PID=""

  echo "--- Analyzing ${LABEL} ---"
  $PYTHON $EXPERIMENT_DIR/analyze.py \
    --gpu-log "$GPU_LOG" \
    --burstgpt-log "$DETAIL" \
    --output "$FINDINGS"

  echo "=== ${LABEL} done ==="
  echo ""
  sleep 8  # cooldown between runs
done

# ===== Comparison Summary =====
echo "=== Generating comparison summary ==="
export DATE LOG_DIR FINDINGS_DIR

$PYTHON - <<'PYEOF'
import json, os, math

def pct(data, p):
    if not data: return float("nan")
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])

def load_jsonl(path):
    if not os.path.exists(path): return []
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: recs.append(json.loads(line))
                except: pass
    return recs

def find_field(rec, candidates):
    for c in candidates:
        if c in rec: return c
    return None

def extract_stats(path):
    recs = load_jsonl(path)
    if not recs: return 0, [], []
    s = recs[0]
    lat_f  = find_field(s, ["total_chunk_time", "latency", "elapsed_time", "e2e_latency",
                             "total_time", "end2end_latency", "response_time"])
    ttft_f = find_field(s, ["first_chunk_time", "ttft", "first_token_latency",
                             "time_to_first_token", "first_token_time", "TTFT"])
    lats  = [r[lat_f]  for r in recs if lat_f  and lat_f  in r]
    ttfts = [r[ttft_f] for r in recs if ttft_f and ttft_f in r]
    return len(recs), ttfts, lats

def gpu_energy(gpu_log):
    if not os.path.exists(gpu_log): return float("nan")
    with open(gpu_log) as f:
        g = json.load(f)
    recs = g.get("records", [])
    if not recs: return float("nan")
    total_ws = sum(r["power_w"] for r in recs) * g["interval_s"]
    return total_ws / 3600  # Wh

d  = os.environ["DATE"]
ld = os.environ["LOG_DIR"]
fd = os.environ["FINDINGS_DIR"]

fmt = lambda v: f"{v:.3f}s" if not math.isnan(v) else "N/A"
fmte = lambda v: f"{v:.4f} Wh" if not math.isnan(v) else "N/A"

rows = []

# Include prior 1x baseline if log exists (different date, still useful)
bl_detail = f"{ld}/2026-06-30-burstgpt-detail.jsonl"
bl_gpu    = f"{ld}/2026-06-30-gpu.json"
if os.path.exists(bl_detail):
    n, ttfts, lats = extract_stats(bl_detail)
    energy = gpu_energy(bl_gpu)
    rows.append(("1x (2026-06-30)", n, ttfts, lats, energy))

for label in ["0.5x", "2x", "4x"]:
    detail  = f"{ld}/{d}-qps-{label}-burstgpt-detail.jsonl"
    gpu_log = f"{ld}/{d}-qps-{label}-gpu.json"
    n, ttfts, lats = extract_stats(detail)
    energy = gpu_energy(gpu_log)
    rows.append((label, n, ttfts, lats, energy))

lines = [
    f"# BurstGPT QPS Sweep Summary — {d}",
    "",
    "**Model:** Qwen2.5-0.5B-Instruct  |  **gpu-memory-utilization:** 0.9",
    "**Scales:** relative to calibrated base (1.2344×); 0.5× = lower QPS, 4× = higher QPS",
    "",
    "| Scale | N | P50 TTFT | P95 TTFT | P50 Lat | P95 Lat | Energy |",
    "|-------|---|----------|----------|---------|---------|--------|",
]
for label, n, ttfts, lats, energy in rows:
    lines.append(
        f"| {label} | {n} | {fmt(pct(ttfts,50))} | {fmt(pct(ttfts,95))} "
        f"| {fmt(pct(lats,50))} | {fmt(pct(lats,95))} | {fmte(energy)} |"
    )

lines += [
    "",
    "## Interpretation Guide",
    "- **P50/P95 TTFT**: measures prefill duration; rises under higher QPS as GPU is saturated",
    "- **P50/P95 Lat**: end-to-end time; rises with queue depth at high QPS",
    "- **Energy**: total GPU Wh during run; higher load ≠ proportionally higher energy (efficiency curve)",
    "",
    "## Per-scale findings",
]
for label in ["0.5x", "2x", "4x"]:
    lines.append(f"- [{label}]({d}-qps-{label}.md)")

out = "\n".join(lines) + "\n"
summary_path = f"{fd}/{d}-qps-sweep-summary.md"
with open(summary_path, "w") as f:
    f.write(out)
print(f"\n=== Summary written: {summary_path} ===\n")
print(out)
PYEOF

echo ""
echo "=== QPS sweep complete! ==="
