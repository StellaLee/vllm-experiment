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
TARGET_REQUESTS=45

mkdir -p "$LOG_DIR" "$FINDINGS_DIR"

# ===== Start vLLM =====
echo "=== Starting vLLM server ==="
$PYTHON -m vllm.entrypoints.api_server \
  --model $MODEL --port $PORT --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  > $LOG_DIR/${DATE}-2x4x-vllm.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

MON_PID=""
BENCH_PID=""
cleanup() {
  [ -n "$BENCH_PID" ] && { kill "$BENCH_PID" 2>/dev/null || true; }
  [ -n "$MON_PID"   ] && { kill "$MON_PID"   2>/dev/null || true; wait "$MON_PID" 2>/dev/null || true; }
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
  [ $i -eq 90 ] && { echo "ERROR: vLLM not ready"; exit 1; }
  printf "."; sleep 2
done
sleep 2

# ===== Sweep 2x and 4x =====
for MULT in 2.0 4.0; do
  SCALE=$($PYTHON -c "print('{:.6f}'.format($BASE_SCALE * $MULT))")
  LABEL=$(echo "$MULT" | awk '{v=$1+0; if(v==int(v)) printf "%dx",int(v); else printf "%sx",v}')

  echo ""
  echo "========================================================"
  echo " QPS ${LABEL}  |  --scale=${SCALE}  |  stopping at ${TARGET_REQUESTS} requests"
  echo "========================================================"

  GPU_LOG=$LOG_DIR/${DATE}-qps-${LABEL}-gpu.json
  DETAIL=$LOG_DIR/${DATE}-qps-${LABEL}-burstgpt-detail.jsonl
  BURSLOG=$LOG_DIR/${DATE}-qps-${LABEL}-burstgpt.jsonl
  FINDINGS=$FINDINGS_DIR/${DATE}-qps-${LABEL}.md

  # Start GPU monitor
  $PYTHON $EXPERIMENT_DIR/monitor_gpu.py --output "$GPU_LOG" &
  MON_PID=$!
  sleep 1

  # Start BurstGPT in background
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
    --scale=$SCALE &
  BENCH_PID=$!
  echo "burstgpt-bench PID: $BENCH_PID"

  # Wait until 45 requests complete or bench finishes naturally
  echo "Waiting for ${TARGET_REQUESTS} requests..."
  for i in $(seq 1 600); do
    if ! kill -0 "$BENCH_PID" 2>/dev/null; then
      echo "burstgpt-bench finished naturally."
      BENCH_PID=""
      break
    fi
    count=$(wc -l < "$DETAIL" 2>/dev/null || echo 0)
    if [ "$count" -ge "$TARGET_REQUESTS" ]; then
      echo "${count} requests reached — killing burstgpt-bench"
      kill "$BENCH_PID" 2>/dev/null || true
      wait "$BENCH_PID" 2>/dev/null || true
      BENCH_PID=""
      break
    fi
    [ $((i % 10)) -eq 0 ] && echo "  ... ${count}/${TARGET_REQUESTS} requests"
    sleep 2
  done

  # Stop GPU monitor
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  MON_PID=""

  # Analyze
  echo "--- Analyzing ${LABEL} ---"
  $PYTHON $EXPERIMENT_DIR/analyze.py \
    --gpu-log "$GPU_LOG" \
    --burstgpt-log "$DETAIL" \
    --output "$FINDINGS"

  echo "=== ${LABEL} done ==="
  sleep 5
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
    return total_ws / 3600

d  = os.environ["DATE"]
ld = os.environ["LOG_DIR"]
fd = os.environ["FINDINGS_DIR"]

fmt  = lambda v: f"{v:.3f}s" if not math.isnan(v) else "N/A"
fmte = lambda v: f"{v:.3f} Wh" if not math.isnan(v) else "N/A"

rows = []
# 0.5x (already analyzed)
for label in ["0.5x", "2x", "4x"]:
    detail  = f"{ld}/{d}-qps-{label}-burstgpt-detail.jsonl"
    gpu_log = f"{ld}/{d}-qps-{label}-gpu.json"
    n, ttfts, lats = extract_stats(detail)
    energy = gpu_energy(gpu_log)
    rows.append((label, n, ttfts, lats, energy))

# Include 2026-06-30 baseline (1x) if available
bl_detail = f"{ld}/2026-06-30-burstgpt-detail.jsonl"
bl_gpu    = f"{ld}/2026-06-30-gpu.json"
if os.path.exists(bl_detail):
    n, ttfts, lats = extract_stats(bl_detail)
    energy = gpu_energy(bl_gpu)
    rows.insert(1, ("1x (baseline 06-30)", n, ttfts, lats, energy))

lines = [
    f"# BurstGPT QPS Sweep Summary — {d}",
    "",
    "**Model:** Qwen2.5-0.5B-Instruct  |  **gpu-memory-utilization:** 0.9",
    "**Note:** 0.5× = lower QPS than baseline; 2×/4× = higher QPS",
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
    "## Per-scale findings",
    f"- [0.5x]({d}-qps-0.5x.md)",
    f"- [2x]({d}-qps-2x.md)",
    f"- [4x]({d}-qps-4x.md)",
]

out = "\n".join(lines) + "\n"
summary_path = f"{fd}/{d}-qps-sweep-summary.md"
with open(summary_path, "w") as f:
    f.write(out)
print(f"\n=== Summary: {summary_path} ===\n")
print(out)
PYEOF

echo "=== Sweep complete! ==="
