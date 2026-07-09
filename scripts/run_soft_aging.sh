#!/bin/bash
# run_soft_aging.sh
# 2 trials of soft-aging reorder-only (PREFIX_REORDER=1, DYNAMIC_CHUNK=0)
# AGING_ALPHA=0.3: score = hit_ratio + 0.3 * log1p(wait_seconds)
set -euo pipefail
cd /root/vllm-experiment

PYTHON=/root/miniconda3/bin/python3
MODEL=/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct
PORT=8000
MAX_SEQS=32
NUM_CONVS=200
MAX_TURNS=4
MAX_TOKENS=128
CONCURRENCY=15
SHAREGPT=data/sharegpt_v3.json
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_soft_pid

mkdir -p "$LOG_DIR"

start_server() {
    local label=$1
    shift
    echo "=== Starting server: ${label} ==="
    "$@" \
        $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
        > "$LOG_DIR/${DATE}-mt-${label}-server.log" 2>&1 &
    echo $! > "$PID_FILE"
    echo -n "Waiting"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" "$LOG_DIR/${DATE}-mt-${label}-server.log" 2>/dev/null; then
            echo " ready (${i}x5s)"; return 0
        fi
        echo -n "."
    done
    echo " TIMEOUT"; tail -20 "$LOG_DIR/${DATE}-mt-${label}-server.log"; exit 1
}

stop_server() {
    if [ -f "$PID_FILE" ]; then
        echo "Stopping server (pid=$(cat $PID_FILE))..."
        kill "$(cat $PID_FILE)" 2>/dev/null || true
        sleep 8
        rm -f "$PID_FILE"
    fi
}

run_replay() {
    local tag=$1
    local out="$LOG_DIR/${DATE}-mt-${tag}.jsonl"
    echo "--- Running: tag=${tag} concurrency=${CONCURRENCY} ---"

    nvidia-smi --query-gpu=utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits --loop=1 > /tmp/gpu_mon_${tag}.csv 2>/dev/null &
    local gpid=$!

    $PYTHON src/replay_sharegpt.py \
        --host localhost --port "$PORT" \
        --model "$MODEL" \
        --dataset "$SHAREGPT" \
        --num-convs "$NUM_CONVS" \
        --max-turns "$MAX_TURNS" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONCURRENCY" \
        --min-turns "$MAX_TURNS" \
        --output "$out"

    kill "$gpid" 2>/dev/null || true; wait "$gpid" 2>/dev/null || true

    $PYTHON - <<PYEOF
rows = [l.strip().split(",") for l in open("/tmp/gpu_mon_${tag}.csv") if l.strip()]
gpu = [float(r[0]) for r in rows if len(r)>=1 and r[0].strip()]
mem = [float(r[1]) for r in rows if len(r)>=2 and r[1].strip()]
if gpu:
    print(f"  GPU util: {sum(gpu)/len(gpu):.1f}%  mem: {sum(mem)/len(mem):.1f}%")
PYEOF

    $PYTHON - <<PYEOF
import urllib.request, re, json
try:
    body = urllib.request.urlopen("http://localhost:${PORT}/metrics", timeout=5).read().decode()
except Exception as e:
    print(f"  WARNING: metrics unreachable: {e}"); exit(0)
hits, queries = 0.0, 0.0
for name, dest in [("vllm:prefix_cache_hits_total","hits"),("vllm:prefix_cache_queries_total","queries")]:
    vals = [float(m) for m in re.compile(r"^"+re.escape(name)+r"(?:\{[^}]*\})?\s+([\d.e+\-]+)",re.MULTILINE).findall(body)]
    if vals:
        if dest=="hits": hits=sum(vals)
        else: queries=sum(vals)
rate = hits/queries if queries>0 else 0.0
print(f"  KV hit rate: {rate:.1%}  ({hits:.0f}/{queries:.0f} blocks)")
PYEOF
}

trap stop_server EXIT

echo "======================================================="
echo "  Soft Aging Reorder: 2 trials (AGING_ALPHA=0.3)"
echo "  PREFIX_REORDER=1  DYNAMIC_CHUNK=0  AGING_ALPHA=0.3"
echo "======================================================="

# Trial 1
start_server "soft_aging_t1" env PREFIX_REORDER=1 DYNAMIC_CHUNK=0 AGING_ALPHA=0.3
run_replay "mt_soft_aging_c15_t1"
stop_server

# Trial 2
start_server "soft_aging_t2" env PREFIX_REORDER=1 DYNAMIC_CHUNK=0 AGING_ALPHA=0.3
run_replay "mt_soft_aging_c15_t2"
stop_server

echo "=== Done. Analyze with: python3 /tmp/analyze_soft_aging.py ==="
