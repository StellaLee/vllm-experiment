#!/bin/bash
# run_chunk_hyst.sh
# 2 trials of chunk-only with hysteresis (DYNAMIC_CHUNK_HOLD=3).
# Compare against existing chunk_t1/chunk_t2 (old bang-bang, HOLD=1 effectively).
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
PID_FILE=/tmp/vllm_hyst_pid

mkdir -p "$LOG_DIR"

start_server() {
    local label=$1; shift
    echo "=== Starting server: ${label} ==="
    "$@" \
        $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
        > "$LOG_DIR/${DATE}-mt-${label}-server.log" 2>&1 &
    echo $! > "$PID_FILE"
    echo -n "Waiting"
    for i in $(seq 1 72); do
        sleep 5
        grep -q "Application startup complete" "$LOG_DIR/${DATE}-mt-${label}-server.log" 2>/dev/null && { echo " ready (${i}x5s)"; return 0; }
        echo -n "."
    done
    echo " TIMEOUT"; tail -20 "$LOG_DIR/${DATE}-mt-${label}-server.log"; exit 1
}

stop_server() {
    [ -f "$PID_FILE" ] && { kill "$(cat $PID_FILE)" 2>/dev/null || true; sleep 8; rm -f "$PID_FILE"; echo "Server stopped."; }
}

run_replay() {
    local tag=$1
    local out="$LOG_DIR/${DATE}-mt-${tag}.jsonl"
    echo "--- Running: tag=${tag} concurrency=${CONCURRENCY} ---"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits --loop=1 > /tmp/gpu_${tag}.csv 2>/dev/null &
    local gpid=$!
    $PYTHON src/replay_sharegpt.py \
        --host localhost --port "$PORT" --model "$MODEL" \
        --dataset "$SHAREGPT" --num-convs "$NUM_CONVS" \
        --max-turns "$MAX_TURNS" --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONCURRENCY" --min-turns "$MAX_TURNS" \
        --output "$out"
    kill "$gpid" 2>/dev/null || true; wait "$gpid" 2>/dev/null || true
    $PYTHON - <<PYEOF
rows=[l.strip().split(",") for l in open("/tmp/gpu_${tag}.csv") if l.strip()]
gpu=[float(r[0]) for r in rows if len(r)>=1 and r[0].strip()]
mem=[float(r[1]) for r in rows if len(r)>=2 and r[1].strip()]
if gpu: print(f"  GPU util: {sum(gpu)/len(gpu):.1f}%  mem: {sum(mem)/len(mem):.1f}%")
PYEOF
    $PYTHON - <<PYEOF
import urllib.request, re
try: body=urllib.request.urlopen("http://localhost:${PORT}/metrics",timeout=5).read().decode()
except Exception as e: print(f"  WARNING: metrics: {e}"); exit(0)
h,q=0.0,0.0
for n,d in [("vllm:prefix_cache_hits_total","h"),("vllm:prefix_cache_queries_total","q")]:
    vals=[float(m) for m in re.compile(r"^"+re.escape(n)+r"(?:\{[^}]*\})?\s+([\d.e+\-]+)",re.MULTILINE).findall(body)]
    if vals:
        if d=="h": h=sum(vals)
        else: q=sum(vals)
print(f"  KV hit rate: {h/q:.1%} ({h:.0f}/{q:.0f} blocks)" if q else "  KV: no data")
PYEOF
}

trap stop_server EXIT

echo "======================================================="
echo "  Chunk Hysteresis: 2 trials (DYNAMIC_CHUNK_HOLD=3)"
echo "  PREFIX_REORDER=0  DYNAMIC_CHUNK=1  HOLD=3"
echo "======================================================="

start_server "chunk_hyst_t1" env PREFIX_REORDER=0 DYNAMIC_CHUNK=1 DYNAMIC_CHUNK_HOLD=3
run_replay "mt_chunk_hyst_c15_t1"
stop_server

start_server "chunk_hyst_t2" env PREFIX_REORDER=0 DYNAMIC_CHUNK=1 DYNAMIC_CHUNK_HOLD=3
run_replay "mt_chunk_hyst_c15_t2"
stop_server

echo "=== Done ==="
