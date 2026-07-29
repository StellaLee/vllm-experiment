#!/bin/bash
# run_decomp_bench.sh
# Ablation decomposition: chunk-only vs reorder-only for multi-turn c=15.
# Adds two conditions to the existing baseline/combined/aging results.
#
# Tags produced:
#   mt_chunk_c15   (DYNAMIC_CHUNK=1, PREFIX_REORDER=0)
#   mt_reorder_c15 (DYNAMIC_CHUNK=0, PREFIX_REORDER=1)

set -euo pipefail
cd /root/vllm-experiment

PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
NUM_CONVS=${NUM_CONVS:-200}
MAX_TURNS=${MAX_TURNS:-4}
MAX_TOKENS=${MAX_TOKENS:-128}
CONCURRENCY=15

SHAREGPT=data/sharegpt_v3.json
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_decomp_pid

mkdir -p "$LOG_DIR"

echo "======================================================="
echo "  Decomposition Ablation: chunk-only vs reorder-only"
echo "  Concurrency: ${CONCURRENCY}  |  Convs: ${NUM_CONVS}"
echo "======================================================="

start_server() {
    local label=$1
    shift
    echo ""
    echo "=== Starting server: ${label} ==="
    "$@" \
        $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
        > "$LOG_DIR/${DATE}-mt-${label}-server.log" 2>&1 &
    echo $! > "$PID_FILE"

    echo -n "Waiting for startup"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" \
                "$LOG_DIR/${DATE}-mt-${label}-server.log" 2>/dev/null; then
            echo " ready (${i}x5s)"
            return 0
        fi
        echo -n "."
    done
    echo " TIMEOUT"
    tail -20 "$LOG_DIR/${DATE}-mt-${label}-server.log"
    exit 1
}

stop_server() {
    if [ -f "$PID_FILE" ]; then
        echo "Stopping server (pid=$(cat "$PID_FILE"))..."
        kill "$(cat "$PID_FILE")" 2>/dev/null || true
        sleep 8
        rm -f "$PID_FILE"
    fi
}

run_replay() {
    local tag=$1
    local out_jsonl="$LOG_DIR/${DATE}-mt-${tag}.jsonl"
    local gpu_csv="/tmp/gpu_mon_mt_${tag}.csv"

    echo ""
    echo "--- Replaying: tag=${tag}  concurrency=${CONCURRENCY}  convs=${NUM_CONVS} ---"

    nvidia-smi --query-gpu=utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits --loop=1 \
        > "$gpu_csv" 2>/dev/null &
    local gpu_mon_pid=$!

    $PYTHON src/replay_sharegpt.py \
        --host localhost --port "$PORT" \
        --model "$MODEL" \
        --dataset "$SHAREGPT" \
        --num-convs "$NUM_CONVS" \
        --max-turns "$MAX_TURNS" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONCURRENCY" \
        --min-turns "$MAX_TURNS" \
        --output "$out_jsonl"

    kill "$gpu_mon_pid" 2>/dev/null || true
    wait "$gpu_mon_pid" 2>/dev/null || true
    $PYTHON - <<PYEOF
import sys
rows = [l.strip().split(",") for l in open("${gpu_csv}") if l.strip()]
gpu = [float(r[0]) for r in rows if len(r) >= 1 and r[0].strip()]
mem = [float(r[1]) for r in rows if len(r) >= 2 and r[1].strip()]
if gpu:
    print(f"  GPU util: {sum(gpu)/len(gpu):.1f}%  mem util: {sum(mem)/len(mem):.1f}%  ({len(gpu)} samples)")
PYEOF

    $PYTHON - <<PYEOF
import urllib.request, re, json
try:
    body = urllib.request.urlopen("http://localhost:${PORT}/metrics", timeout=5).read().decode()
except Exception as e:
    print(f"  WARNING: could not reach metrics: {e}")
    exit(0)

hits, queries = 0.0, 0.0
for name, val_dest in [("vllm:prefix_cache_hits_total", "hits"),
                        ("vllm:prefix_cache_queries_total", "queries")]:
    pat = re.compile(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([\d.e+\-]+)", re.MULTILINE)
    vals = [float(m) for m in pat.findall(body)]
    if vals:
        if val_dest == "hits": hits = sum(vals)
        else: queries = sum(vals)

rate = hits / queries if queries > 0 else 0.0
print(f"  KV hit rate: {rate:.1%}  ({hits:.0f}/{queries:.0f} blocks)")

summary = {"tag": "${tag}", "kv_hit_rate": rate, "kv_hits": hits, "kv_queries": queries}
out = "${out_jsonl}".replace(".jsonl", "-summary.json")
json.dump(summary, open(out, "w"), indent=2)
PYEOF
}

trap stop_server EXIT

# ── Condition A: chunk-only (DYNAMIC_CHUNK=1, PREFIX_REORDER=0) ───────────────
start_server "mt-chunk" env PREFIX_REORDER=0 DYNAMIC_CHUNK=1
run_replay "mt_chunk_c15"
stop_server

# ── Condition B: reorder-only (DYNAMIC_CHUNK=0, PREFIX_REORDER=1) ─────────────
start_server "mt-reorder" env PREFIX_REORDER=1 DYNAMIC_CHUNK=0
run_replay "mt_reorder_c15"
stop_server

echo ""
echo "=== Decomposition done. ==="
echo "    Analyze with: python3 src/analyze_multiturn.py --concurrency 15"
