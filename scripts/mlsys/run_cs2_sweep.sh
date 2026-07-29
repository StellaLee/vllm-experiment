#!/bin/bash
# run_cs2_sweep.sh
# GENUINE-TERM sweep. Open-loop Poisson arrivals, un-cacheable padded prefills whose
# per-request SIZE dispersion (Cs^2) we sweep at (approximately) fixed mean E[S_prefill].
# Two policy arms, reorder OFF, dynamic OFF:
#   mono   (budget=MONO_BUDGET >= max prefill)  -> prefill runs to completion (FCFS-like)
#   chunk  (budget=CHUNK_BUDGET, e.g. 512)      -> prefill processor-shared (PS)
# Prediction (eq. genuine, T_FCFS - T_PS = E[S]*(rho/(1-rho))*(Cs^2-1)/2):
#   at Cs^2<1  chunk is WORSE than mono on mean TTFT (head-of-line term negative);
#   at Cs^2>1  chunk becomes BETTER  -> the genuine term switching on, on ONE GPU.
# Server started once per arm; inner loop over the Cs^2 grid (identical padded prompts
# across arms via the deterministic --pad-seed, so the comparison is paired).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAX_SEQS=${MAX_SEQS:-32}
MAXLEN=${MAXLEN:-16384}
MONO_BUDGET=${MONO_BUDGET:-16384}     # >= max prefill -> monolithic (FCFS-like)
CHUNK_BUDGET=${CHUNK_BUDGET:-512}     # Sarathi-style chunk -> processor-sharing
GPU_UTIL=${GPU_UTIL:-0.90}

# workload
DATASET=${DATASET:-data/sharegpt_v3.json}
NUM_CONVS=${NUM_CONVS:-120}
MAX_TURNS=${MAX_TURNS:-3}
MAX_TOKENS=${MAX_TOKENS:-128}         # short decode: keep the signal on TTFT, not TPOT
RATE=${RATE:-2.5}                     # conv/s open-loop (set from calibration)
PAD_MEAN=${PAD_MEAN:-8000}            # ~1750 prefill tokens: expensive prefill (gate open)
PAD_MIN=${PAD_MIN:-100}
PAD_MAX=${PAD_MAX:-50000}             # ~11k-token cap: keeps KV safe under concurrency
PAD_SEED=${PAD_SEED:-12345}
CV2_GRID=${CV2_GRID:-"0 0.5 1 2 4 8"}
TRIAL=${TRIAL:-t1}

PREFIX=cs2sweep
LOG_DIR=logs
DATE=$(date +%Y-%m-%d)
PID_FILE=/tmp/vllm_cs2_pid
mkdir -p "$LOG_DIR"

echo "==== Cs^2 genuine-term sweep | open-loop rate=${RATE} conv/s | pad_mean=${PAD_MEAN}c out=${MAX_TOKENS} ===="
echo "  arms: mono(${MONO_BUDGET}) vs chunk(${CHUNK_BUDGET}) | reorder OFF dynamic OFF | Cs2 grid: ${CV2_GRID}"

echo "=== patch vLLM (canonical apply_patches.sh; flags OFF => native static) ==="
PYTHON="$PYTHON" bash scripts/apply_patches.sh

start_server() {
    local label=$1; local budget=$2
    echo ""; echo "=== server: ${label} (budget=${budget}, maxlen=${MAXLEN}) ==="
    env PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
        --max-num-batched-tokens "$budget" --max-model-len "$MAXLEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        > "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log" 2>&1 &
    echo $! > "$PID_FILE"
    echo -n "waiting"
    for i in $(seq 1 72); do
        sleep 5
        if grep -q "Application startup complete" "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log" 2>/dev/null; then echo " ready"; return 0; fi
        echo -n "."
    done
    echo " TIMEOUT"; tail -25 "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log"; exit 1
}
stop_server() { if [ -f "$PID_FILE" ]; then kill "$(cat "$PID_FILE")" 2>/dev/null || true; sleep 8; rm -f "$PID_FILE"; fi; }

run_replay() {
    local arm=$1; local cv2=$2
    local cvtag=$(echo "$cv2" | tr '.' 'p')
    local out="$LOG_DIR/${DATE}-${PREFIX}-${arm}-cv${cvtag}-${TRIAL}.jsonl"
    echo "--- replay ${arm} cv2=${cv2} -> $(basename "$out") ---"
    $PYTHON src/replay_sharegpt.py --host localhost --port "$PORT" --model "$MODEL" \
        --dataset "$DATASET" --num-convs "$NUM_CONVS" \
        --max-turns "$MAX_TURNS" --min-turns "$MAX_TURNS" --max-tokens "$MAX_TOKENS" \
        --rate "$RATE" \
        --pad-mean-chars "$PAD_MEAN" --pad-cv2 "$cv2" \
        --pad-min "$PAD_MIN" --pad-max "$PAD_MAX" --pad-seed "$PAD_SEED" \
        --output "$out" > "${out%.jsonl}.client.log" 2>&1
    # quick preemption check from the server log
    local preempt=$(grep -c -i "preempt" "$LOG_DIR/${DATE}-${PREFIX}-${arm}-server.log" 2>/dev/null || true)
    echo "    done ($(grep -c . "$out" 2>/dev/null || echo 0) records; server 'preempt' log lines: ${preempt})"
}

trap stop_server EXIT

for spec in "mono ${MONO_BUDGET}" "chunk ${CHUNK_BUDGET}"; do
    set -- $spec; arm=$1; budget=$2
    start_server "$arm" "$budget"
    for cv2 in $CV2_GRID; do
        run_replay "$arm" "$cv2"
    done
    stop_server
done

echo "=== cs2 sweep done: logs/${DATE}-${PREFIX}-*.jsonl ==="
