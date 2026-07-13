#!/bin/bash
# run_cs2_repl.sh -- REPLICATED Cs^2 crossover: mono vs chunk, grid {0,1,2,4} (drop the
# ultra-heavy-tail cv8 noise), 3 trials with distinct pad seeds, 400 req/point, single-turn,
# open-loop rate 2.0. Within a trial, mono & chunk use the SAME seed => identical prompts
# (paired). Server started once per arm; inner loops trial x Cs^2. Produces mean+-CI curve.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}; MAX_SEQS=${MAX_SEQS:-32}; MAXLEN=${MAXLEN:-16384}
MONO_BUDGET=${MONO_BUDGET:-16384}; CHUNK_BUDGET=${CHUNK_BUDGET:-512}; GPU_UTIL=${GPU_UTIL:-0.90}
TP=${TP:-1}   # set TP=2 on the 2x4090 box (also set CUDA_VISIBLE_DEVICES to a same-switch pair)
DATASET=${DATASET:-data/sharegpt_v3.json}
NUM_CONVS=${NUM_CONVS:-400}; MAX_TURNS=1; MAX_TOKENS=${MAX_TOKENS:-128}; RATE=${RATE:-2.0}
PAD_MEAN=${PAD_MEAN:-8000}; PAD_MIN=${PAD_MIN:-100}; PAD_MAX=${PAD_MAX:-50000}
SEED_BASE=${SEED_BASE:-1000}
CV2_GRID=${CV2_GRID:-"0 1 2 4"}
TRIALS=${TRIALS:-"1 2 3"}
PREFIX=cs2repl
LOG_DIR=logs; DATE=$(date +%Y-%m-%d); PID_FILE=/tmp/vllm_cs2repl_pid
mkdir -p "$LOG_DIR"

echo "==== Cs^2 REPLICATION | mono($MONO_BUDGET) vs chunk($CHUNK_BUDGET) | grid:${CV2_GRID} | trials:${TRIALS} | rate=$RATE n=$NUM_CONVS ===="
$PYTHON scripts/patch_scheduler.py

start_server() { local label=$1 budget=$2
  echo ""; echo "=== server: ${label} (budget=${budget}) ==="
  env PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port "$PORT" --max-num-seqs "$MAX_SEQS" \
      --max-num-batched-tokens "$budget" --max-model-len "$MAXLEN" \
      --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_UTIL" \
      > "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log" 2>&1 &
  echo $! > "$PID_FILE"; echo -n "waiting"
  for i in $(seq 1 72); do sleep 5
    if grep -q "Application startup complete" "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log" 2>/dev/null; then echo " ready"; return 0; fi
    echo -n "."; done
  echo " TIMEOUT"; tail -20 "$LOG_DIR/${DATE}-${PREFIX}-${label}-server.log"; exit 1; }
stop_server() { if [ -f "$PID_FILE" ]; then kill "$(cat "$PID_FILE")" 2>/dev/null || true; sleep 8; rm -f "$PID_FILE"; fi; }
trap stop_server EXIT

for spec in "mono ${MONO_BUDGET}" "chunk ${CHUNK_BUDGET}"; do
  set -- $spec; arm=$1; budget=$2
  start_server "$arm" "$budget"
  for tr in $TRIALS; do
    seed=$((SEED_BASE + tr))
    for cv2 in $CV2_GRID; do
      cvtag=$(echo "$cv2" | tr '.' 'p')
      out="$LOG_DIR/${DATE}-${PREFIX}-${arm}-cv${cvtag}-t${tr}.jsonl"
      echo "--- ${arm} cv2=${cv2} trial=${tr} seed=${seed} ---"
      $PYTHON src/replay_sharegpt.py --host localhost --port "$PORT" --model "$MODEL" \
        --dataset "$DATASET" --num-convs "$NUM_CONVS" --max-turns "$MAX_TURNS" --min-turns "$MAX_TURNS" \
        --max-tokens "$MAX_TOKENS" --rate "$RATE" \
        --pad-mean-chars "$PAD_MEAN" --pad-cv2 "$cv2" --pad-min "$PAD_MIN" --pad-max "$PAD_MAX" --pad-seed "$seed" \
        --output "$out" > "${out%.jsonl}.client.log" 2>&1
      echo "    done ($(grep -c . "$out" 2>/dev/null || echo 0) records)"
    done
  done
  stop_server
done
echo "=== cs2 replication done ==="
