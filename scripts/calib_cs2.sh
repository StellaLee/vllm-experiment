#!/bin/bash
# calib_cs2.sh -- find a sub-saturation open-loop rate for the padded regime.
# Starts the mono server once, runs short cv2=0 probes at several rates, prints
# median/p95 TTFT and wall-time. Overload shows as p95>>median and wall-time >> n/rate.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}
MAXLEN=16384; BUDGET=16384
LOG=logs; DATE=$(date +%Y-%m-%d); mkdir -p "$LOG"
PID=/tmp/vllm_calib_pid

$PYTHON scripts/patch_scheduler.py >/dev/null 2>&1
echo "=== starting mono server (budget=$BUDGET) ==="
env PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" --max-num-seqs 32 \
    --max-num-batched-tokens "$BUDGET" --max-model-len "$MAXLEN" \
    --gpu-memory-utilization 0.90 > "$LOG/${DATE}-calib-server.log" 2>&1 &
echo $! > "$PID"
trap 'kill "$(cat $PID)" 2>/dev/null || true' EXIT
echo -n "waiting"; for i in $(seq 1 72); do sleep 5
  if grep -q "Application startup complete" "$LOG/${DATE}-calib-server.log" 2>/dev/null; then echo " ready"; break; fi
  echo -n "."; done

for RATE in 1.5 2.5 3.5; do
  out="$LOG/${DATE}-calib-r${RATE}.jsonl"
  t0=$(date +%s)
  $PYTHON src/replay_sharegpt.py --host localhost --port "$PORT" --model "$MODEL" \
    --dataset data/sharegpt_v3.json --num-convs 30 --max-turns 3 --min-turns 3 \
    --max-tokens 128 --rate "$RATE" \
    --pad-mean-chars 8000 --pad-cv2 0 --pad-min 100 --pad-max 50000 \
    --output "$out" > "${out%.jsonl}.log" 2>&1 || true
  t1=$(date +%s)
  preempt=$(grep -c -i preempt "$LOG/${DATE}-calib-server.log" 2>/dev/null || echo 0)
  $PYTHON - "$out" "$RATE" "$((t1-t0))" "$preempt" <<'PY'
import json,sys,statistics as st
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
t=[r["ttft"]*1000 for r in rows if r.get("ttft") is not None]
t.sort()
med=st.median(t) if t else 0; p95=t[int(0.95*len(t))] if t else 0
print(f"rate={sys.argv[2]:>4}  n={len(t):3d}  wall={sys.argv[3]:>3}s  med_ttft={med:8.0f}ms  p95_ttft={p95:9.0f}ms  preempt_lines={sys.argv[4]}")
PY
done
echo "=== calib done ==="
