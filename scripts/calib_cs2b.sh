#!/bin/bash
# calib_cs2b.sh -- single-turn open-loop capacity probe for the padded regime.
# Starts mono server once; runs cv2=0 at several rates with a FULL-length run so the
# queue has time to build; reports first40 vs last40 median TTFT. Sub-saturation =
# last ~ first (bounded); saturation = last >> first (queue growing).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/root/miniconda3/bin/python3}
MODEL=${MODEL:-/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${PORT:-8000}; MAXLEN=16384; BUDGET=16384
NUM=${NUM:-140}
LOG=logs; DATE=$(date +%Y-%m-%d); mkdir -p "$LOG"; PID=/tmp/vllm_calibb_pid

$PYTHON scripts/patch_scheduler.py >/dev/null 2>&1
echo "=== starting mono server ==="
env PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" --max-num-seqs 32 \
    --max-num-batched-tokens "$BUDGET" --max-model-len "$MAXLEN" \
    --gpu-memory-utilization 0.90 > "$LOG/${DATE}-calibb-server.log" 2>&1 &
echo $! > "$PID"
trap 'kill "$(cat $PID)" 2>/dev/null || true' EXIT
echo -n "waiting"; for i in $(seq 1 72); do sleep 5
  if grep -q "Application startup complete" "$LOG/${DATE}-calibb-server.log" 2>/dev/null; then echo " ready"; break; fi
  echo -n "."; done

for RATE in ${RATES:-1.0 1.5 2.0}; do
  out="$LOG/${DATE}-calibb-r${RATE}.jsonl"
  $PYTHON src/replay_sharegpt.py --host localhost --port "$PORT" --model "$MODEL" \
    --dataset data/sharegpt_v3.json --num-convs "$NUM" --max-turns 1 --min-turns 1 \
    --max-tokens 128 --rate "$RATE" \
    --pad-mean-chars 8000 --pad-cv2 0 --pad-min 100 --pad-max 50000 \
    --output "$out" > "${out%.jsonl}.log" 2>&1 || true
  $PYTHON - "$out" "$RATE" <<'PY'
import json,sys,statistics as st
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
tt=sorted((r["ts"],r["ttft"]*1000) for r in rows if r.get("ttft"))
v=[x for _,x in tt]; f=[x for _,x in tt[:40]]; l=[x for _,x in tt[-40:]]
fm=st.median(f) if f else 0; lm=st.median(l) if l else 0
flag="BOUNDED (sub-sat)" if lm<2.0*fm else "GROWING (saturated)"
print(f"rate={sys.argv[2]:>4} n={len(v):3d} med={st.median(v):6.0f}ms  first40={fm:6.0f}  last40={lm:7.0f}  -> {flag}")
PY
done
echo "=== calibb done ==="
