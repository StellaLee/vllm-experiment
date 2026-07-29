#!/bin/bash
# probe_gpuutil.sh -- does raising gpu-memory-utilization 0.90->0.95 deepen the decode batch?
# Run the 14B whale workload at fixed high concurrency (48) under each util. Read:
#   - KV pool size vLLM reports at startup (GPU blocks / cache tokens)
#   - decode depth (running batch) distribution from the hslo chunktrace
#   - db (signal_ms) distribution  -- does it climb off the flat ~20-38ms region?
#   - preempt count (the instability risk at high util)
# hslo mode is only used to emit the depth/db trace; budget value is irrelevant here.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
DATE=$(date +%Y-%m-%d); PORT=8050; CONC=48; NCONV=200
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 5; }
$PYTHON scripts/mlsys/hotpatch_hslo.py && $PYTHON scripts/mlsys/hotpatch_hslo_alphafloor.py
rm -f logs/gpuutil_ALLDONE

run_util(){ # $1=util
  local U=$1 TAG="u${1/./}"
  local FB="logs/${DATE}-gputil-${TAG}"
  local TRACE="${FB}-chunktrace.csv"; rm -f "$TRACE"
  echo ">>> gpu-util=$U  conc=$CONC"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 \
    DYNAMIC_CHUNK=1 CHUNK_MODE=hslo DYNAMIC_CHUNK_MIN=512 DYNAMIC_CHUNK_START=512 \
    DYNAMIC_CHUNK_SLO_MS=400 DYNAMIC_CHUNK_ALPHA_MIN_PREFILL=256 DYNAMIC_CHUNK_ALPHA_MIN=0.18 \
    DYNAMIC_CHUNK_TRACE=$TRACE \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 96 --max-num-batched-tokens 16384 --max-model-len 16384 \
    --tensor-parallel-size 2 --gpu-memory-utilization $U > ${FB}-server.log 2>&1 &
  local SV=$!
  for i in $(seq 1 120); do sleep 5; grep -q "Application startup complete" ${FB}-server.log && break
    [ "$i" = 120 ] && { echo "  SERVER TIMEOUT (util=$U)"; kill $SV 2>/dev/null; kill_ours; return; }; done
  # KV pool size vLLM logged at startup
  echo -n "  KV pool: "; grep -iE "GPU KV cache size|# GPU blocks|GPU blocks:|KV cache" ${FB}-server.log | head -2 | tr '\n' ' '; echo
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 --max-tokens 256 --concurrency $CONC \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
    --max-prompt-chars 50000 --pad-seed 1001 \
    --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
  local PREE=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)
  kill $SV 2>/dev/null; sleep 6; kill -9 $SV 2>/dev/null; kill_ours
  TRACE=$TRACE PREE=$PREE UTIL=$U $PYTHON - <<'PY'
import csv, os, statistics as st
f=os.environ['TRACE']
rows=[r for r in csv.DictReader(open(f))] if os.path.exists(f) else []
dep=[int(float(r['depth'])) for r in rows if r.get('depth')]
db=[float(r['signal_ms']) for r in rows if r.get('signal_ms')]
# steady-state: drop first 10% (warmup)
dep=dep[len(dep)//10:]; db=db[len(db)//10:]
def q(x,p):
    y=sorted(x); return y[min(len(y)-1,int(p/100*len(y)))] if y else 0
print(f"  util={os.environ['UTIL']}  preempt={os.environ['PREE']}  n={len(dep)}")
print(f"    depth : med={q(dep,50)}  p90={q(dep,90)}  max={max(dep) if dep else 0}")
print(f"    db(ms): med={q(db,50):.1f}  p90={q(db,90):.1f}  max={max(db) if db else 0:.1f}")
PY
}

run_util 0.90
run_util 0.95
touch logs/gpuutil_ALLDONE
echo "GPUUTIL PROBE DONE"
