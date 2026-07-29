#!/bin/bash
# probe_7b_depth.sh -- does the 7B (bigger KV headroom) reach a DEEP decode batch, and does db
# climb off the flat ~30ms region as depth grows? Same whale workload, TP2, gpu-util 0.90,
# max-num-seqs 256 (so seqs isn't the cap). One hslo server per concurrency; read depth+db.
# SUCCESS = at high conc, depth reaches the hundreds AND db becomes a big fraction of the 400ms SLO
# (that's the load-sensitive regime where a controller's optimal chunk could finally move).
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-7B-Instruct
DATASET=data/sharegpt_v3.json
DATE=$(date +%Y-%m-%d); PORT=8050; NCONV=300
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 5; }
$PYTHON scripts/mlsys/hotpatch_hslo.py && $PYTHON scripts/mlsys/hotpatch_hslo_alphafloor.py
rm -f logs/probe7b_ALLDONE

run_conc(){ # $1=concurrency
  local C=$1
  local FB="logs/${DATE}-probe7b-c${C}"
  local TRACE="${FB}-chunktrace.csv"; rm -f "$TRACE"
  echo ">>> 7B  conc=$C"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 \
    DYNAMIC_CHUNK=1 CHUNK_MODE=hslo DYNAMIC_CHUNK_MIN=512 DYNAMIC_CHUNK_START=512 \
    DYNAMIC_CHUNK_SLO_MS=400 DYNAMIC_CHUNK_ALPHA_MIN_PREFILL=256 DYNAMIC_CHUNK_ALPHA_MIN=0.18 \
    DYNAMIC_CHUNK_TRACE=$TRACE \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  for i in $(seq 1 120); do sleep 5; grep -q "Application startup complete" ${FB}-server.log && break
    [ "$i" = 120 ] && { echo "  SERVER TIMEOUT (conc=$C)"; kill $SV 2>/dev/null; kill_ours; return; }; done
  echo -n "  KV pool: "; grep -iE "Available KV cache memory|GPU KV cache size" ${FB}-server.log | head -1 | sed 's/.*INFO[^]]*] //'
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 --max-tokens 256 --concurrency $C \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
    --max-prompt-chars 50000 --pad-seed 1001 \
    --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
  local PREE=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)
  kill $SV 2>/dev/null; sleep 6; kill -9 $SV 2>/dev/null; kill_ours
  TRACE=$TRACE PREE=$PREE CONC=$C SLO=400 $PYTHON - <<'PY'
import csv, os
f=os.environ['TRACE']
rows=[r for r in csv.DictReader(open(f))] if os.path.exists(f) else []
dep=[int(float(r['depth'])) for r in rows if r.get('depth')]
db=[float(r['signal_ms']) for r in rows if r.get('signal_ms')]
dep=dep[len(dep)//10:]; db=db[len(db)//10:]
def q(x,p):
    y=sorted(x); return y[min(len(y)-1,int(p/100*len(y)))] if y else 0
slo=float(os.environ['SLO']); dbm=q(db,50)
print(f"  conc={os.environ['CONC']:>3} preempt={os.environ['PREE']} n={len(dep)}  "
      f"depth med/p90/max={q(dep,50)}/{q(dep,90)}/{max(dep) if dep else 0}  "
      f"db med/p90/max={dbm:.1f}/{q(db,90):.1f}/{max(db) if db else 0:.1f}ms  "
      f"db/SLO={100*dbm/slo:.0f}%")
PY
}

run_conc 32
run_conc 96
run_conc 192
touch logs/probe7b_ALLDONE
echo "PROBE7B DONE"
