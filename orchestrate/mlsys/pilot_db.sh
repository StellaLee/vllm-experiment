#!/bin/bash
# de-risk pilot: one hslo server, a 2-phase (8<->40) 90s run, read db + budget per phase.
# Validates the new --concurrency-schedule driver end-to-end AND calibrates db(conc) to pick the SLO.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
DATE=$(date +%Y-%m-%d); PORT=8050
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 8
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 4; }
$PYTHON scripts/mlsys/hotpatch_hslo.py && $PYTHON scripts/mlsys/hotpatch_hslo_alphafloor.py
TRACE=logs/${DATE}-pilotdb-chunktrace.csv; rm -f "$TRACE"
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 \
  DYNAMIC_CHUNK=1 CHUNK_MODE=hslo DYNAMIC_CHUNK_MIN=512 DYNAMIC_CHUNK_START=512 \
  DYNAMIC_CHUNK_SLO_MS=400 DYNAMIC_CHUNK_ALPHA_MIN_PREFILL=256 DYNAMIC_CHUNK_ALPHA_MIN=0.18 \
  DYNAMIC_CHUNK_TRACE=$TRACE \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 48 --max-num-batched-tokens 16384 --max-model-len 16384 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > logs/${DATE}-pilotdb-server.log 2>&1 &
SV=$!
for i in $(seq 1 120); do sleep 5; grep -q "Application startup complete" logs/${DATE}-pilotdb-server.log && break
  [ "$i" = 120 ] && { echo "SERVER TIMEOUT"; kill $SV; kill_ours; exit 1; }; done
echo "server up, running 2-phase pilot"
$PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
  --dataset "$DATASET" --num-convs 200 --max-turns 1 --min-turns 1 --max-tokens 256 \
  --concurrency-schedule "8@45,40@45" --duration 90 \
  --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
  --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
  --max-prompt-chars 50000 --pad-seed 1001 \
  --output logs/${DATE}-pilotdb-t1.jsonl > logs/${DATE}-pilotdb.client.log 2>&1 || true
kill $SV 2>/dev/null; sleep 6; kill -9 $SV 2>/dev/null; kill_ours
echo "=== db + budget by phase (schedule 8@45,40@45) ==="
SCHEDULE="8@45,40@45" $PYTHON - <<'PY'
import csv, os, sys, statistics as st
sys.path.insert(0,'scripts'); from replay_timing import parse_schedule, phase_at
import glob
f=sorted(glob.glob('logs/*-pilotdb-chunktrace.csv'))[-1]
rows=list(csv.DictReader(open(f)))
sched=parse_schedule(os.environ['SCHEDULE'])
active=[float(r['wall_s']) for r in rows if int(float(r['depth']))>0]
t0=min(active) if active else 0
from collections import defaultdict
b=defaultdict(lambda:{'db':[],'ch':[],'depth':[]})
for r in rows:
    w=float(r['wall_s'])
    if w<t0: continue
    idx,conc,_=phase_at(sched,w-t0)
    b[idx]['db'].append(float(r['signal_ms'])); b[idx]['ch'].append(int(float(r['chunk']))); b[idx]['depth'].append(int(float(r['depth'])))
def med(x): return st.median(x) if x else 0
for idx in sorted(b):
    d=b[idx]; conc=sched[idx][0]
    print(f"phase{idx} conc={conc:>3}: n={len(d['db']):>4} depth_med={med(d['depth']):>4.0f} db_med={med(d['db']):>6.1f}ms budget_med={med(d['ch']):>6.0f} budget_p10={sorted(d['ch'])[int(0.1*(len(d['ch'])-1))] if d['ch'] else 0:>5}")
PY
touch logs/pilotdb_ALLDONE
echo "PILOT DONE"
