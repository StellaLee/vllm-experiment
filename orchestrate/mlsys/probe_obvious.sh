#!/bin/bash
# probe_obvious.sh -- de-risk the "obvious win" E2 settings BEFORE the 30-min 4-arm run.
# One hslo server, SLO=600, schedule 12<->64, whale-frac 0.10, max-seqs 96. Read per-phase:
#   depth_med (did the high phase reach a deep decode batch, D>=40?)
#   budget_med (did hslo pick <<2048 in high -> static-2048 would VIOLATE 600ms?)
#   implied alpha = (SLO-db)/budget on unclamped steps.
# SUCCESS: high-phase depth_med>=40 AND budget_med<=~1200 ; low-phase budget_med>=~2400.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
DATE=$(date +%Y-%m-%d); PORT=8050
SCHED="12@60,64@60"; DUR=160; SLO=600
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 8
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 4; }
$PYTHON scripts/mlsys/hotpatch_hslo.py && $PYTHON scripts/mlsys/hotpatch_hslo_alphafloor.py
TRACE=logs/${DATE}-probeobv-chunktrace.csv; rm -f "$TRACE" logs/probeobv_ALLDONE
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 \
  DYNAMIC_CHUNK=1 CHUNK_MODE=hslo DYNAMIC_CHUNK_MIN=512 DYNAMIC_CHUNK_START=512 \
  DYNAMIC_CHUNK_SLO_MS=$SLO DYNAMIC_CHUNK_ALPHA_MIN_PREFILL=256 DYNAMIC_CHUNK_ALPHA_MIN=0.18 \
  DYNAMIC_CHUNK_TRACE=$TRACE \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 96 --max-num-batched-tokens 16384 --max-model-len 16384 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > logs/${DATE}-probeobv-server.log 2>&1 &
SV=$!
for i in $(seq 1 120); do sleep 5; grep -q "Application startup complete" logs/${DATE}-probeobv-server.log && break
  [ "$i" = 120 ] && { echo "SERVER TIMEOUT"; kill $SV; kill_ours; exit 1; }; done
echo "server up (SLO=$SLO seqs=96), running probe $SCHED for ${DUR}s"
$PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
  --dataset "$DATASET" --num-convs 400 --max-turns 1 --min-turns 1 --max-tokens 256 \
  --concurrency-schedule "$SCHED" --duration $DUR \
  --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
  --whale-frac 0.10 --whale-min-chars 44000 --whale-max-chars 50000 \
  --max-prompt-chars 50000 --pad-seed 1001 \
  --output logs/${DATE}-probeobv-t1.jsonl > logs/${DATE}-probeobv.client.log 2>&1 || true
kill $SV 2>/dev/null; sleep 6; kill -9 $SV 2>/dev/null; kill_ours
PREE=$(grep -c -i preempt logs/${DATE}-probeobv-server.log 2>/dev/null || echo 0)
echo "=== probe: db/depth/budget by phase (schedule $SCHED, SLO=$SLO)  preempt=$PREE ==="
SCHEDULE="$SCHED" SLO=$SLO $PYTHON - <<'PY'
import csv, os, sys, statistics as st, glob
sys.path.insert(0,'scripts'); from replay_timing import parse_schedule, phase_at
SLO=float(os.environ['SLO'])
f=sorted(glob.glob('logs/*-probeobv-chunktrace.csv'))[-1]
rows=list(csv.DictReader(open(f)))
sched=parse_schedule(os.environ['SCHEDULE'])
active=[float(r['wall_s']) for r in rows if int(float(r['depth']))>0]
t0=min(active) if active else 0
from collections import defaultdict
b=defaultdict(lambda:{'db':[],'ch':[],'depth':[],'a':[]})
for r in rows:
    w=float(r['wall_s'])
    if w<t0: continue
    idx,conc,_=phase_at(sched,w-t0)
    db=float(r['signal_ms']); ch=int(float(r['chunk'])); dp=int(float(r['depth']))
    b[idx]['db'].append(db); b[idx]['ch'].append(ch); b[idx]['depth'].append(dp)
    if 512<ch<16384:  # unclamped -> back out effective alpha
        b[idx]['a'].append((SLO-db)/ch)
def med(x): return st.median(x) if x else 0
def p10(x): return sorted(x)[int(0.1*(len(x)-1))] if x else 0
print(f"{'phase':>5} {'conc':>4} {'n':>5} {'depth_med':>9} {'db_med':>7} {'bud_med':>7} {'bud_p10':>7} {'alpha':>6} | static-2048 step@this-alpha")
for idx in sorted(b):
    d=b[idx]; conc=sched[idx][0]; a=med(d['a']); dbm=med(d['db'])
    s2048 = dbm + 2048*a if a else 0
    verdict = "VIOLATES" if s2048>SLO else "ok" if s2048 else "?"
    print(f"{idx:>5} {conc:>4} {len(d['db']):>5} {med(d['depth']):>9.0f} {dbm:>6.1f}m {med(d['ch']):>7.0f} {p10(d['ch']):>7} {a:>6.3f} | {s2048:>6.0f}ms {verdict}")
print(f"\nSUCCESS if high phase (conc 64): depth_med>=40 AND static-2048 step VIOLATES {SLO}ms (bud_med<<2048).")
print( "         and low phase (conc 12): bud_med>=~2400 (hslo opens above 2048 -> room to beat 512 on TTFT).")
PY
touch logs/probeobv_ALLDONE
echo "PROBE DONE"
