#!/bin/bash
# reprobe_alpha.sh -- measure alpha(low)/alpha(high) in the ALPHA-RESPONSIVE regime (pilot config:
# seqs 48, whale-frac 0.15), so we can set the full-run SLO precisely. Run hslo-only at SLO=400
# (small-chunk regime where alpha moves), schedule 8<->48, read db/alpha/budget/depth per phase.
# DECISION OUTPUT: with measured (db,alpha) per phase, the ideal full-run SLO = midpoint of
#   [2048*a_low+db_low, 2048*a_high+db_high] -> 2048 PASSES low, VIOLATES high.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
DATE=$(date +%Y-%m-%d); PORT=8050
SCHED="8@60,48@60"; DUR=160; SLO=400
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 8
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 4; }
$PYTHON scripts/mlsys/hotpatch_hslo.py && $PYTHON scripts/mlsys/hotpatch_hslo_alphafloor.py
TRACE=logs/${DATE}-reprobe-chunktrace.csv; rm -f "$TRACE" logs/reprobe_ALLDONE
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 \
  DYNAMIC_CHUNK=1 CHUNK_MODE=hslo DYNAMIC_CHUNK_MIN=512 DYNAMIC_CHUNK_START=512 \
  DYNAMIC_CHUNK_SLO_MS=$SLO DYNAMIC_CHUNK_ALPHA_MIN_PREFILL=256 DYNAMIC_CHUNK_ALPHA_MIN=0.18 \
  DYNAMIC_CHUNK_TRACE=$TRACE \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 48 --max-num-batched-tokens 16384 --max-model-len 16384 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > logs/${DATE}-reprobe-server.log 2>&1 &
SV=$!
for i in $(seq 1 120); do sleep 5; grep -q "Application startup complete" logs/${DATE}-reprobe-server.log && break
  [ "$i" = 120 ] && { echo "SERVER TIMEOUT"; kill $SV; kill_ours; exit 1; }; done
echo "server up (SLO=$SLO seqs=48 frac=0.15), running $SCHED for ${DUR}s"
$PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
  --dataset "$DATASET" --num-convs 300 --max-turns 1 --min-turns 1 --max-tokens 256 \
  --concurrency-schedule "$SCHED" --duration $DUR \
  --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
  --whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 \
  --max-prompt-chars 50000 --pad-seed 1001 \
  --output logs/${DATE}-reprobe-t1.jsonl > logs/${DATE}-reprobe.client.log 2>&1 || true
kill $SV 2>/dev/null; sleep 6; kill -9 $SV 2>/dev/null; kill_ours
PREE=$(grep -c -i preempt logs/${DATE}-reprobe-server.log 2>/dev/null || echo 0)
echo "=== reprobe: db/alpha/budget by phase ($SCHED, SLO=$SLO)  preempt=$PREE ==="
SCHEDULE="$SCHED" SLO=$SLO $PYTHON - <<'PY'
import csv, os, sys, statistics as st, glob
sys.path.insert(0,'scripts'); from replay_timing import parse_schedule, phase_at
SLO=float(os.environ['SLO'])
f=sorted(glob.glob('logs/*-reprobe-chunktrace.csv'))[-1]
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
    if 512<ch<16384: b[idx]['a'].append((SLO-db)/ch)
def med(x): return st.median(x) if x else 0
res={}
print(f"{'phase':>5} {'conc':>4} {'n':>5} {'depth_med':>9} {'db_med':>7} {'bud_med':>7} {'alpha_med':>9} | 2048 step")
for idx in sorted(b):
    d=b[idx]; conc=sched[idx][0]; a=med(d['a']); dbm=med(d['db'])
    res[conc]=(dbm,a); s=dbm+2048*a if a else 0
    print(f"{idx:>5} {conc:>4} {len(d['db']):>5} {med(d['depth']):>9.0f} {dbm:>6.1f}m {med(d['ch']):>7.0f} {a:>9.3f} | {s:>6.0f}ms")
# ideal full-run SLO: midpoint of 2048's low-step and high-step -> 2048 passes low, violates high
if len(res)>=2:
    lo=min(res); hi=max(res)
    s_lo=res[lo][0]+2048*res[lo][1]; s_hi=res[hi][0]+2048*res[hi][1]
    print(f"\n2048 step: low(conc{lo})={s_lo:.0f}ms  high(conc{hi})={s_hi:.0f}ms")
    if s_hi>s_lo:
        slo=round((s_lo+s_hi)/2/10)*10
        print(f"==> RECOMMENDED full-run SLO_MS = {slo}  (2048 passes low {s_lo:.0f}<{slo}, violates high {s_hi:.0f}>{slo})")
        for conc,(dbm,a) in sorted(res.items()):
            if a: print(f"    hslo budget @conc{conc}: {(slo-dbm)/a:.0f}  (step at that budget = {slo}ms)")
    else:
        print("==> alpha did NOT rise with load (s_hi<=s_lo): no usable split; moving optimum absent on this HW.")
PY
touch logs/reprobe_ALLDONE
echo "REPROBE DONE"
