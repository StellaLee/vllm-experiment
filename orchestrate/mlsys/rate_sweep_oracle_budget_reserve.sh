#!/bin/bash
# rate_sweep_oracle_budget_reserve.sh -- oracle budget switch + decode-reserve fix, same
# non-stationary schedule as rate_sweep_oracle_ns.sh / rate_sweep_oracle_budget.sh, reusing
# their existing mono/static512 baselines and running only the new arm.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
PORT=8050
WHALE_MIN=10000
WHALE_MAX=50000
MAXTOK=1024
LO_S=120
HI_S=120
HI_FRAC=0.3
RATE=1.0
DUR=720
SCHED="${RATE}:0.0@${LO_S},${RATE}:${HI_FRAC}@${HI_S}"
DATE=$(date +%Y-%m-%d)
SCHED_PY=/root/pli/venv-vllm023/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py

kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server.*port $PORT" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023.*port $PORT" && kill -9 $pid 2>/dev/null; done; sleep 6; }

cp ${SCHED_PY}.pristine ${SCHED_PY}
$PYTHON scripts/hotpatch_oracle_budget_reserve.py || { log "PATCH FAILED"; touch logs/ratesweep_oraclebudgetreserve_FAILED; exit 1; }

ARM="nsoraclebudgetreserve"
FB="logs/${DATE}-lgate-b${ARM}"
FB_TRACE="${FB}-oracletrace.csv"
rm -f "${FB}-t1.jsonl" "$FB_TRACE"

log "[$ARM] starting server (ORACLE_BUDGET_RESERVE)"
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
  ORACLE_BUDGET_RESERVE=1 ORACLE_LO_S=$LO_S ORACLE_HI_S=$HI_S \
  ORACLE_LO_BUDGET=16384 ORACLE_HI_BUDGET=512 ORACLE_TRACE="$FB_TRACE" \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
SV=$!
UP=0
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" ${FB}-server.log && { UP=1; break; }
done
if [ "$UP" = 0 ]; then
  log "[$ARM] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
  touch logs/ratesweep_oraclebudgetreserve_FAILED; exit 1
fi

$PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
  --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens $MAXTOK \
  --phase-schedule "$SCHED" --duration $DUR \
  --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
  --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX --whale-pareto-alpha 0 \
  --max-prompt-chars 50000 --pad-seed 1001 \
  --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
log "  [$ARM] recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
cp ${SCHED_PY}.pristine ${SCHED_PY}

$PYTHON -c "
import csv
rows = list(csv.reader(open('${FB_TRACE}')))
if rows:
    n = len(rows)
    b = [int(r[2]) for r in rows]
    n_hi = sum(1 for x in b if x == 512)
    print(f'[$ARM] n_rounds={n} hi_rounds={n_hi} ({100*n_hi/n:.1f}%) lo_rounds={n-n_hi} ({100*(n-n_hi)/n:.1f}%)')
else:
    print('[$ARM] NO TRACE ROWS')
" | tee -a logs/ratesweep_oraclebudgetreserve_THRESHOLDS.txt

log "re-analyzing (phase-split): nsmono nsstatic512 ${ARM}"
OUT="logs/ratesweep_oraclebudgetreserve_ANALYSIS.txt"
SCHEDULE="$SCHED" ARMS="nsmono nsstatic512 ${ARM}" SLO_TBT_MS=500 \
  $PYTHON scripts/analyze_lengthgate.py > "$OUT" 2>&1 || true
cat "$OUT"

log "pooled (phase-agnostic) comparison"
$PYTHON -c "
import json
def pooled(path, label):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    gaps = []
    for r in recs:
        gaps.extend(r.get('tbt_ms') or [])
    gaps.sort()
    n = len(gaps)
    def pctl(p): return gaps[min(n-1, int(p/100*n))]
    print(f'{label}: n={n} mean={sum(gaps)/n:.1f} p99={pctl(99):.1f} p99.9={pctl(99.9):.1f} max={gaps[-1]:.1f}')
for label, path in [
    ('mono          ', 'logs/${DATE}-lgate-bnsmono-t1.jsonl'),
    ('static512     ', 'logs/${DATE}-lgate-bnsstatic512-t1.jsonl'),
    ('oraclebudgetreserve', '${FB}-t1.jsonl'),
]:
    pooled(path, label)
" | tee -a "$OUT"

log "done -- see $OUT"
touch logs/ratesweep_oraclebudgetreserve_ALLDONE
