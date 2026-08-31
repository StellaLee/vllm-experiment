#!/bin/bash
# budget_lofrac_sweep.sh -- tests whether the aggregate budget C (with decode-reorder always
# on) has a genuine moving optimum under non-stationary load, at a quiet-phase whale fraction
# raised from 0% to a configurable LO_FRAC (default 0.05) -- the budget-lever analogue of
# oracle_lofrac_sweep.sh (which tests the SAME question on tau, not C).
#
# Three arms, decode-reorder on throughout:
#   reorder-only : C=16384 always (uncapped; the arm that tracked mono's bad tail at LO_FRAC=0)
#   static-reserve: C=512 always (our fixed-protective-budget arm)
#   oracle-budget-reserve: C oracle-switched 16384<->512 on the known phase boundary
# If oracle-budget-reserve doesn't beat BOTH fixed-C arms here, the dynamic gating still
# has no headroom even once the quiet phase is genuinely non-trivial (5% whales, not 0%).
#
# Required env: ARM (reorderonly|staticreserve|oraclebudgetreserve), LO_FRAC, PORT,
#               GPUS (e.g. "0,1"), TAG
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] [$TAG/$ARM] $*" >&2; }

: "${ARM:?set ARM=reorderonly|staticreserve|oraclebudgetreserve}"
: "${LO_FRAC:?set LO_FRAC, e.g. 0.05}"
: "${PORT:?set PORT}"
: "${GPUS:?set GPUS, e.g. 0,1}"
: "${TAG:?set TAG, short label for filenames}"

WHALE_MIN=10000
WHALE_MAX=50000
MAXTOK=1024
RATE=1.0
LO_S=120
HI_S=120
HI_FRAC=0.3
DUR=720
SCHED="${RATE}:${LO_FRAC}@${LO_S},${RATE}:${HI_FRAC}@${HI_S}"
DATE=$(date +%Y-%m-%d)
SCHED_PY=/root/pli/venv-vllm023/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py

kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server.*port $PORT" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023.*port $PORT" && kill -9 $pid 2>/dev/null; done; sleep 6; }

cp ${SCHED_PY}.pristine ${SCHED_PY}

ORACLE_TRACE=""
if [ "$ARM" = "reorderonly" ]; then
  $PYTHON scripts/hotpatch_static_reserve.py || { log "PATCH FAILED"; touch logs/budgetlofrac_${TAG}${ARM}_FAILED; exit 1; }
  ENV_ARGS="STATIC_RESERVE=1 STATIC_RESERVE_BUDGET=16384"
elif [ "$ARM" = "staticreserve" ]; then
  $PYTHON scripts/hotpatch_static_reserve.py || { log "PATCH FAILED"; touch logs/budgetlofrac_${TAG}${ARM}_FAILED; exit 1; }
  ENV_ARGS="STATIC_RESERVE=1 STATIC_RESERVE_BUDGET=512"
elif [ "$ARM" = "oraclebudgetreserve" ]; then
  $PYTHON scripts/hotpatch_oracle_budget_reserve.py || { log "PATCH FAILED"; touch logs/budgetlofrac_${TAG}${ARM}_FAILED; exit 1; }
  ORACLE_TRACE="logs/${DATE}-budgetlofrac-${TAG}-trace.csv"
  rm -f "$ORACLE_TRACE"
  ENV_ARGS="ORACLE_BUDGET_RESERVE=1 ORACLE_LO_S=$LO_S ORACLE_HI_S=$HI_S ORACLE_LO_BUDGET=16384 ORACLE_HI_BUDGET=512 ORACLE_TRACE=$ORACLE_TRACE"
else
  log "unknown ARM=$ARM"; exit 1
fi

ARMTAG="${TAG}${ARM}"
FB="logs/${DATE}-budgetlofrac-${ARMTAG}"
rm -f "${FB}-t1.jsonl"

log "starting server (lo_frac=$LO_FRAC hi_frac=$HI_FRAC gpus=$GPUS port=$PORT)"
env CUDA_VISIBLE_DEVICES=$GPUS PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $ENV_ARGS \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
SV=$!
UP=0
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" ${FB}-server.log && { UP=1; break; }
done
if [ "$UP" = 0 ]; then
  log "SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
  touch logs/budgetlofrac_${ARMTAG}_FAILED; exit 1
fi

$PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
  --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens $MAXTOK \
  --phase-schedule "$SCHED" --duration $DUR \
  --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
  --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX --whale-pareto-alpha 0 \
  --max-prompt-chars 50000 --pad-seed 1001 \
  --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
log "recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
cp ${SCHED_PY}.pristine ${SCHED_PY}

if [ -n "$ORACLE_TRACE" ] && [ -f "$ORACLE_TRACE" ]; then
  $PYTHON -c "
import csv
rows = list(csv.reader(open('$ORACLE_TRACE')))
if rows:
    n = len(rows)
    b = [int(r[2]) for r in rows]
    n_hi = sum(1 for x in b if x == 512)
    print(f'[$ARMTAG] n_rounds={n} hi_rounds={n_hi} ({100*n_hi/n:.1f}%) lo_rounds={n-n_hi} ({100*(n-n_hi)/n:.1f}%)')
"
fi

log "done -- ${FB}-t1.jsonl"
touch logs/budgetlofrac_${ARMTAG}_ALLDONE
