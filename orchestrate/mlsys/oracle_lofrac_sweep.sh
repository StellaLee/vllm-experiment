#!/bin/bash
# oracle_lofrac_sweep.sh -- parameterized re-run of rate_sweep_oracle_ns.sh's three arms
# (mono, static-512, oracle-lpt), but with the "quiet" phase's whale fraction raised from
# 0% to a configurable LO_FRAC (default 0.05) instead of exactly zero.
#
# Motivation: the original oracle-lpt test (0%<->30% whale fraction) found oracle-lpt could
# not beat both static arms -- but 0% whales means there's nothing for tau to do in the LO
# phase at all, so mono trivially wins there and the "crossover" isn't really being tested.
# The whale-fraction x size grid (2026-07-24) found a genuine, non-trivial crossover: at 5%
# whale fraction specifically, chunking (tau=2048) makes p99 WORSE than mono (+90-106%),
# flipping to a clear win only at higher whale density. This script re-runs the oracle-lpt
# probe with LO_FRAC=0.05 (matching where the grid found the sign flip) instead of 0.0, to
# test whether a real, exploitable moving optimum exists on the tau lever once the quiet
# phase is genuinely non-trivial rather than whale-free.
#
# Required env: ARM (mono|static512|oracle), LO_FRAC, PORT, GPUS (e.g. "0,1"), TAG
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] [$TAG/$ARM] $*" >&2; }

: "${ARM:?set ARM=mono|static512|oracle}"
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

EXTRA_CLI=""
ENV_ARGS=""
ORACLE_TRACE=""
if [ "$ARM" = "mono" ]; then
  : # no threshold flag at all
elif [ "$ARM" = "static512" ]; then
  EXTRA_CLI="--long-prefill-token-threshold 512"
elif [ "$ARM" = "oracle" ]; then
  $PYTHON scripts/hotpatch_oracle_lpt.py || { log "PATCH FAILED"; touch logs/oraclelofrac_${TAG}_${ARM}_FAILED; exit 1; }
  ORACLE_TRACE="logs/${DATE}-oraclelofrac-${TAG}-trace.csv"
  rm -f "$ORACLE_TRACE"
  ENV_ARGS="ORACLE_LPT=1 ORACLE_LO_S=$LO_S ORACLE_HI_S=$HI_S ORACLE_LO_THRESH=0 ORACLE_HI_THRESH=512 ORACLE_TRACE=$ORACLE_TRACE"
else
  log "unknown ARM=$ARM"; exit 1
fi

ARMTAG="${TAG}${ARM}"
FB="logs/${DATE}-oraclelofrac-${ARMTAG}"
rm -f "${FB}-t1.jsonl"

log "starting server (lo_frac=$LO_FRAC hi_frac=$HI_FRAC gpus=$GPUS port=$PORT)"
env CUDA_VISIBLE_DEVICES=$GPUS PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $ENV_ARGS \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 $EXTRA_CLI > ${FB}-server.log 2>&1 &
SV=$!
UP=0
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" ${FB}-server.log && { UP=1; break; }
done
if [ "$UP" = 0 ]; then
  log "SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
  touch logs/oraclelofrac_${ARMTAG}_FAILED; exit 1
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
    thr = [int(r[2]) for r in rows]
    n_lo = sum(1 for t in thr if t == 0)
    n_hi = sum(1 for t in thr if t == 512)
    print(f'[$ARMTAG] n_rounds={n} lo_rounds={n_lo} ({100*n_lo/n:.1f}%) hi_rounds={n_hi} ({100*n_hi/n:.1f}%)')
"
fi

log "done -- ${FB}-t1.jsonl"
touch logs/oraclelofrac_${ARMTAG}_ALLDONE
