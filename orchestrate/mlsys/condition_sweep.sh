#!/bin/bash
# condition_sweep.sh -- parameterized version of rate_sweep_static_reserve.sh /
# rate_sweep_whale_aware_budget_v2.sh, for running the static-reserve vs. whale-aware-v2
# head-to-head under conditions other than the headline rate=1.0/whale-frac=0.3 run.
#
# Motivation: the headline-condition ablation (2026-08-28) found static-reserve (decode-
# reserve + fixed C=512) statistically matches whale-aware-v2 (dynamic C in {512,16384}) on
# every tail-latency and throughput metric -- the opposite of what motivated building the
# dynamic gating. The queuing-theory prediction (T(t)~=b+alpha*A(t)) says the gap SHOULD open
# up once the round is genuinely throughput-constrained even without a whale present, since
# only then does relaxing to C=16384 during no-whale windows actually buy anything. rate=1.0
# may simply be under-loaded. This script lets us sweep RATE (and HI_FRAC) to test that.
#
# Required env: ARM (mono|static512|static-reserve|whale-aware-v2|backlog-aware), RATE, HI_FRAC, PORT, GPUS (e.g. "0,1"),
#               TAG (short label for filenames, e.g. "rate25")
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL="${MODEL:-/data/pli/models/Qwen2.5-Coder-14B-Instruct}"
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] [$TAG/$ARM] $*" >&2; }

: "${ARM:?set ARM=static-reserve or whale-aware-v2}"
: "${RATE:?set RATE, e.g. 2.5}"
: "${HI_FRAC:?set HI_FRAC, e.g. 0.3}"
: "${PORT:?set PORT}"
: "${GPUS:?set GPUS, e.g. 0,1}"
: "${TAG:?set TAG, short label for filenames}"

WHALE_MIN=10000
WHALE_MAX=50000
MAXTOK=1024
DUR=720
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
SCHED="${RATE}:0.0@120,${RATE}:${HI_FRAC}@120"
DATE=$(date +%Y-%m-%d)
SCHED_PY=/root/pli/venv-vllm023/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py

kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server.*port $PORT" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023.*port $PORT" && kill -9 $pid 2>/dev/null; done; sleep 6; }

cp ${SCHED_PY}.pristine ${SCHED_PY}

BUDGET="${BUDGET:-512}"

ARMTAG="${TAG}${ARM//-/}"
if [ "$ARM" = "static-reserve" ] && [ "$BUDGET" != "512" ]; then ARMTAG="${ARMTAG}b${BUDGET}"; fi

MAXBATCHED=16384
if [ "$ARM" = "mono" ]; then
  ENV_ARGS=""
elif [ "$ARM" = "static512" ]; then
  # True vanilla static BUDGET chunking: native vLLM aggregate round token budget (C) fixed
  # at 512 via --max-num-batched-tokens, no hotpatch, no decode-reserve reordering, no whale
  # detection. NOT the per-request long-prefill-token-threshold (that's tau, a distinct
  # mechanism -- see the necessity-argument work distinguishing tau from C). This is the
  # baseline static-reserve was built to ablate against (static-reserve = this + decode-
  # reserve reorder, same aggregate budget C=512).
  ENV_ARGS=""
  MAXBATCHED=512
elif [ "$ARM" = "static-reserve" ]; then
  $PYTHON scripts/hotpatch_static_reserve.py || { log "PATCH FAILED"; touch logs/condsweep_${ARMTAG}_FAILED; exit 1; }
  ENV_ARGS="STATIC_RESERVE=1 STATIC_RESERVE_BUDGET=${BUDGET}"
elif [ "$ARM" = "whale-aware-v2" ]; then
  $PYTHON scripts/hotpatch_whale_aware_budget_v2.py || { log "PATCH FAILED"; touch logs/condsweep_${ARMTAG}_FAILED; exit 1; }
  ENV_ARGS="WHALE_AWARE_BUDGET=1 WHALE_AWARE_WHALE_TOK=4000 WHALE_AWARE_LO_BUDGET=16384 WHALE_AWARE_HI_BUDGET=512"
elif [ "$ARM" = "backlog-aware" ]; then
  $PYTHON scripts/hotpatch_backlog_aware_budget.py || { log "PATCH FAILED"; touch logs/condsweep_${ARMTAG}_FAILED; exit 1; }
  BACKLOG_THRESH="${BACKLOG_THRESH:-4000}"
  ENV_ARGS="BACKLOG_AWARE_BUDGET=1 BACKLOG_AWARE_WHALE_TOK=4000 BACKLOG_AWARE_LO_BUDGET=16384 BACKLOG_AWARE_HI_BUDGET=512 BACKLOG_AWARE_BACKLOG_THRESH=${BACKLOG_THRESH}"
  if [ -n "${WITH_TRACE:-}" ]; then
    BACKLOG_TRACE_FILE="logs/$(date +%Y-%m-%d)-condsweep-${TAG}${ARM//-/}-backlogtrace.csv"
    rm -f "$BACKLOG_TRACE_FILE"
    ENV_ARGS="${ENV_ARGS} BACKLOG_AWARE_TRACE=${BACKLOG_TRACE_FILE}"
  fi
else
  log "unknown ARM=$ARM"; exit 1
fi

FB="logs/${DATE}-condsweep-${ARMTAG}"
rm -f "${FB}-t1.jsonl"

log "starting server (rate=$RATE hi_frac=$HI_FRAC gpus=$GPUS port=$PORT)"
env CUDA_VISIBLE_DEVICES=$GPUS PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $ENV_ARGS \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 128 --max-num-batched-tokens $MAXBATCHED --max-model-len 16384 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
SV=$!
UP=0
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" ${FB}-server.log && { UP=1; break; }
done
if [ "$UP" = 0 ]; then
  log "SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
  touch logs/condsweep_${ARMTAG}_FAILED; exit 1
fi

$PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
  --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens $MAXTOK \
  --phase-schedule "$SCHED" --duration $DUR \
  --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
  --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX --whale-pareto-alpha 0 \
  --max-prompt-chars 50000 --pad-seed 1001 --request-timeout $REQUEST_TIMEOUT \
  --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
log "recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
cp ${SCHED_PY}.pristine ${SCHED_PY}

if [ -n "${BACKLOG_TRACE_FILE:-}" ] && [ -f "$BACKLOG_TRACE_FILE" ]; then
  $PYTHON -c "
import csv
rows = list(csv.reader(open('$BACKLOG_TRACE_FILE')))
if rows:
    n = len(rows)
    b = [int(r[3]) for r in rows]
    n_hi = sum(1 for x in b if x == 512)
    pending = [int(r[2]) for r in rows]
    print(f'[$ARMTAG] n_rounds={n} tight_rounds={n_hi} ({100*n_hi/n:.1f}%) relaxed_rounds={n-n_hi} ({100*(n-n_hi)/n:.1f}%)')
    print(f'[$ARMTAG] pending_backlog: min={min(pending)} p50={sorted(pending)[n//2]} p95={sorted(pending)[int(0.95*n)]} max={max(pending)}')
"
fi

log "done -- ${FB}-t1.jsonl"
touch logs/condsweep_${ARMTAG}_ALLDONE
