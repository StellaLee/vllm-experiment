#!/bin/bash
# condition_sweep_burstgpt.sh -- same server-side ARM/hotpatch logic as condition_sweep.sh,
# but drives the client via vLLM's own native `vllm bench serve --dataset-name burstgpt`
# instead of src/replay_sharegpt.py. Real GPT-4 request/response token-length distribution
# (215127 rows, whale frac ~0.14% naturally, no synthetic whale injection), synthetic Poisson
# arrival process via --request-rate/--burstiness (the CSV's own timestamps are NOT replayed
# by this dataset class -- see scripts/make_burstgpt_trace.py for a real-arrival-timing
# alternative built earlier in this line of work, not used here per user's steer toward the
# native tool).
#
# Required env: ARM (mono|static512|static-reserve|whale-aware-v2|backlog-aware), PORT, GPUS,
#               TAG. Optional: NUMPROMPTS (default 6000), RATE (req/s, default 5),
#               BUDGET, BACKLOG_THRESH.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL="${MODEL:-/data/pli/models/Qwen2.5-Coder-14B-Instruct}"
BGCSV=/root/pli/BurstGPT/data/BurstGPT_1.csv
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] [$TAG/$ARM] $*" >&2; }

: "${ARM:?set ARM}"
: "${PORT:?set PORT}"
: "${GPUS:?set GPUS, e.g. 0,1}"
: "${TAG:?set TAG}"
NUMPROMPTS="${NUMPROMPTS:-6000}"
RATE="${RATE:-5}"
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
  ENV_ARGS=""
  MAXBATCHED=512
elif [ "$ARM" = "static-reserve" ]; then
  $PYTHON scripts/hotpatch_static_reserve.py || { log "PATCH FAILED"; touch logs/condsweepbg_${ARMTAG}_FAILED; exit 1; }
  ENV_ARGS="STATIC_RESERVE=1 STATIC_RESERVE_BUDGET=${BUDGET}"
elif [ "$ARM" = "whale-aware-v2" ]; then
  $PYTHON scripts/hotpatch_whale_aware_budget_v2.py || { log "PATCH FAILED"; touch logs/condsweepbg_${ARMTAG}_FAILED; exit 1; }
  ENV_ARGS="WHALE_AWARE_BUDGET=1 WHALE_AWARE_WHALE_TOK=4000 WHALE_AWARE_LO_BUDGET=16384 WHALE_AWARE_HI_BUDGET=512"
elif [ "$ARM" = "backlog-aware" ]; then
  $PYTHON scripts/hotpatch_backlog_aware_budget.py || { log "PATCH FAILED"; touch logs/condsweepbg_${ARMTAG}_FAILED; exit 1; }
  BACKLOG_THRESH="${BACKLOG_THRESH:-4000}"
  ENV_ARGS="BACKLOG_AWARE_BUDGET=1 BACKLOG_AWARE_WHALE_TOK=4000 BACKLOG_AWARE_LO_BUDGET=16384 BACKLOG_AWARE_HI_BUDGET=512 BACKLOG_AWARE_BACKLOG_THRESH=${BACKLOG_THRESH}"
else
  log "unknown ARM=$ARM"; exit 1
fi

FB="logs/${DATE}-condsweepbg-${ARMTAG}"

log "starting server (gpus=$GPUS port=$PORT maxbatched=$MAXBATCHED)"
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
  touch logs/condsweepbg_${ARMTAG}_FAILED; exit 1
fi

log "starting vllm bench serve (num_prompts=$NUMPROMPTS rate=$RATE)"
$PYTHON -m vllm.entrypoints.cli.main bench serve \
  --dataset-name burstgpt --dataset-path "$BGCSV" \
  --model "$MODEL" --host localhost --port $PORT \
  --num-prompts $NUMPROMPTS --request-rate $RATE --burstiness 1.0 --seed 1001 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99,99.9 \
  --save-result --save-detailed --result-dir logs --result-filename ${ARMTAG}-burstgpt.json \
  > ${FB}.client.log 2>&1 || true
log "recs written: $(grep -c completed ${FB}.client.log 2>/dev/null || echo '?') preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
cp ${SCHED_PY}.pristine ${SCHED_PY}

log "done -- logs/${ARMTAG}-burstgpt.json"
touch logs/condsweepbg_${ARMTAG}_ALLDONE
