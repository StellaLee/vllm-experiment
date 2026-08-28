#!/bin/bash
# rate_sweep_lpt_v2.sh -- same 4-arm comparison (mono / static-512 / always-on 16384lpt512 /
# adaptive controller) as rate_sweep_lpt.sh, but parameterized on GPU pair + port (so it can
# run alongside the original on a free GPU pair) and on the whale-size distribution (floor +
# Pareto alpha), to test whether widening the whale-size spread (currently a concentrated
# uniform 44-50k chars) changes the static-512-vs-per-request-cap story.

# Preliminary result: raising the cap from 256→512 cut truncation from ~50% to 24.7%; →1024 got it to 9.3%.


set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
GPUS=${GPUS:?"set GPUS, e.g. GPUS=2,3"}
PORT=${PORT:?"set PORT, e.g. PORT=8060"}
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server.*port $PORT" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023.*port $PORT" && kill -9 $pid 2>/dev/null; done; sleep 6; }

WRATE=${WRATE:?"set WRATE (Phase-W req/s), e.g. WRATE=1.0"}
TAG=${TAG:?"set TAG (short label for filenames), e.g. TAG=wdist"}
WHALE_MIN=${WHALE_MIN:-44000}
WHALE_MAX=${WHALE_MAX:-50000}
WHALE_ALPHA=${WHALE_ALPHA:-0}
SCHED="6:0.0@60,${WRATE}:0.2@60"; DUR=360
DATE=$(date +%Y-%m-%d)
GATE=4096; PROTECT=512; OFF=0

log "rate sweep v2: SCHED='$SCHED' DUR=${DUR}s TAG=$TAG GPUS=$GPUS PORT=$PORT whale=[$WHALE_MIN,$WHALE_MAX] alpha=$WHALE_ALPHA"

$PYTHON scripts/hotpatch_adaptive_lpt.py || { log "PATCH FAILED"; touch logs/ratesweep_v2_FAILED; exit 1; }

replay(){ # $1=arm-label -> writes ${FB}-t1.jsonl, assumes server already up on $PORT
  local ARM=$1; local FB="logs/${DATE}-lgate-b${ARM}"
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens 256 \
    --phase-schedule "$SCHED" --duration $DUR \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX --whale-pareto-alpha $WHALE_ALPHA \
    --max-prompt-chars 50000 --pad-seed 1001 \
    --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
  log "  [$ARM] recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
}

wait_up(){ local FB=$1
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" ${FB}-server.log && return 0
    [ "$i" = 120 ] && return 1
  done
}

run_static(){ # $1=arm-label $2...=extra server CLI flags
  local ARM=$1; shift
  local FB="logs/${DATE}-lgate-b${ARM}"
  rm -f "${FB}-t1.jsonl"
  log "  [$ARM] starting server ($*)"
  env CUDA_VISIBLE_DEVICES=$GPUS PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-model-len 16384 "$@" \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  wait_up "$FB" || { log "  [$ARM] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours; touch logs/ratesweep_v2_FAILED; exit 1; }
  replay "$ARM"
  kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
}

run_adaptive(){ # $1=arm-label
  local ARM=$1
  local FB="logs/${DATE}-lgate-b${ARM}"
  rm -f "${FB}-t1.jsonl" "${FB}-chunktrace.csv"
  log "  [$ARM] starting server (ADAPTIVE_LPT gate=$GATE protect=$PROTECT)"
  env CUDA_VISIBLE_DEVICES=$GPUS PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
    ADAPTIVE_LPT=1 ADAPTIVE_LPT_GATE=$GATE ADAPTIVE_LPT_PROTECT=$PROTECT ADAPTIVE_LPT_OFF=$OFF \
    ADAPTIVE_LPT_TRACE="${FB}-chunktrace.csv" \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  wait_up "$FB" || { log "  [$ARM] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours; touch logs/ratesweep_v2_FAILED; exit 1; }
  replay "$ARM"
  kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
}

run_static "16384${TAG}"      --max-num-batched-tokens 16384
run_static "512${TAG}"        --max-num-batched-tokens 512
run_static "lpt512${TAG}"     --max-num-batched-tokens 16384 --long-prefill-token-threshold 512
run_adaptive "adaptivelpt${TAG}"

log "analyzing: mono vs static-512 vs 16384lpt512 vs adaptivelpt @ WRATE=$WRATE whale=[$WHALE_MIN,$WHALE_MAX]/alpha=$WHALE_ALPHA"
SCHEDULE="$SCHED" ARMS="16384${TAG} 512${TAG} lpt512${TAG} adaptivelpt${TAG}" SLO_TBT_MS=500 \
  $PYTHON scripts/analyze_lengthgate.py > logs/ratesweep_${TAG}_ANALYSIS.txt 2>&1
cat logs/ratesweep_${TAG}_ANALYSIS.txt
echo "[$(STAMP)] DONE ${TAG}" >> logs/ratesweep_${TAG}_ANALYSIS.txt
log "done -> logs/ratesweep_${TAG}_ANALYSIS.txt"
