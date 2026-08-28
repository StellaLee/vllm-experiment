#!/bin/bash
# rate_sweep_lpt.sh -- does the static-512 vs per-request-cap tail-latency gap change with
# Phase-W load? Runs mono (16384, no protection baseline), static-512, mono+long-prefill-
# token-threshold=512 (always-on per-request cap), and the adaptive controller (ADAPTIVE_LPT)
# at a swept Phase-W arrival rate (Phase-S rate and whale-frac held at the original
# calibrated values: rate_S=6, whale_frac=0.2, duration=360). Same workload/model/dataset as
# rerun_lpt_arm.sh / run_adaptive_lpt_arm2.sh; only the Phase-W rate and output tags differ so
# results don't collide with the existing baseline (WRATE=1.0) logs.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }

WRATE=${WRATE:?"set WRATE (Phase-W req/s), e.g. WRATE=0.5"}
TAG=${TAG:?"set TAG (short label for filenames), e.g. TAG=w05"}
SCHED="6:0.0@60,${WRATE}:0.2@60"; DUR=360
PORT=8050
DATE=$(date +%Y-%m-%d)
GATE=4096; PROTECT=512; OFF=0

log "rate sweep point: SCHED='$SCHED' DUR=${DUR}s TAG=$TAG"

$PYTHON scripts/hotpatch_adaptive_lpt.py || { log "PATCH FAILED"; touch logs/ratesweep_FAILED; exit 1; }

replay(){ # $1=arm-label -> writes ${FB}-t1.jsonl, assumes server already up on $PORT
  local ARM=$1; local FB="logs/${DATE}-lgate-b${ARM}"
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens 256 \
    --phase-schedule "$SCHED" --duration $DUR \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-min-chars 44000 --whale-max-chars 50000 \
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
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-model-len 16384 "$@" \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  wait_up "$FB" || { log "  [$ARM] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours; touch logs/ratesweep_FAILED; exit 1; }
  replay "$ARM"
  kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
}

run_adaptive(){ # $1=arm-label
  local ARM=$1
  local FB="logs/${DATE}-lgate-b${ARM}"
  rm -f "${FB}-t1.jsonl" "${FB}-chunktrace.csv"
  log "  [$ARM] starting server (ADAPTIVE_LPT gate=$GATE protect=$PROTECT)"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
    ADAPTIVE_LPT=1 ADAPTIVE_LPT_GATE=$GATE ADAPTIVE_LPT_PROTECT=$PROTECT ADAPTIVE_LPT_OFF=$OFF \
    ADAPTIVE_LPT_TRACE="${FB}-chunktrace.csv" \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  wait_up "$FB" || { log "  [$ARM] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours; touch logs/ratesweep_FAILED; exit 1; }
  replay "$ARM"
  kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
}

run_static "16384${TAG}"      --max-num-batched-tokens 16384
run_static "512${TAG}"        --max-num-batched-tokens 512
run_static "lpt512${TAG}"     --max-num-batched-tokens 16384 --long-prefill-token-threshold 512
run_adaptive "adaptivelpt${TAG}"

log "analyzing: mono vs static-512 vs 16384lpt512 vs adaptivelpt @ WRATE=$WRATE"
SCHEDULE="$SCHED" ARMS="16384${TAG} 512${TAG} lpt512${TAG} adaptivelpt${TAG}" SLO_TBT_MS=500 \
  $PYTHON scripts/analyze_lengthgate.py > logs/ratesweep_${TAG}_ANALYSIS.txt 2>&1
cat logs/ratesweep_${TAG}_ANALYSIS.txt
echo "[$(STAMP)] DONE ${TAG}" >> logs/ratesweep_${TAG}_ANALYSIS.txt
log "done -> logs/ratesweep_${TAG}_ANALYSIS.txt"
