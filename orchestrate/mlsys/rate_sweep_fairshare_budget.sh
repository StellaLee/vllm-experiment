#!/bin/bash
# rate_sweep_fairshare_budget.sh -- test whether static-512's dominance under load is a workload
# artifact (narrow/high-mean whale distribution, artificially truncated max-tokens=256), and
# whether the budget-level fair-share formula (token_budget = budget/(1+running), no calibrated
# constant) closes the gap once it's applied to the right lever (aggregate round cost, not just
# one request's slice). Changes vs. every earlier sweep this session:
#   - whale distribution: WIDER and LOWER-mean (10000-50000 chars, uniform) vs. the original
#     44000-50000 uniform (narrow, ~14.5k-token mean) -- per-request user steer.
#   - --max-tokens 1024 (was 256, which truncated ~50% of natural completions -- see
#     logs/*-mtprobe-* from earlier this session) -- removes the artificial decode-length cap.
# 3 arms only (mono, static-512, fairshare-budget) since this is testing a specific hypothesis,
# not a full re-characterization -- lpt512/adaptivelpt2 already established to lose under load,
# no need to re-run them under a workload change to re-confirm that. Fresh baselines are required
# (not reused from earlier this session) because the workload itself changed.
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
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server.*port $PORT" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023.*port $PORT" && kill -9 $pid 2>/dev/null; done; sleep 6; }

replay(){ # $1=arm-label -> writes ${FB}-t1.jsonl, assumes server already up on $PORT
  local ARM=$1; local FB="logs/${DATE}-lgate-b${ARM}"
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens $MAXTOK \
    --phase-schedule "$SCHED" --duration $DUR \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX --whale-pareto-alpha 0 \
    --max-prompt-chars 50000 --pad-seed 1001 \
    --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
  log "  [$ARM] recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0)"
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
  wait_up "$FB" || { log "  [$ARM] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours; touch logs/ratesweep_fsb_FAILED; return 1; }
  replay "$ARM"
  kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
}

run_fairshare_budget(){ # $1=arm-label
  local ARM=$1
  local FB="logs/${DATE}-lgate-b${ARM}"
  rm -f "${FB}-t1.jsonl" "${FB}-chunktrace.csv"
  log "  [$ARM] starting server (FAIRSHARE_BUDGET)"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
    FAIRSHARE_BUDGET=1 FAIRSHARE_BUDGET_TRACE="${FB}-chunktrace.csv" \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  wait_up "$FB" || { log "  [$ARM] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours; touch logs/ratesweep_fsb_FAILED; return 1; }
  replay "$ARM"
  kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours

  $PYTHON -c "
import csv
rows = list(csv.reader(open('${FB}-chunktrace.csv')))
b = [int(r[2]) for r in rows if len(r) > 2]
if b:
    s = sorted(b); n = len(s)
    ceiling = sum(1 for x in b if x >= 16384)
    floor = sum(1 for x in b if x <= 130)
    print(f'[$ARM] n_steps={n}  budget: min={s[0]} p50={s[n//2]} p90={s[int(0.9*n)]} p99={s[int(0.99*n)]} max={s[-1]}')
    print(f'[$ARM] at_ceiling(>=16384)={ceiling} ({100*ceiling/n:.1f}%)  at_floor(<=130)={floor} ({100*floor/n:.1f}%)')
else:
    print('[$ARM] NO TRACE ROWS')
" | tee -a logs/ratesweep_fsb_THRESHOLDS.txt
}

$PYTHON scripts/hotpatch_fairshare_budget.py || { log "PATCH FAILED"; touch logs/ratesweep_fsb_FAILED; exit 1; }

for WRATE in 0.5 1.0 2.0; do
  TAG=$(echo $WRATE | tr -d '.')
  SCHED="6:0.0@60,${WRATE}:0.2@60"; DUR=360
  DATE=$(date +%Y-%m-%d)
  log "WRATE=$WRATE: whale=[$WHALE_MIN,$WHALE_MAX] uniform, max-tokens=$MAXTOK"

  run_static "16384fsb${TAG}" --max-num-batched-tokens 16384
  run_static "512fsb${TAG}"   --max-num-batched-tokens 512
  run_fairshare_budget "fsbudget${TAG}"

  log "analyzing WRATE=$WRATE"
  SCHEDULE="$SCHED" ARMS="16384fsb${TAG} 512fsb${TAG} fsbudget${TAG}" SLO_TBT_MS=500 \
    $PYTHON scripts/analyze_lengthgate.py > logs/ratesweep_fsb_${TAG}_ANALYSIS.txt 2>&1
  cat logs/ratesweep_fsb_${TAG}_ANALYSIS.txt
done

log "done -- see logs/ratesweep_fsb_{05,10,20}_ANALYSIS.txt and logs/ratesweep_fsb_THRESHOLDS.txt"
touch logs/ratesweep_fsb_ALLDONE
