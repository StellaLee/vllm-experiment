#!/bin/bash
# rate_sweep_oracle_ns.sh -- non-stationary headroom probe (2026-08-28).
# Tests whether adaptive control has ANY theoretical ceiling above the best single
# static choice, under a load where the optimal threshold genuinely moves over time
# (whale_frac alternates 0% <-> 30% in sustained 120s blocks, arrival rate held
# constant so only whale-pressure varies). Three arms on the IDENTICAL schedule:
#   mono_ns    : long_prefill_token_threshold off the whole time (matches LO's optimum)
#   static512_ns: long_prefill_token_threshold=512 the whole time (matches HI's optimum)
#   oracle_ns  : switches 0 <-> 512 on the known wall-clock phase boundary
# If oracle doesn't beat both static arms pooled AND per-regime, non-stationarity alone
# doesn't create real headroom and there's no point building a real reactive controller.
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
RATE=1.0
LO_S=120
HI_S=120
HI_FRAC=0.3
DUR=720   # 3 full LO+HI cycles
SCHED="${RATE}:0.0@${LO_S},${RATE}:${HI_FRAC}@${HI_S}"
DATE=$(date +%Y-%m-%d)

kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server.*port $PORT" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023.*port $PORT" && kill -9 $pid 2>/dev/null; done; sleep 6; }

revert_pristine(){
  SCHED_PY=/root/pli/venv-vllm023/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py
  cp "${SCHED_PY}.pristine" "$SCHED_PY"
}

run_replay(){ # $1=ARM  $2=extra CLI flags for api_server (single string, may be empty)  $3...=env VAR=VAL pairs
  local ARM=$1; local EXTRA_CLI=$2; shift 2
  local FB="logs/${DATE}-lgate-b${ARM}"
  rm -f "${FB}-t1.jsonl" "${FB}-oracletrace.csv"
  log "[$ARM] starting server (env: $* | cli: $EXTRA_CLI)"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 "$@" \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 $EXTRA_CLI > ${FB}-server.log 2>&1 &
  local SV=$!
  local UP=0
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" ${FB}-server.log && { UP=1; break; }
  done
  if [ "$UP" = 0 ]; then
    log "[$ARM] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
    touch logs/ratesweep_oraclens_FAILED; return 1
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
  revert_pristine
}

revert_pristine

# mono: no threshold flag at all
run_replay "nsmono" "" || { log "nsmono FAILED"; touch logs/ratesweep_oraclens_FAILED; exit 1; }

# static-512: native flag, no hotpatch
run_replay "nsstatic512" "--long-prefill-token-threshold 512" || { log "nsstatic512 FAILED"; touch logs/ratesweep_oraclens_FAILED; exit 1; }

# oracle: hotpatch + env-gated switch
$PYTHON scripts/hotpatch_oracle_lpt.py || { log "PATCH FAILED"; touch logs/ratesweep_oraclens_FAILED; exit 1; }
FB_ORACLE="logs/${DATE}-lgate-bnsoracle-oracletrace.csv"
run_replay "nsoracle" "" ORACLE_LPT=1 ORACLE_LO_S=$LO_S ORACLE_HI_S=$HI_S \
  ORACLE_LO_THRESH=0 ORACLE_HI_THRESH=512 ORACLE_TRACE="$FB_ORACLE" || { log "nsoracle FAILED"; touch logs/ratesweep_oraclens_FAILED; exit 1; }

$PYTHON -c "
import csv
rows = list(csv.reader(open('${FB_ORACLE}')))
if rows:
    n = len(rows)
    thr = [int(r[2]) for r in rows]
    n_lo = sum(1 for t in thr if t == 0)
    n_hi = sum(1 for t in thr if t == 512)
    print(f'[oracle] n_rounds={n} lo_rounds={n_lo} ({100*n_lo/n:.1f}%) hi_rounds={n_hi} ({100*n_hi/n:.1f}%)')
else:
    print('[oracle] NO TRACE ROWS')
" | tee -a logs/ratesweep_oraclens_THRESHOLDS.txt

log "re-analyzing (S=LO-regime, W=HI-regime, via existing phase-split logic): nsmono nsstatic512 nsoracle"
OUT="logs/ratesweep_oraclens_ANALYSIS.txt"
SCHEDULE="$SCHED" ARMS="nsmono nsstatic512 nsoracle" SLO_TBT_MS=500 \
  $PYTHON scripts/analyze_lengthgate.py > "$OUT" 2>&1 || true
cat "$OUT"

log "done -- see $OUT and logs/ratesweep_oraclens_THRESHOLDS.txt"
touch logs/ratesweep_oraclens_ALLDONE
