#!/bin/bash
# orchestrate_cs2_threshold_sweep.sh -- tests whether long_prefill_token_threshold (a genuine
# per-request cap, independent of the aggregate step budget) shows a cleaner Cs^2-correlated
# TTFT signal than swapping the aggregate BUDGET did (orchestrate_cs2_openloop_sweep.sh).
#
# Rationale: capping the step BUDGET (16384->2048) still lets ONE request monopolize whatever
# budget remains each round (running-loop drains self.running in list order, token_budget -=
# num_new_tokens with no per-request ceiling) -- that protects already-running DECODE requests
# (their tiny 1-token need always fits in whatever's left), but does NOT let a second competing
# admission share the SAME step, since the first request can still eat the whole remaining
# budget. long_prefill_token_threshold caps each individual request's own slice regardless of
# the aggregate budget -- with budget FIXED LARGE (16384) and threshold=512, several requests
# each capped at 512 tokens can ALL be scheduled in the same step (16384 has room for ~32 such
# slices), which is genuine simultaneous multi-way sharing, not just smaller turns for one job.
# This is the mechanism-matched test of Eq. genuine / classical PS that budget-swapping wasn't.
#
# Same validated open-loop setup as orchestrate_cs2_openloop_sweep.sh (rate=0.05, pad_mean=20000/
# pad_max=46000 chars per the measured 3.235 chars/token ratio, no whale mode) -- only the
# server-side axis changes: budget fixed, long-prefill-token-threshold swept.
# Server boots ONCE per threshold and serves every cv2 point (pad-cv2 is client-side).
# Tunables: THRESHOLDS CV2_LIST RATE NCONV MAXTOK BUDGET PAD_MEAN PAD_MIN PAD_MAX MAX_PROMPT_CHARS.
# Markers: logs/cs2th_r${RATE}_n${NCONV}_ALLDONE|FAILED.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs
DATE=$(date +%Y-%m-%d)
BUDGET=${BUDGET:-16384}                    # FIXED, large -- the aggregate is never the bottleneck
THRESHOLDS=${THRESHOLDS:-"0 512"}           # first = mono-equivalent baseline (0 = off)
CV2_LIST=${CV2_LIST:-"0.1 1 4 8"}
RATE=${RATE:-0.05}
NCONV=${NCONV:-30}
MAXTOK=${MAXTOK:-256}
PAD_MEAN=${PAD_MEAN:-20000}
PAD_MIN=${PAD_MIN:-200}
PAD_MAX=${PAD_MAX:-46000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-46000}
PAD_SEED=${PAD_SEED:-2001}                  # same seed as the budget sweep -> paired comparison
rm -f "logs/cs2th_r${RATE}_n${NCONV}_ALLDONE" "logs/cs2th_r${RATE}_n${NCONV}_FAILED"
log "cs2 threshold sweep: budget(fixed)=$BUDGET thresholds=[$THRESHOLDS] cv2=[$CV2_LIST] rate=$RATE nconv=$NCONV mt=$MAXTOK pad_mean=$PAD_MEAN[$PAD_MIN,$PAD_MAX] seed=$PAD_SEED"

run_threshold(){ # threshold
  local thr=$1 arm="t$1" port=8050
  log "  [$arm] server on GPUs 0,1 port=$port budget=$BUDGET threshold=$thr"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $port --max-num-seqs 48 --max-num-batched-tokens $BUDGET \
      --long-prefill-token-threshold $thr \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-cs2th-${arm}-server.log 2>&1 &
  local sv=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-cs2th-${arm}-server.log && break
    [ "$i" = 120 ] && { log "  [$arm] SERVER TIMEOUT"; kill "$sv" 2>/dev/null; return 1; }
  done
  local rc=0
  for cv2 in $CV2_LIST; do
    local ctag=$($PYTHON -c "print(f'{round(float(\"$cv2\")*100):04d}')")
    local out="logs/${DATE}-cs2th-${arm}-c${ctag}-r${RATE}-n${NCONV}-t1.jsonl"
    $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
      --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
      --max-tokens $MAXTOK --rate $RATE \
      --pad-mean-chars $PAD_MEAN --pad-cv2 $cv2 --pad-min $PAD_MIN --pad-max $PAD_MAX \
      --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed $PAD_SEED \
      --output "$out" > "${out%.jsonl}.client.log" 2>&1 || rc=1
    log "  [$arm cv2=$cv2] done recs=$(grep -c . "$out" 2>/dev/null || echo 0)"
  done
  kill "$sv" 2>/dev/null; sleep 8; kill -9 "$sv" 2>/dev/null
  kill_ours
  return $rc
}

RC=0
for t in $THRESHOLDS; do run_threshold "$t" || RC=1; done
log "analyzing"
OUT_ANALYSIS="logs/cs2th_r${RATE}_n${NCONV}_ANALYSIS.txt"
THRESHOLDS="$THRESHOLDS" CV2_LIST="$CV2_LIST" RATE="$RATE" NCONV="$NCONV" \
  $PYTHON scripts/analyze_cs2_threshold_sweep.py > "$OUT_ANALYSIS" 2>&1
echo "[$(STAMP)] DONE (rc=$RC)" >> "$OUT_ANALYSIS"
if [ "$RC" = 0 ]; then touch "logs/cs2th_r${RATE}_n${NCONV}_ALLDONE"; else touch "logs/cs2th_r${RATE}_n${NCONV}_FAILED"; fi
log "done -> $OUT_ANALYSIS"
