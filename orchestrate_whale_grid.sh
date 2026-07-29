#!/bin/bash
# orchestrate_whale_grid.sh -- MLSys Phase 2, sub-part 3: whale-fraction x whale-size grid,
# mapping the genuine term's frontier (flagged as the natural next step in
# longprompt-tbt-win's own writeup, 2026-07-21, never executed until now).
# Uses the BUDGET mechanism (mono=16384 vs chunk=2048), matching §5.6's own lever, at
# CONC=20 (established safe point). Grid: whale_frac in {5%,15%,30%} x whale_size in
# {8k tok, 14k tok} = 6 combos x 2 budgets = 12 client runs. Server boots once per budget,
# serves all 6 (frac,size) combos.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/whalegrid_ALLDONE logs/whalegrid_FAILED
DATE=$(date +%Y-%m-%d)
BUDGETS=${BUDGETS:-"16384 2048"}
WHALE_FRACS=${WHALE_FRACS:-"0.05 0.15 0.30"}
CONC=${CONC:-20}
MAX_SEQS=${MAX_SEQS:-48}
MAXTOK=${MAXTOK:-256}
NCONV=${NCONV:-200}
PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
PAD_SEED=${PAD_SEED:-1001}
# Two size bands: "8k" ~ 8000 tok (25880 chars), "14k" ~ 14000 tok (45290 chars, matches the
# already-validated-safe 44-50k range used throughout this session).
SIZE_LABELS="8k 14k"
declare -A SIZE_MIN=( [8k]=24000 [14k]=44000 )
declare -A SIZE_MAX=( [8k]=28000 [14k]=50000 )
declare -A SIZE_MAXPC=( [8k]=30000 [14k]=50000 )

log "whale-size grid: budgets=[$BUDGETS] whale_fracs=[$WHALE_FRACS] sizes=[$SIZE_LABELS] conc=$CONC nconv=$NCONV"

run_budget(){ # budget
  local budget=$1 arm="b$1" port=8050
  log "  [$arm] server on GPUs 0,1 port=$port budget=$budget"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $port --max-num-seqs $MAX_SEQS --max-num-batched-tokens $budget \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-whalegrid-${arm}-server.log 2>&1 &
  local sv=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-whalegrid-${arm}-server.log && break
    [ "$i" = 120 ] && { log "  [$arm] SERVER TIMEOUT"; kill "$sv" 2>/dev/null; return 1; }
  done
  local rc=0
  for size in $SIZE_LABELS; do
    for wf in $WHALE_FRACS; do
      local wtag=$($PYTHON -c "print(f'{round(float(\"$wf\")*100):02d}')")
      local out="logs/${DATE}-whalegrid-${arm}-s${size}-w${wtag}-t1.jsonl"
      $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
        --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
        --max-tokens $MAXTOK --concurrency $CONC \
        --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
        --whale-frac $wf --whale-min-chars ${SIZE_MIN[$size]} --whale-max-chars ${SIZE_MAX[$size]} \
        --max-prompt-chars ${SIZE_MAXPC[$size]} --pad-seed $PAD_SEED \
        --output "$out" > "${out%.jsonl}.client.log" 2>&1 || rc=1
      log "  [$arm size=${size} wf=${wf}] done recs=$(grep -c . "$out" 2>/dev/null || echo 0) " \
          "preempt=$(grep -c -i preempt logs/${DATE}-whalegrid-${arm}-server.log 2>/dev/null || echo 0)"
    done
  done
  kill "$sv" 2>/dev/null; sleep 8; kill -9 "$sv" 2>/dev/null
  kill_ours
  return $rc
}

RC=0
for b in $BUDGETS; do run_budget "$b" || RC=1; done
if [ "$RC" = 0 ]; then touch logs/whalegrid_ALLDONE; else touch logs/whalegrid_FAILED; fi
log "done (rc=$RC)"
