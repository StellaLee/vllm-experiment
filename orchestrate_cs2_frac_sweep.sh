#!/bin/bash
# orchestrate_cs2_frac_sweep.sh -- cheap first-pass crossover sweep along the Cs^2 axis.
# Fixes whale size at the known-working point (44-50k chars ~= 12k tokens, same as
# orchestrate_longprompt.sh) and sweeps whale FRACTION {0,5,10,15}% to trace where the
# chunk-vs-mono TBT-tail win turns on. Same PAD_SEED across every (budget, frac) combo so
# whale positions are paired/nested (higher frac is a superset of lower frac's whale draws --
# see replay_sharegpt.py's sample_pad_len docstring), giving a clean monotonic comparison.
# Cheaper than orchestrate_longprompt.sh's per-arm restart pattern: server boots ONCE per
# budget and serves all 4 fractions (whale-frac is a client-side replay param, not a server
# flag), so this is 3 server boots + 12 client runs instead of 12 full restarts.
# Tunables: BUDGETS FRAC_PCTS CONC MAX_SEQS MAXTOK NCONV WHALE_MIN WHALE_MAX MAX_PROMPT_CHARS.
# Markers: logs/cs2fracsweep_ALLDONE|FAILED.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f "logs/cs2fracsweep_n${NCONV:-200}_ALLDONE" "logs/cs2fracsweep_n${NCONV:-200}_FAILED"
DATE=$(date +%Y-%m-%d)
BUDGETS=${BUDGETS:-"16384 2048 512"}      # first = mono baseline
FRAC_PCTS=${FRAC_PCTS:-"0 5 10 15"}       # whale fraction, as integer percent
CONC=${CONC:-20}; MAX_SEQS=${MAX_SEQS:-48}; MAXTOK=${MAXTOK:-256}; NCONV=${NCONV:-200}
WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}
PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
PAD_SEED=${PAD_SEED:-1001}                # fixed across every (budget,frac) -> paired/nested whales
log "cs2 fraction sweep: budgets=[$BUDGETS] fracs%=[$FRAC_PCTS] conc=$CONC max_seqs=$MAX_SEQS mt=$MAXTOK nconv=$NCONV whale=[$WHALE_MIN,$WHALE_MAX]chars seed=$PAD_SEED"

run_budget(){ # budget
  local budget=$1 arm="b$1" port=8050
  log "  [$arm] server on GPUs 0,1 port=$port budget=$budget"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $port --max-num-seqs $MAX_SEQS --max-num-batched-tokens $budget \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-cs2f-${arm}-server.log 2>&1 &
  local sv=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-cs2f-${arm}-server.log && break
    [ "$i" = 120 ] && { log "  [$arm] SERVER TIMEOUT"; kill "$sv" 2>/dev/null; return 1; }
  done
  local rc=0
  for pct in $FRAC_PCTS; do
    local ftag=$(printf "%02d" "$pct")
    local frac=$($PYTHON -c "print($pct/100)")
    local out="logs/${DATE}-cs2f-${arm}-f${ftag}-n${NCONV}-t1.jsonl"
    $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
      --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
      --max-tokens $MAXTOK --concurrency $CONC \
      --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
      --whale-frac $frac --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
      --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed $PAD_SEED \
      --output "$out" > "${out%.jsonl}.client.log" 2>&1 || rc=1
    log "  [$arm f=${pct}%] done recs=$(grep -c . "$out" 2>/dev/null || echo 0)"
  done
  kill "$sv" 2>/dev/null; sleep 8; kill -9 "$sv" 2>/dev/null
  kill_ours
  return $rc
}

RC=0
for b in $BUDGETS; do run_budget "$b" || RC=1; done
log "analyzing"
OUT_ANALYSIS="logs/cs2fracsweep_n${NCONV}_ANALYSIS.txt"
BUDGETS="$BUDGETS" FRAC_PCTS="$FRAC_PCTS" NCONV="$NCONV" $PYTHON scripts/mlsys/analyze_cs2_frac_sweep.py > "$OUT_ANALYSIS" 2>&1
echo "[$(STAMP)] DONE (rc=$RC)" >> "$OUT_ANALYSIS"
if [ "$RC" = 0 ]; then touch "logs/cs2fracsweep_n${NCONV}_ALLDONE"; else touch "logs/cs2fracsweep_n${NCONV}_FAILED"; fi
log "done -> $OUT_ANALYSIS"
