#!/bin/bash
# orchestrate_longprompt.sh -- LONG-PROMPT (12-16k whale) test of the 2048-vs-16k rationale.
# Bimodal single-turn workload: ~85% short requests keep a live decode batch, ~15% are 12-16k-token
# whales. Under mono (budget 16384) a whale prefills in ONE giant iteration that stalls the whole
# decode batch -> P99 TBT spike; under chunk (2048/512) it is sliced -> bounded TBT. Low concurrency
# so whales don't blow KV capacity (the overload run's mistake). Arms run ISOLATED & SEQUENTIAL (each
# alone on GPUs 0,1; other 6 idle) -> zero cross-arm PCIe/NCCL contention -> clean headline.
# Measures P99 TBT (pooled per-token ITL) + TTFT. Markers: logs/longp_ALLDONE|FAILED.
# Tunables: BUDGETS CONC MAX_SEQS MAXTOK NCONV WHALE_FRAC WHALE_MIN WHALE_MAX TRIALS.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/longp_ALLDONE logs/longp_FAILED
DATE=$(date +%Y-%m-%d)
BUDGETS=${BUDGETS:-"16384 2048 512"}      # first = mono baseline
CONC=${CONC:-20}; MAX_SEQS=${MAX_SEQS:-48}; MAXTOK=${MAXTOK:-256}; NCONV=${NCONV:-200}
WHALE_FRAC=${WHALE_FRAC:-0.15}; WHALE_MIN=${WHALE_MIN:-48000}; WHALE_MAX=${WHALE_MAX:-60000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-62000}   # (16384-256)*~4 chars/tok, guard vs context overflow
PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}    # small base pad for the short majority
TRIALS=${TRIALS:-"1"}
log "long-prompt test: budgets=[$BUDGETS] conc=$CONC max_seqs=$MAX_SEQS mt=$MAXTOK nconv=$NCONV whale_frac=$WHALE_FRAC whale=[$WHALE_MIN,$WHALE_MAX]chars trials=[$TRIALS] (ISOLATED sequential)"

run_arm(){ # budget
  local budget=$1 arm="b$1" port=8050
  log "  [$arm] server on GPUs 0,1 port=$port budget=$budget"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $port --max-num-seqs $MAX_SEQS --max-num-batched-tokens $budget \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-longp-${arm}-server.log 2>&1 &
  local sv=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-longp-${arm}-server.log && break
    [ "$i" = 120 ] && { log "  [$arm] SERVER TIMEOUT"; kill "$sv" 2>/dev/null; return 1; }
  done
  local rc=0
  for tr in $TRIALS; do
    local out="logs/${DATE}-longp-${arm}-t${tr}.jsonl"
    $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
      --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
      --max-tokens $MAXTOK --concurrency $CONC \
      --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
      --whale-frac $WHALE_FRAC --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
      --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed $((1000+tr)) \
      --output "$out" > "${out%.jsonl}.client.log" 2>&1 || rc=1
    log "  [$arm] trial $tr done recs=$(grep -c . "$out" 2>/dev/null || echo 0) preempt=$(grep -c -i preempt logs/${DATE}-longp-${arm}-server.log 2>/dev/null || echo 0)"
  done
  kill "$sv" 2>/dev/null; sleep 8; kill -9 "$sv" 2>/dev/null
  kill_ours
  return $rc
}

RC=0
for b in $BUDGETS; do run_arm "$b" || RC=1; done
log "analyzing"
BUDGETS="$BUDGETS" $PYTHON scripts/analyze_longprompt.py > logs/longp_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE (rc=$RC)" >> logs/longp_ANALYSIS.txt
touch logs/longp_ALLDONE
log "done -> logs/longp_ANALYSIS.txt"
