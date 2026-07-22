#!/bin/bash
# orchestrate_chunk_sweep.sh -- sweep the prefill token-budget (chunk size) to find the knee where
# chunking beats mono. SINGLE-TURN closed-loop (every request carries a real prefill; no prefix-cache
# dilution), heavy-tailed prompt mix (whales), near-saturation load. This is the regime built to FAVOR
# chunk: short requests queued behind whales are what chunking rescues (TTFT head-of-line), and a live
# decode batch is present for any TPOT-protection to show. All arms run CONCURRENTLY on separate GPU
# pairs -> identical box load -> valid RANKING among budgets (phase 1). Phase 2 (separate) re-runs the
# winner vs mono ISOLATED for a contention-free headline delta.
# Arms are named b<budget>; b16384 == mono (no chunking). Markers: logs/sweep_ALLDONE|FAILED.
# Tunables: BUDGETS CONC MAX_SEQS MAXTOK NCONV PAD_MEAN PAD_CV2 PAD_MAX TRIALS.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/sweep_ALLDONE logs/sweep_FAILED
DATE=$(date +%Y-%m-%d)
BUDGETS=${BUDGETS:-"16384 512 1024 2048"}   # first entry treated as mono baseline
CONC=${CONC:-40}; MAX_SEQS=${MAX_SEQS:-48}; MAXTOK=${MAXTOK:-512}; NCONV=${NCONV:-240}
PAD_MEAN=${PAD_MEAN:-1800}; PAD_CV2=${PAD_CV2:-4}; PAD_MIN=${PAD_MIN:-100}; PAD_MAX=${PAD_MAX:-40000}
TRIALS=${TRIALS:-"1"}
log "chunk-size sweep: budgets=[$BUDGETS] conc=$CONC max_seqs=$MAX_SEQS mt=$MAXTOK nconv=$NCONV pad_mean=$PAD_MEAN pad_cv2=$PAD_CV2 pad_max=$PAD_MAX trials=[$TRIALS] (single-turn)"

# assign a GPU pair + port per budget, in order
declare -a GPUPAIRS=("0,1" "2,3" "4,5" "6,7")
run_arm(){ # gpus port budget
  local gpus=$1 port=$2 budget=$3 arm="b$3"
  log "  [$arm] server gpus=$gpus port=$port budget=$budget"
  env CUDA_VISIBLE_DEVICES=$gpus PREFIX_REORDER=0 DYNAMIC_CHUNK=0 $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $port --max-num-seqs $MAX_SEQS --max-num-batched-tokens $budget \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-sweep-${arm}-server.log 2>&1 &
  local sv=$!
  for i in $(seq 1 100); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-sweep-${arm}-server.log && break
    [ "$i" = 100 ] && { log "  [$arm] SERVER TIMEOUT"; kill "$sv" 2>/dev/null; return 1; }
  done
  local rc=0
  for tr in $TRIALS; do
    local out="logs/${DATE}-sweep-${arm}-t${tr}.jsonl"
    $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
      --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
      --max-tokens $MAXTOK --concurrency $CONC \
      --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min $PAD_MIN --pad-max $PAD_MAX --pad-seed $((1000+tr)) \
      --output "$out" > "${out%.jsonl}.client.log" 2>&1 || rc=1
    log "  [$arm] trial $tr done recs=$(grep -c . "$out" 2>/dev/null || echo 0)"
  done
  kill "$sv" 2>/dev/null; sleep 10; kill -9 "$sv" 2>/dev/null
  return $rc
}

log "launching arms concurrently"
PIDS=""; i=0
for b in $BUDGETS; do
  pair=${GPUPAIRS[$i]}; port=$((8040+i))
  run_arm "$pair" "$port" "$b" & PIDS="$PIDS $!"
  i=$((i+1))
done
RC=0; for p in $PIDS; do wait $p || RC=1; done
kill_ours
log "analyzing"
BUDGETS="$BUDGETS" $PYTHON scripts/analyze_chunk_sweep.py > logs/sweep_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE (rc=$RC)" >> logs/sweep_ANALYSIS.txt
touch logs/sweep_ALLDONE
log "done -> logs/sweep_ANALYSIS.txt"
