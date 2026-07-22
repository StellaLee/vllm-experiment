#!/bin/bash
# orchestrate_sharegpt_cl.sh -- 3-arm (mono/chunk/ours) on ShareGPT, CLOSED-LOOP, MULTI-TURN.
# This is the near-saturation regime where chunking historically showed a TPOT-tail benefit
# (decode starvation) -- unlike open-loop where decode never stalls. Real ShareGPT prompts (no
# synthetic pad); service-size variance comes from natural turn/history growth. Prefix caching
# left ON (realistic for multi-turn; reused equally by all arms). All 3 arms run CONCURRENTLY on
# 3 GPU pairs so they see identical box load -> clean deltas. OpenAI api_server (/v1/completions),
# same engine flags + scheduler patch as the synthetic sweep. Markers: logs/sharecl_ALLDONE|FAILED.
# Tunables: CONC (concurrency) TURNS NCONV MAXTOK TRIALS.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ # SIGTERM our parent api_servers first (graceful -> tears down TP workers), then force any stragglers
  pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/sharecl_ALLDONE logs/sharecl_FAILED
DATE=$(date +%Y-%m-%d)
CONC=${CONC:-16}; TURNS=${TURNS:-4}; NCONV=${NCONV:-60}; MAXTOK=${MAXTOK:-1024}; TRIALS=${TRIALS:-"1 2 3"}
MAX_SEQS=${MAX_SEQS:-32}; PAD_MEAN=${PAD_MEAN:-0}; PAD_CV2=${PAD_CV2:-0}
PAD_MIN=${PAD_MIN:-100}; PAD_MAX=${PAD_MAX:-50000}
RUN_ARMS=${RUN_ARMS:-"mono chunk ours feed"}
FLOOR=512; SLO_MS=50; CVAR_PCTL=90
log "closed-loop 3/4-arm: arms=[$RUN_ARMS] conc=$CONC turns=$TURNS nconv=$NCONV mt=$MAXTOK max_seqs=$MAX_SEQS pad_mean=$PAD_MEAN pad_cv2=$PAD_CV2 trials=[$TRIALS]"

run_arm(){ # gpus port arm budget mode
  local gpus=$1 port=$2 arm=$3 budget=$4 mode=$5
  local EX="DYNAMIC_CHUNK=0"
  if [ "$mode" = slocvar ] || [ "$mode" = feedforward ]; then
    EX="DYNAMIC_CHUNK=1 CHUNK_MODE=$mode DYNAMIC_CHUNK_MIN=$FLOOR DYNAMIC_CHUNK_SLO_MS=$SLO_MS DYNAMIC_CHUNK_CVAR_PCTL=$CVAR_PCTL DYNAMIC_CHUNK_START=$FLOOR DYNAMIC_CHUNK_TRACE=logs/${DATE}-sharecl-${arm}-chunktrace.csv"
  fi
  log "  [$arm] server gpus=$gpus port=$port budget=$budget mode=$mode"
  env CUDA_VISIBLE_DEVICES=$gpus PREFIX_REORDER=0 $EX $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $port --max-num-seqs $MAX_SEQS --max-num-batched-tokens $budget \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-sharecl-${arm}-server.log 2>&1 &
  local sv=$!
  for i in $(seq 1 100); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-sharecl-${arm}-server.log && break
    [ "$i" = 100 ] && { log "  [$arm] SERVER TIMEOUT"; kill "$sv" 2>/dev/null; return 1; }
  done
  local rc=0
  for tr in $TRIALS; do
    local out="logs/${DATE}-sharecl-${arm}-t${tr}.jsonl"
    $PYTHON src/replay_sharegpt.py --host localhost --port $port --model "$MODEL" \
      --dataset "$DATASET" --num-convs $NCONV --max-turns $TURNS --min-turns $TURNS \
      --max-tokens $MAXTOK --concurrency $CONC \
      --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min $PAD_MIN --pad-max $PAD_MAX --pad-seed $((1000+tr)) \
      --output "$out" > "${out%.jsonl}.client.log" 2>&1 || rc=1
    log "  [$arm] trial $tr done recs=$(grep -c . "$out" 2>/dev/null || echo 0)"
  done
  kill "$sv" 2>/dev/null; sleep 10; kill -9 "$sv" 2>/dev/null
  return $rc
}

# arm registry: name -> "gpus port budget mode"
declare -A ARMCFG=(
  [mono]="0,1 8031 16384 static"
  [chunk]="2,3 8032 512 static"
  [ours]="4,5 8033 16384 slocvar"
  [feed]="6,7 8034 16384 feedforward"
)
log "launching arms concurrently: [$RUN_ARMS]"
PIDS=""
for a in $RUN_ARMS; do
  set -- ${ARMCFG[$a]}; run_arm "$1" "$2" "$a" "$3" "$4" & PIDS="$PIDS $!"
done
RC=0; for p in $PIDS; do wait $p || RC=1; done
kill_ours
log "analyzing"
$PYTHON scripts/analyze_sharegpt_cl.py > logs/sharecl_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE (rc=$RC)" >> logs/sharecl_ANALYSIS.txt
touch logs/sharecl_ALLDONE
log "done -> logs/sharecl_ANALYSIS.txt"
