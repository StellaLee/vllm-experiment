#!/bin/bash
# orchestrate_crossover.sh -- 3-arm Cs^2 crossover sweep (mono/chunk/ours-slocvar) straddling
# the theoretical crossover Cs^2=1. 14B/TP=2, mt=1024, rate 0.64 FIXED across points (pad-mean
# constant => E[S] constant; only the variance changes). Each Cs^2 point gets a distinct PREFIX
# so output files don't collide; the controller arm starts at the chunk budget (512) and emits
# a per-step budget trace. Self-contained (nohup-able). Markers: logs/crossover_ALLDONE|FAILED.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
STAMP(){ date +%H:%M:%S; }
log(){ echo "[$(STAMP)] $*" >&2; }
free_gpus(){
  sleep 8
  for i in $(seq 1 24); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${used:-9999}" -lt 2000 ] && { log "GPUs free (${used}MiB)"; return; }
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      if tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023"; then kill -9 "$pid" 2>/dev/null; fi
    done
    sleep 5
  done
}
[ -f "$MODEL/config.json" ] || { log "14B missing ABORT"; touch logs/crossover_FAILED; exit 1; }
rm -f logs/crossover_ALLDONE logs/crossover_FAILED
log "crossover sweep start (14B/TP2 mt=1024 rate=0.64; Cs2 in {0.5,1.0,1.5}; 3-arm; ctrl start=512 trace on)"

run_point(){  # $1=cv2  $2=prefix
  local cv=$1 pfx=$2
  log "=== Cs2=$cv (prefix=$pfx) ==="
  env CUDA_VISIBLE_DEVICES=0,1 PYTHON="$PYTHON" MODEL="$MODEL" PREFIX="$pfx" \
      TP=2 RATE=0.64 MAX_TOKENS=1024 CV2="$cv" NUM_CONVS=80 TRIALS="1 2 3" \
      MONO_BUDGET=16384 CHUNK_BUDGET=512 FLOOR=512 SLO_MS=50 CVAR_PCTL=90 \
      bash scripts/run_cs2_3arm.sh > "logs/crossover-${pfx}-run.log" 2>&1
  free_gpus >/dev/null 2>&1
}

run_point 0.5 cross05
run_point 1.0 cross10
run_point 1.5 cross15

log "analyzing"
python scripts/analyze_crossover.py > logs/crossover_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/crossover_ANALYSIS.txt
touch logs/crossover_ALLDONE
log "crossover sweep done -> logs/crossover_ANALYSIS.txt"
