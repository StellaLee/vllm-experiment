#!/bin/bash
# orchestrate_controller.sh -- 3-arm controller run at the stall-bound point found by the
# decode sweep (14B/TP=2, mt=1024, Cs2=0, rate 0.64 = 0.8*knee from the sweep). Self-contained.
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
[ -f "$MODEL/config.json" ] || { log "14B missing ABORT"; touch logs/controller_FAILED; exit 1; }
log "controller 3-arm start (14B/TP2 mt=1024 cv2=0 rate=0.64 slocvar slo=50 cvar=90)"

env CUDA_VISIBLE_DEVICES=0,1 PYTHON="$PYTHON" MODEL="$MODEL" \
    TP=2 RATE=0.64 MAX_TOKENS=1024 CV2=0 NUM_CONVS=80 TRIALS="1 2 3" \
    MONO_BUDGET=16384 CHUNK_BUDGET=512 FLOOR=512 SLO_MS=50 CVAR_PCTL=90 \
    bash scripts/run_cs2_3arm.sh > logs/controller-run.log 2>&1
free_gpus >/dev/null 2>&1

log "analyzing"
python scripts/analyze_3arm.py > logs/controller_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/controller_ANALYSIS.txt
touch logs/controller_ALLDONE
log "orchestrator done -> logs/controller_ANALYSIS.txt"
