#!/bin/bash
# orchestrate_crossover_hi.sh -- extend the Cs^2 sweep ABOVE the crossover. The first sweep
# (nominal pad-cv2 0.5/1.0/1.5) only reached REALIZED Cs^2 up to 0.91 -- lognormal clipping at
# pad_max compresses the knob ~0.6x, so we never crossed 1. These two points (nominal 1.75/2.25
# -> realized ~1.05/~1.25 by the measured 0.62*nom+0.06 mapping) straddle Cs^2=1 so we can see
# whether chunk's TTFT penalty flips sign. Same 14B/TP2, mt=1024, rate=0.64, pad_mean=8000 (E[S]
# held ~constant). Markers: logs/crossoverhi_ALLDONE|FAILED. Self-contained (nohup/setsid-able).
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
[ -f "$MODEL/config.json" ] || { log "14B missing ABORT"; touch logs/crossoverhi_FAILED; exit 1; }
rm -f logs/crossoverhi_ALLDONE logs/crossoverhi_FAILED
log "crossover-hi sweep start (14B/TP2 mt=1024 rate=0.64; nominal cv2 in {1.75,2.25}; 3-arm; ctrl start=512 trace on)"

run_point(){  # $1=cv2  $2=prefix
  local cv=$1 pfx=$2
  log "=== nominal cv2=$cv (prefix=$pfx) ==="
  env CUDA_VISIBLE_DEVICES=0,1 PYTHON="$PYTHON" MODEL="$MODEL" PREFIX="$pfx" \
      TP=2 RATE=0.64 MAX_TOKENS=1024 CV2="$cv" NUM_CONVS=80 TRIALS="1 2 3" \
      MONO_BUDGET=16384 CHUNK_BUDGET=512 FLOOR=512 SLO_MS=50 CVAR_PCTL=90 \
      bash scripts/run_cs2_3arm.sh > "logs/crossoverhi-${pfx}-run.log" 2>&1
  free_gpus >/dev/null 2>&1
}

run_point 1.75 cross175
run_point 2.25 cross225

log "analyzing (full 5-point curve)"
python scripts/analyze_crossover.py > logs/crossoverhi_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/crossoverhi_ANALYSIS.txt
touch logs/crossoverhi_ALLDONE
log "crossover-hi sweep done -> logs/crossoverhi_ANALYSIS.txt"
