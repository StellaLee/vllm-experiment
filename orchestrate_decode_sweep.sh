#!/bin/bash
# orchestrate_decode_sweep.sh -- sweep max-tokens {128,512,1024} on 14B / TP=2, Cs2=0.
# Per decode length: calibrate capacity knee -> run mono vs chunk at 0.8*knee (3 trials)
# -> tag outputs cs2repl_mt<MT>-* -> analyze TTFT + TPOT. Self-contained (nohup).
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DEV=0,1; TP=2
LOGROOT=logs/decodesweep; mkdir -p "$LOGROOT"
STAMP(){ date +%H:%M:%S; }
log(){ echo "[$(STAMP)] $*" >&2; }   # stderr: lands in log, not captured by $(calibrate)

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
  log "WARN GPUs still busy"
}
calibrate(){ # mt rates -> echoes knee rate
  local mt=$1 rates=$2
  local clog="$LOGROOT/calib-mt$mt.log"
  log "calibrate mt=$mt rates='$rates'"
  env CUDA_VISIBLE_DEVICES=$DEV PYTHON="$PYTHON" MODEL="$MODEL" \
      TP=$TP NUM=100 MAX_TOKENS=$mt RATES="$rates" bash scripts/mlsys/calib_cs2b.sh > "$clog" 2>&1
  free_gpus >/dev/null 2>&1
  python3 -c "import re;b=[float(re.search(r'rate=\s*([0-9.]+)',l).group(1)) for l in open('$clog') if 'BOUNDED' in l and re.search(r'rate=\s*([0-9.]+)',l)];print(b and max(b) or 0.4)"
}
sweep(){ # mt rate
  local mt=$1 rate=$2
  log "sweep mt=$mt rate=$rate"
  env CUDA_VISIBLE_DEVICES=$DEV PYTHON="$PYTHON" MODEL="$MODEL" \
      TP=$TP RATE=$rate MAX_TOKENS=$mt CV2_GRID="0" NUM_CONVS=80 TRIALS="1 2 3" \
      bash scripts/mlsys/run_cs2_repl.sh > "$LOGROOT/run-mt$mt.log" 2>&1
  for f in logs/*-cs2repl-*-cv*-t*.jsonl; do
    [ -f "$f" ] && mv "$f" "${f/-cs2repl-/-cs2repl_mt${mt}-}"
  done
  free_gpus >/dev/null 2>&1
  log "sweep mt=$mt done"
}
run_one(){ # mt rates
  local mt=$1 rates=$2 knee rate
  knee=$(calibrate "$mt" "$rates")
  rate=$(python3 -c "print(round(0.8*$knee,2))")
  log "mt=$mt knee=$knee -> rate=$rate (rho~0.8)"
  sweep "$mt" "$rate"
}

[ -f "$MODEL/config.json" ] || { log "14B model missing — ABORT"; touch logs/decodesweep_FAILED; exit 1; }
log "decode-length sweep start (14B TP=2, Cs2=0)"

# capacity drops as decode grows -> lower grids for longer max-tokens
run_one 128  "1.0 1.25 1.5"
run_one 512  "0.6 0.8 1.0"
run_one 1024 "0.4 0.6 0.8"

log "analyzing"
python scripts/mlsys/analyze_decode_sweep.py > logs/decodesweep_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/decodesweep_ANALYSIS.txt
touch logs/decodesweep_ALLDONE
log "orchestrator done -> logs/decodesweep_ANALYSIS.txt"
