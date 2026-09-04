#!/bin/bash
# orchestrate_7b_nccl.sh -- fully automated 7B single-GPU vs TP=2 NCCL control.
# Waits for 7B download -> calibrates each config (highest COLD-bounded rate, NUM=250) ->
# runs the mono-vs-chunk Cs2 sweep for both -> tags outputs per config -> analyzes.
# Self-contained (nohup); survives session death. NOT set -e (continue on transient errors).
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL7B=/data/pli/models/Qwen2.5-Coder-7B-Instruct
LOGROOT=logs/7bnccl; mkdir -p "$LOGROOT"
STAMP(){ date +%H:%M:%S; }
# log to STDERR so it lands in the nohup log (via 2>&1) but is NOT captured by
# $(calibrate ...) command substitution — the rate must be calibrate's only stdout.
log(){ echo "[$(STAMP)] $*" >&2; }

# free GPUs safely between phases: only kill OUR vllm procs (cmdline references our venv),
# so another user's job on the shared box is never touched.
free_gpus(){
  sleep 8
  for i in $(seq 1 24); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${used:-9999}" -lt 2000 ] && { log "GPUs free (${used}MiB)"; return; }
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      if tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023"; then
        kill -9 "$pid" 2>/dev/null
      fi
    done
    sleep 5
  done
  log "WARN: GPUs still busy after cleanup"
}

# ---- 1. wait for 7B download + verify ----
log "waiting for 7B download..."
DLPID=$(cut -d= -f2 /data/pli/models/dl_qwen7b.pid 2>/dev/null)
[ -n "$DLPID" ] && while kill -0 "$DLPID" 2>/dev/null; do sleep 30; done
if ! python - <<PY
import json, os, sys
d="$MODEL7B"
idx=json.load(open(d+"/model.safetensors.index.json"))
miss=[s for s in set(idx["weight_map"].values()) if not os.path.exists(os.path.join(d,s))]
sys.exit(1 if miss else 0)
PY
then log "7B model incomplete — ABORT"; touch logs/cs2_7bnccl_FAILED; exit 1; fi
log "7B download verified"

# ---- helpers ----
calibrate(){ # label TP devices rates  -> echoes highest-bounded (capacity knee) rate
  local label=$1 tp=$2 dev=$3 rates=$4
  local clog="$LOGROOT/calib-$label.log"
  log "calibrate $label (TP=$tp dev=$dev rates='$rates')"
  env CUDA_VISIBLE_DEVICES=$dev PYTHON="$PYTHON" MODEL="$MODEL7B" \
      TP=$tp NUM=250 RATES="$rates" bash scripts/calib_cs2b.sh > "$clog" 2>&1
  free_gpus >/dev/null 2>&1
  python3 -c "import re;b=[float(re.search(r'rate=\s*([0-9.]+)',l).group(1)) for l in open('$clog') if 'BOUNDED' in l and re.search(r'rate=\s*([0-9.]+)',l)];print(b and max(b) or 1.5)"
}
sweep(){ # label TP devices rate
  local label=$1 tp=$2 dev=$3 rate=$4
  log "sweep $label @ rate=$rate (TP=$tp dev=$dev)"
  env CUDA_VISIBLE_DEVICES=$dev PYTHON="$PYTHON" MODEL="$MODEL7B" \
      TP=$tp RATE=$rate CV2_GRID="0 2 4" NUM_CONVS=150 TRIALS="1 2 3" \
      bash scripts/run_cs2_repl.sh > "$LOGROOT/run-$label.log" 2>&1
  # tag outputs per config (run_cs2_repl uses fixed prefix 'cs2repl'); the trailing dash
  # in '-cs2repl-' ensures already-tagged files are not re-matched.
  for f in logs/*-cs2repl-*-cv*-t*.jsonl; do
    [ -f "$f" ] && mv "$f" "${f/-cs2repl-/-cs2repl${label}-}"
  done
  free_gpus >/dev/null 2>&1
  log "sweep $label done"
}

# ---- 2. single-GPU (no NCCL) — find capacity knee, run at rho~0.8 ----
KNEE_SINGLE=$(calibrate SINGLE 1 0 "2.0 2.5 3.0")
R_SINGLE=$(python3 -c "print(round(0.8*$KNEE_SINGLE,2))")
log "SINGLE knee=$KNEE_SINGLE -> matched rho~0.8 rate=$R_SINGLE"
sweep SINGLE 1 0 "$R_SINGLE"

# ---- 3. TP=2 (host-staged NCCL) — probe higher (true cap unknown), run at rho~0.8 ----
KNEE_TP2=$(calibrate TP2 2 0,1 "2.5 3.0 4.0 5.0")
R_TP2=$(python3 -c "print(round(0.8*$KNEE_TP2,2))")
log "TP2 knee=$KNEE_TP2 -> matched rho~0.8 rate=$R_TP2"
sweep TP2 2 0,1 "$R_TP2"

# ---- 4. analyze ----
log "analyzing"
python scripts/analyze_7b_nccl.py "$R_SINGLE" "$R_TP2" > logs/cs2_7bnccl_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE (R_SINGLE=$R_SINGLE R_TP2=$R_TP2)" >> logs/cs2_7bnccl_ANALYSIS.txt
touch logs/cs2_7bnccl_ALLDONE
log "orchestrator done -> logs/cs2_7bnccl_ANALYSIS.txt"
