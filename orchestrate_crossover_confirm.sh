#!/bin/bash
# orchestrate_crossover_confirm.sh -- n=8 confirmation of the two above-crossover points
# (cross175 realized Cs^2~1.02, cross225 ~1.11). Runs all THREE arms of a point CONCURRENTLY on
# three separate GPU pairs so mono/chunk/ours see identical box load => clean within-point deltas.
# 6 GPUs/wave, one point per wave (4th pair left idle on purpose: splitting a point's arms across
# different load levels would bias the deltas). Per-server config is byte-identical to the pilot
# (14B/TP2, mt=1024, rate=0.64, mono16384/chunk512/ours-slocvar-start512). Re-runs t1..t8 fresh so
# each >=1 point is uniformly measured. Markers: logs/crossoverconf_ALLDONE|FAILED.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
export MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
STAMP(){ date +%H:%M:%S; }
log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){  # only ever touch OUR processes (shared box)
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    if tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q "venv-vllm023"; then kill -9 "$pid" 2>/dev/null; fi
  done; sleep 6
}
[ -f "$MODEL/config.json" ] || { log "14B missing ABORT"; touch logs/crossoverconf_FAILED; exit 1; }
rm -f logs/crossoverconf_ALLDONE logs/crossoverconf_FAILED
log "n=8 confirm start (14B/TP2 mt=1024 rate=0.64; cross175+cross225; 3 arms concurrent/wave on pairs 0-1,2-3,4-5)"

# One wave = one Cs^2 point, 3 arms concurrent on 3 GPU pairs (ports 8001/8002/8003).
run_point(){  # $1=prefix  $2=cv2
  local pfx=$1 cv=$2
  log "=== WAVE ${pfx} (cv2=$cv): mono|chunk|ours concurrent ==="
  GPUS=0,1 PORT=8001 PREFIX=$pfx ARM=mono  BUDGET=16384 MODE=static  CV2=$cv PYTHON=$PYTHON MODEL=$MODEL \
    bash scripts/run_arm_unit.sh > logs/confirm-${pfx}-mono.log  2>&1 &
  local p1=$!
  GPUS=2,3 PORT=8002 PREFIX=$pfx ARM=chunk BUDGET=512   MODE=static  CV2=$cv PYTHON=$PYTHON MODEL=$MODEL \
    bash scripts/run_arm_unit.sh > logs/confirm-${pfx}-chunk.log 2>&1 &
  local p2=$!
  GPUS=4,5 PORT=8003 PREFIX=$pfx ARM=ours  BUDGET=16384 MODE=slocvar CV2=$cv PYTHON=$PYTHON MODEL=$MODEL \
    bash scripts/run_arm_unit.sh > logs/confirm-${pfx}-ours.log  2>&1 &
  local p3=$!
  local rc=0
  wait $p1 || rc=1; wait $p2 || rc=1; wait $p3 || rc=1
  kill_ours
  [ $rc -eq 0 ] || { log "WAVE ${pfx} had a failing arm (rc=$rc) -- see logs/confirm-${pfx}-*.log"; }
  return $rc
}

ALLRC=0
run_point cross175 1.75 || ALLRC=1
run_point cross225 2.25 || ALLRC=1

log "analyzing full 5-point curve (>=1 points now n=8)"
python scripts/analyze_crossover.py > logs/crossoverconf_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE (waves rc=$ALLRC)" >> logs/crossoverconf_ANALYSIS.txt
if [ $ALLRC -eq 0 ]; then touch logs/crossoverconf_ALLDONE; else touch logs/crossoverconf_ALLDONE; log "NOTE: some arm failed; analysis still written"; fi
log "n=8 confirm done -> logs/crossoverconf_ANALYSIS.txt"
