#!/bin/bash
# orchestrate_r15.sh -- autonomous remainder of the RATE=1.5 crossover experiment.
# Waits for the in-flight trial-1 run to finish, then runs trials 2 & 3, then writes
# the combined 3-trial paired analysis to logs/cs2_r15_ANALYSIS.txt and drops a
# DONE marker. Fully self-contained (nohup) so it survives session/watcher death.
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=0,1
export PYTHON="$(command -v python)"
export MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct

STAMP() { date +%H:%M:%S; }
echo "[$(STAMP)] orchestrator start"

# 1. wait for the trial-1 run to finish (PID recorded at launch)
T1PID=$(cut -d= -f2 /tmp/cs2_r15_pid 2>/dev/null)
if [ -n "$T1PID" ]; then
  echo "[$(STAMP)] waiting for trial-1 pid $T1PID ..."
  while kill -0 "$T1PID" 2>/dev/null; do sleep 15; done
fi
# also wait until any lingering server/replay from trial 1 is gone
while pgrep -f "run_cs2_repl|replay_sharegpt" >/dev/null 2>&1; do sleep 10; done
echo "[$(STAMP)] trial 1 complete"

# 2. run trials 2 and 3 (writes -t2 / -t3 files; rate/knobs identical to trial 1)
echo "[$(STAMP)] launching trials 2 & 3 ..."
env TP=2 RATE=1.5 CV2_GRID="0 1 2 4" NUM_CONVS=300 TRIALS="2 3" \
  bash scripts/mlsys/run_cs2_repl.sh > logs/cs2repl-14b-tp2-r15-t23.log 2>&1
echo "[$(STAMP)] trials 2 & 3 complete"

# 3. combined 3-trial paired analysis -> results file
echo "[$(STAMP)] analyzing ..."
python scripts/mlsys/analyze_r15.py > logs/cs2_r15_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/cs2_r15_ANALYSIS.txt
touch logs/cs2_r15_ALLDONE
echo "[$(STAMP)] orchestrator done -> logs/cs2_r15_ANALYSIS.txt"
