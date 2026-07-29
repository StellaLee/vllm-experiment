#!/bin/bash
# orchestrate_r10.sh -- full 3-trial crossover at RATE=1.0 (the cold-validated sub-sat rate).
# Runs trials 1,2,3 in one run_cs2_repl invocation (2 server boots total), then writes the
# combined paired analysis to logs/cs2_r10_ANALYSIS.txt and an ALLDONE marker. Self-contained.
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=0,1
export PYTHON="$(command -v python)"
export MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
STAMP() { date +%H:%M:%S; }

echo "[$(STAMP)] rate-1.0 orchestrator start"
env TP=2 RATE=1.0 CV2_GRID="0 1 2 4" NUM_CONVS=150 TRIALS="1 2 3" \
  bash scripts/mlsys/run_cs2_repl.sh > logs/cs2repl-14b-tp2-r10.log 2>&1
echo "[$(STAMP)] all trials complete; analyzing"
python scripts/mlsys/analyze_r15.py > logs/cs2_r10_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/cs2_r10_ANALYSIS.txt
touch logs/cs2_r10_ALLDONE
echo "[$(STAMP)] orchestrator done -> logs/cs2_r10_ANALYSIS.txt"
