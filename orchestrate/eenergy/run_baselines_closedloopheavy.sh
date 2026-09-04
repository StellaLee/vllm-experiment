#!/bin/bash
# Fills the round_robin/lmetric gap for the heavy/closed-loop condition.
set -x
cd /root/pli/vllm-experiment
bash orchestrate/eenergy/run_round_robin_closedloopheavy.sh
bash orchestrate/eenergy/run_lmetric_closedloopheavy.sh
echo "=== BASELINES CLOSEDLOOPHEAVY COMPLETE ==="
