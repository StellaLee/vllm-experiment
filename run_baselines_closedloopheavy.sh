#!/bin/bash
# Fills the round_robin/lmetric gap for the heavy/closed-loop condition.
set -x
cd /root/pli/vllm-experiment
bash run_round_robin_closedloopheavy.sh
bash run_lmetric_closedloopheavy.sh
echo "=== BASELINES CLOSEDLOOPHEAVY COMPLETE ==="
