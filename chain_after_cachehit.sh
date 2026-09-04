#!/bin/bash
# Waits (server-side) for the currently-running 3-arm cachehit batch to finish, then
# launches the 5-arm closed-loop-heavy batch on the same GPUs.
cd /root/pli/vllm-experiment
while ! grep -q "ALL THREE CACHEHIT ARMS COMPLETE" logs/run_three_arms_cachehit_stdout.log 2>/dev/null; do
  sleep 20
done
echo "=== cachehit batch done, starting closedloopheavy batch ==="
bash run_five_arms_closedloopheavy.sh
