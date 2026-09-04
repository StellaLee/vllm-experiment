#!/bin/bash
# Waits (server-side) for the currently-running 5-arm closedloopheavy batch to finish, then
# launches the drf_power_tiebreak_adaptive_isolated batch across all 3 conditions.
cd /root/pli/vllm-experiment
while ! grep -q "ALL FIVE CLOSEDLOOPHEAVY ARMS COMPLETE" logs/chain_after_cachehit_stdout.log 2>/dev/null; do
  sleep 20
done
echo "=== closedloopheavy batch done, starting adaptive_isolated batch ==="
bash run_adaptive_isolated_all_conditions.sh
