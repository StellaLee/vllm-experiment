#!/bin/bash
# Waits (server-side) for the currently-running drf_power_tiebreak_adaptive_isolated batch
# (3 conditions) to finish, then reruns drf_fixed for closedloopheavy -- its first attempt
# failed silently (router not ready when the harness started, "Connection refused" on all
# 150 conversations, 0 records collected across all 3 trials).
cd /root/pli/vllm-experiment
while ! grep -q "ALL ADAPTIVE_ISOLATED CONDITIONS COMPLETE" logs/chain_after_closedloopheavy_stdout.log 2>/dev/null; do
  sleep 20
done
echo "=== adaptive_isolated batch done, rerunning drf_fixed closedloopheavy ==="
bash run_drf_fixed_closedloopheavy.sh
echo "=== DRF_FIXED CLOSEDLOOPHEAVY RERUN COMPLETE ==="
