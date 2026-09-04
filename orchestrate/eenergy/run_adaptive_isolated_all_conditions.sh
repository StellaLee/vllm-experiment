#!/bin/bash
# Runs drf_power_tiebreak_adaptive_isolated across all 3 conditions: heavy/matched (where
# the naive adaptive design had its worst p99_ramp), light/cachehit (where it partially
# helped), and closed-loop-heavy (the new condition, most likely to exercise sustained
# concentration episodes).
set -x
cd /root/pli/vllm-experiment
bash orchestrate/eenergy/run_drf_power_tiebreak_adaptive_isolated_matched.sh
bash orchestrate/eenergy/run_drf_power_tiebreak_adaptive_isolated_cachehit.sh
bash orchestrate/eenergy/run_drf_power_tiebreak_adaptive_isolated_closedloopheavy.sh
echo "=== ALL ADAPTIVE_ISOLATED CONDITIONS COMPLETE ==="
