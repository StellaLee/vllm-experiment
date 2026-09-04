#!/bin/bash
# Runs all 5 arms for the new heavy-load closed-loop condition sequentially: the two
# established baselines (drf_fixed, drf_power_tiebreak) plus the 3 arms still missing a
# comparison point here (drf_coincidence_tiebreak, drf_peak_power_tiebreak,
# drf_power_tiebreak_p2c).
set -x
cd /root/pli/vllm-experiment
bash orchestrate/eenergy/run_drf_fixed_closedloopheavy.sh
bash orchestrate/eenergy/run_drf_power_tiebreak_closedloopheavy.sh
bash orchestrate/eenergy/run_drf_coincidence_tiebreak_closedloopheavy.sh
bash orchestrate/eenergy/run_drf_peak_power_tiebreak_closedloopheavy.sh
bash orchestrate/eenergy/run_drf_power_tiebreak_p2c_closedloopheavy.sh
echo "=== ALL FIVE CLOSEDLOOPHEAVY ARMS COMPLETE ==="
