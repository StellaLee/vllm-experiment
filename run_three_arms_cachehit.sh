#!/bin/bash
# Fills in the missing light-load (cachehit) cells for the 3 hypothesis-test arms that were
# only ever run against the heavy/matched condition: drf_coincidence_tiebreak,
# drf_peak_power_tiebreak, drf_power_tiebreak_p2c.
set -x
cd /root/pli/vllm-experiment
bash run_drf_coincidence_tiebreak_cachehit.sh
bash run_drf_peak_power_tiebreak_cachehit.sh
bash run_drf_power_tiebreak_p2c_cachehit.sh
echo "=== ALL THREE CACHEHIT ARMS COMPLETE ==="
