#!/bin/bash
# Real server-side cache-hit-rate verification for the remaining 7 arms under light/cachehit
# (drf_power_tiebreak already done: real 46.92+-0.47% vs mirror 46.9+-1.0%, confirmed match).
set -x
cd /root/pli/vllm-experiment
bash orchestrate/eenergy/run_round_robin_cachehit_realcache.sh
bash orchestrate/eenergy/run_lmetric_cachehit_realcache.sh
bash orchestrate/eenergy/run_drf_fixed_cachehit_realcache.sh
bash orchestrate/eenergy/run_drf_peak_power_tiebreak_cachehit_realcache.sh
bash orchestrate/eenergy/run_drf_power_tiebreak_p2c_cachehit_realcache.sh
bash orchestrate/eenergy/run_drf_power_tiebreak_adaptive_cachehit_realcache.sh
bash orchestrate/eenergy/run_drf_power_tiebreak_adaptive_isolated_cachehit_realcache.sh
echo "=== ALL SEVEN CACHEHIT REALCACHE ARMS COMPLETE ==="
