#!/bin/bash
# Runs all 6 new arms (built this session, not in the original 8-arm "Ramp & Route" table)
# under the exact reconstructed Ramp & Route condition: closed-loop conc=24, whale-frac 0.15,
# max-tokens 128, N=6/GPUs 2-7, ramp_ceiling=450.
set -x
cd /root/pli/vllm-experiment
bash run_drf_power_tiebreak_rampandroute.sh
bash run_drf_coincidence_tiebreak_rampandroute.sh
bash run_drf_peak_power_tiebreak_rampandroute.sh
bash run_drf_power_tiebreak_p2c_rampandroute.sh
bash run_drf_power_tiebreak_adaptive_rampandroute.sh
bash run_drf_power_tiebreak_adaptive_isolated_rampandroute.sh
echo "=== ALL SIX RAMPANDROUTE ARMS COMPLETE ==="
