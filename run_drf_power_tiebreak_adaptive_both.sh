#!/bin/bash
# Chains both reference conditions for drf_power_tiebreak_adaptive so both run unattended:
# matched (cache-hit-rate~0, heavy load -- where drf_power_tiebreak already wins) then
# cachehit (cache-hit-rate>0, light load -- where drf_power_tiebreak regresses).
set -x
cd /root/pli/vllm-experiment
bash run_drf_power_tiebreak_adaptive_matched.sh
bash run_drf_power_tiebreak_adaptive_cachehit.sh
echo "=== BOTH DRFADAPTIVE CONDITIONS COMPLETE ==="
