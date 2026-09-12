#!/bin/bash
# Extends the Poisson comparison from a single 92.5%-cap point to a full 4-point sweep
# matching the closed-loop gating-intensity sweep -- reuses the SAME absolute cap values as
# the closed-loop 80%/65% points (470W/716W, 382W/582W), just under Poisson arrivals this
# time, so each new point is directly comparable to its closed-loop counterpart at the same
# cap. n=1 each, consistent with the closed-loop 80%/65% points' own replication level.
set -x
cd /root/pli/vllm-experiment

OUTNAME=disagg_8gpu_poisson_gated_80pct CAP_PREFILL_W=470 CAP_DECODE_W=716 ARRIVAL_MODE=poisson \
  bash orchestrate/eenergy/run_disagg_8gpu_battery.sh

OUTNAME=disagg_8gpu_poisson_gated_65pct CAP_PREFILL_W=382 CAP_DECODE_W=582 ARRIVAL_MODE=poisson \
  bash orchestrate/eenergy/run_disagg_8gpu_battery.sh

echo "=== DISAGG_8GPU_POISSON_INTENSITY_SWEEP_COMPLETE ==="
