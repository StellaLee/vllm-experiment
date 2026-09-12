#!/bin/bash
# Driver: runs 4 follow-on batteries sequentially on run_disagg_8gpu_battery.sh (share the
# same 8 GPUs, can't run in parallel) -- (1-2) Poisson-arrival verification of the existing
# baseline/gated result (rate=10.7, matching this project's established colocated "Matched"
# condition), (3-4) two additional gating-intensity points (80%/65% of baseline mean per
# pool) for the TTFT/TPOT-vs-peak-power diagram, alongside the already-collected no-gate and
# 92.5%-of-mean points.
set -x
cd /root/pli/vllm-experiment

OUTNAME=disagg_8gpu_poisson_baseline CAP_PREFILL_W= CAP_DECODE_W= ARRIVAL_MODE=poisson \
  bash orchestrate/eenergy/run_disagg_8gpu_battery.sh

OUTNAME=disagg_8gpu_poisson_gated CAP_PREFILL_W=544 CAP_DECODE_W=828 ARRIVAL_MODE=poisson \
  bash orchestrate/eenergy/run_disagg_8gpu_battery.sh

OUTNAME=disagg_8gpu_gated_80pct CAP_PREFILL_W=470 CAP_DECODE_W=716 ARRIVAL_MODE=closedloop \
  bash orchestrate/eenergy/run_disagg_8gpu_battery.sh

OUTNAME=disagg_8gpu_gated_65pct CAP_PREFILL_W=382 CAP_DECODE_W=582 ARRIVAL_MODE=closedloop \
  bash orchestrate/eenergy/run_disagg_8gpu_battery.sh

echo "=== DISAGG_8GPU_FOLLOWUP_BATCHES_ALL_COMPLETE ==="
