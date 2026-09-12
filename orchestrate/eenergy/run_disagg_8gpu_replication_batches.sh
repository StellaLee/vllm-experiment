#!/bin/bash
# Replication driver: adds trials 2-3 for the paper's two flagship comparisons (trial 1
# already on disk from earlier this session) -- closed-loop no-gate/92.5%-cap-gated, and
# Poisson no-gate/92.5%-cap-gated. Naming: trial 1 = base OUTNAME (already exists), trials
# 2-3 = base OUTNAME + _t2/_t3. Runs 8 battery invocations sequentially (share the same 8
# GPUs, can't parallelize).
set -x
cd /root/pli/vllm-experiment

for TRIAL in t2 t3; do
  OUTNAME=disagg_8gpu_baseline_${TRIAL} CAP_PREFILL_W= CAP_DECODE_W= ARRIVAL_MODE=closedloop \
    bash orchestrate/eenergy/run_disagg_8gpu_battery.sh
done

for TRIAL in t2 t3; do
  OUTNAME=disagg_8gpu_gated_${TRIAL} CAP_PREFILL_W=544 CAP_DECODE_W=828 ARRIVAL_MODE=closedloop \
    bash orchestrate/eenergy/run_disagg_8gpu_battery.sh
done

for TRIAL in t2 t3; do
  OUTNAME=disagg_8gpu_poisson_baseline_${TRIAL} CAP_PREFILL_W= CAP_DECODE_W= ARRIVAL_MODE=poisson \
    bash orchestrate/eenergy/run_disagg_8gpu_battery.sh
done

for TRIAL in t2 t3; do
  OUTNAME=disagg_8gpu_poisson_gated_${TRIAL} CAP_PREFILL_W=544 CAP_DECODE_W=828 ARRIVAL_MODE=poisson \
    bash orchestrate/eenergy/run_disagg_8gpu_battery.sh
done

echo "=== DISAGG_8GPU_REPLICATION_BATCHES_ALL_COMPLETE ==="
