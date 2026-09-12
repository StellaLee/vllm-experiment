#!/bin/bash
# Item 2 of the four-part paper-strengthening critique: "try 1 hour duration ... because in
# reality peak is usually measured at 15 min interval." Runs the flagship no-gate/92.5%-cap
# closed-loop pair (same cap as Table 2/Table 9, 544W prefill / 828W decode) for ~1 hour of
# real wall-clock duration each (not the flagship's original 200-380s), so the subsequent
# multi-window-length analysis (15s/1min/5min/15min) has enough data to say whether the
# reduction demonstrated at a 15s window survives being measured at something close to a real
# demand-charge billing interval.
#
# NUM_CONVS is sized from the flagship battery's own measured throughput (600 convs in 200.7s
# no-gate / 382.9s gated -- see analyze_disagg_8gpu_combined_power.py's duration check) to
# *naturally* finish around 60 minutes, deliberately NOT relying on the outer `timeout` to cut
# the run short: replay_sharegpt.py buffers all records in memory and only writes
# records.jsonl in one shot at the very end, so a SIGTERM mid-run would silently lose the
# entire records file (the per-pool power CSVs are unaffected, written incrementally by a
# separate process, but request-level accounting would be gone). HARNESS_TIMEOUT_S=5400 (90
# min) is a safety net only, well above the ~60 min each arm is expected to actually take.
set -x
cd /root/pli/vllm-experiment

OUTNAME=disagg_8gpu_1hr_baseline CAP_PREFILL_W= CAP_DECODE_W= ARRIVAL_MODE=closedloop \
  NUM_CONVS=10800 HARNESS_TIMEOUT_S=5400 \
  bash orchestrate/eenergy/run_disagg_8gpu_battery.sh

OUTNAME=disagg_8gpu_1hr_gated CAP_PREFILL_W=544 CAP_DECODE_W=828 ARRIVAL_MODE=closedloop \
  NUM_CONVS=5700 HARNESS_TIMEOUT_S=5400 \
  bash orchestrate/eenergy/run_disagg_8gpu_battery.sh

echo "=== DISAGG_8GPU_1HOUR_DURATION_COMPLETE ==="
