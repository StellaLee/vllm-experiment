#!/bin/bash
# Sequential chain: cache-hit ShareGPT (light load) first, then throughput-matched
# whale-injection (moderate-heavy load) -- both test lmetric_power against its two
# predictions (collapses to ~lmetric under no pressure; meaningfully protects ramp under
# real pressure). Chained into one script so only one background task/wakeup cycle is
# needed instead of two.
bash orchestrate/eenergy/run_lmetric_power_cachehit.sh
echo "=== CACHEHIT LEG DONE, STARTING MATCHED LEG ==="
bash orchestrate/eenergy/run_lmetric_power_matched.sh
echo "=== BOTH LMETRICPOWER LEGS COMPLETE ==="
