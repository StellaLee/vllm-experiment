#!/bin/bash
bash run_whale_argmin_power_switch_cachehit.sh
echo "=== CACHEHIT LEG DONE, STARTING MATCHED LEG ==="
bash run_whale_argmin_power_switch_matched.sh
echo "=== BOTH WHALEARGMINPOWERSWITCH LEGS COMPLETE ==="
