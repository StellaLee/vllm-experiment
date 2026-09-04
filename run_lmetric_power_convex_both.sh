#!/bin/bash
bash run_lmetric_power_convex_cachehit.sh
echo "=== CACHEHIT LEG DONE, STARTING MATCHED LEG ==="
bash run_lmetric_power_convex_matched.sh
echo "=== BOTH LMETRICPOWERCONVEX LEGS COMPLETE ==="
