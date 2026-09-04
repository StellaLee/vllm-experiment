#!/bin/bash
cd /root/pli/vllm-experiment
STAMP(){ date '+%Y-%m-%d %H:%M:%S'; }
LOG=logs/queued_conc20_growth_status.log
echo "[$(STAMP)] waiting for widened_full_rerun (stages 1-4) to finish..." > $LOG

while [ ! -f logs/widened_full_rerun_ALLDONE ]; do
  sleep 60
done

echo "[$(STAMP)] stages 1-4 done -- launching Stage 5: CONC=20 widened, mono+chunk512, trials 11-100 (90 more, to grow the reserve-procurement library to 100)" >> $LOG
TRIALS_11_100=$(seq -s' ' 11 100)
env GPU=0 TAG=convdiverse_conc20_widened_gpu0 CONC=20 WHALE_MIN=18000 BUDGETS='16384 512' TRIALS="$TRIALS_11_100" PORT=8073 ./orchestrate_pesim_gate.sh >> $LOG 2>&1
echo "[$(STAMP)] Stage 5 done, exit=$?" >> $LOG
touch logs/queued_conc20_growth_ALLDONE
