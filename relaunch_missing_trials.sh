#!/bin/bash
cd /root/pli/vllm-experiment
STAMP(){ date '+%Y-%m-%d %H:%M:%S'; }
LOG=logs/relaunch_missing_trials_status.log
MISSING=$(seq -s' ' 2 10)

echo "[$(STAMP)] Fill A: CONC=10 widened, chunk2048, trials 2-10 (9 more, completes single-chip observation)" > $LOG
env GPU=0 TAG=convdiverse_conc10_widened_gpu0 CONC=10 WHALE_MIN=18000 BUDGETS='2048' TRIALS="$MISSING" PORT=8073 ./orchestrate_pesim_gate.sh >> $LOG 2>&1
echo "[$(STAMP)] Fill A done, exit=$?" >> $LOG

echo "[$(STAMP)] Fill B: CONC=6 widened, all 3 budgets, trials 2-10 (9 more each)" >> $LOG
env GPU=0 TAG=convdiverse_conc6_widened_gpu0 CONC=6 WHALE_MIN=18000 BUDGETS='16384 2048 512' TRIALS="$MISSING" PORT=8073 ./orchestrate_pesim_gate.sh >> $LOG 2>&1
echo "[$(STAMP)] Fill B done, exit=$?" >> $LOG

echo "[$(STAMP)] Fill C: CONC=20 widened, all 3 budgets, trials 2-10 (9 more each, also completes reserve library to 100 for mono/512)" >> $LOG
env GPU=0 TAG=convdiverse_conc20_widened_gpu0 CONC=20 WHALE_MIN=18000 BUDGETS='16384 2048 512' TRIALS="$MISSING" PORT=8073 ./orchestrate_pesim_gate.sh >> $LOG 2>&1
echo "[$(STAMP)] Fill C done, exit=$?" >> $LOG

echo "[$(STAMP)] ALL FILL STAGES COMPLETE" >> $LOG
touch logs/relaunch_missing_trials_ALLDONE
