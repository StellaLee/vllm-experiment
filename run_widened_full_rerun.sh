#!/bin/bash
cd /root/pli/vllm-experiment
STAMP(){ date '+%Y-%m-%d %H:%M:%S'; }
LOG=logs/widened_full_rerun_status.log
TRIALS_10=$(seq -s' ' 1 10)
TRIALS_100=$(seq -s' ' 1 100)

echo "[$(STAMP)] Stage 1: CONC=10 widened, mono+chunk512, 100 trials (feeds reserve bootstrap)" > $LOG
env GPU=0 TAG=convdiverse_conc10_widened_gpu0 CONC=10 WHALE_MIN=18000 BUDGETS='16384 512' TRIALS="$TRIALS_100" PORT=8073 ./orchestrate_pesim_gate.sh >> $LOG 2>&1
echo "[$(STAMP)] Stage 1 done, exit=$?" >> $LOG

echo "[$(STAMP)] Stage 2: CONC=10 widened, chunk2048, 10 trials (single-chip observation only)" >> $LOG
env GPU=0 TAG=convdiverse_conc10_widened_gpu0 CONC=10 WHALE_MIN=18000 BUDGETS='2048' TRIALS="$TRIALS_10" PORT=8073 ./orchestrate_pesim_gate.sh >> $LOG 2>&1
echo "[$(STAMP)] Stage 2 done, exit=$?" >> $LOG

echo "[$(STAMP)] Stage 3: CONC=6 widened, all 3 budgets, 10 trials" >> $LOG
env GPU=0 TAG=convdiverse_conc6_widened_gpu0 CONC=6 WHALE_MIN=18000 BUDGETS='16384 2048 512' TRIALS="$TRIALS_10" PORT=8073 ./orchestrate_pesim_gate.sh >> $LOG 2>&1
echo "[$(STAMP)] Stage 3 done, exit=$?" >> $LOG

echo "[$(STAMP)] Stage 4: CONC=20 widened, all 3 budgets, 10 trials" >> $LOG
env GPU=0 TAG=convdiverse_conc20_widened_gpu0 CONC=20 WHALE_MIN=18000 BUDGETS='16384 2048 512' TRIALS="$TRIALS_10" PORT=8073 ./orchestrate_pesim_gate.sh >> $LOG 2>&1
echo "[$(STAMP)] Stage 4 done, exit=$?" >> $LOG

echo "[$(STAMP)] ALL STAGES COMPLETE" >> $LOG
touch logs/widened_full_rerun_ALLDONE
