#!/bin/bash
cd /root/pli/vllm-experiment
rm -f logs/AUTONOMOUS_TASKAB_ALLDONE logs/AUTONOMOUS_TASKAB_STATUS.txt
STAMP(){ date '+%Y-%m-%d %H:%M:%S'; }
status(){ echo "[$(STAMP)] $1" | tee -a logs/AUTONOMOUS_TASKAB_STATUS.txt; }

status '=== AUTONOMOUS TASK A+B RUN STARTED ==='

status '[1/7] Task A: CONC=6 diverse-conv (mono+chunk, 20 trials/arm)'
env GPU=0 TAG=convdiverse_conc6_gpu0 CONC=6 BUDGETS='16384 512' TRIALS='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20' PORT=8073 ./orchestrate/pesim/orchestrate_pesim_gate.sh
status "[1/7] done exit=$?"

status '[2/7] Task A: CONC=20 diverse-conv (mono+chunk, 20 trials/arm)'
env GPU=0 TAG=convdiverse_conc20_gpu0 CONC=20 BUDGETS='16384 512' TRIALS='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20' PORT=8073 ./orchestrate/pesim/orchestrate_pesim_gate.sh
status "[2/7] done exit=$?"

status '[3/7] Task B: whale grid frac=5% size=8k (10 trials/arm)'
env GPU=0 TAG=whalegrid_f05_s8k CONC=10 BUDGETS='16384 512' WHALE_FRAC=0.05 WHALE_MIN=24000 WHALE_MAX=30000 TRIALS='1 2 3 4 5 6 7 8 9 10' PORT=8073 ./orchestrate/pesim/orchestrate_pesim_gate.sh
status "[3/7] done exit=$?"

status '[4/7] Task B: whale grid frac=5% size=14k (10 trials/arm)'
env GPU=0 TAG=whalegrid_f05_s14k CONC=10 BUDGETS='16384 512' WHALE_FRAC=0.05 WHALE_MIN=44000 WHALE_MAX=50000 TRIALS='1 2 3 4 5 6 7 8 9 10' PORT=8073 ./orchestrate/pesim/orchestrate_pesim_gate.sh
status "[4/7] done exit=$?"

status '[5/7] Task B: whale grid frac=15% size=8k (10 trials/arm)'
env GPU=0 TAG=whalegrid_f15_s8k CONC=10 BUDGETS='16384 512' WHALE_FRAC=0.15 WHALE_MIN=24000 WHALE_MAX=30000 TRIALS='1 2 3 4 5 6 7 8 9 10' PORT=8073 ./orchestrate/pesim/orchestrate_pesim_gate.sh
status "[5/7] done exit=$?"

status '[6/7] Task B: whale grid frac=30% size=8k (10 trials/arm)'
env GPU=0 TAG=whalegrid_f30_s8k CONC=10 BUDGETS='16384 512' WHALE_FRAC=0.30 WHALE_MIN=24000 WHALE_MAX=30000 TRIALS='1 2 3 4 5 6 7 8 9 10' PORT=8073 ./orchestrate/pesim/orchestrate_pesim_gate.sh
status "[6/7] done exit=$?"

status '[7/7] Task B: whale grid frac=30% size=14k (10 trials/arm)'
env GPU=0 TAG=whalegrid_f30_s14k CONC=10 BUDGETS='16384 512' WHALE_FRAC=0.30 WHALE_MIN=44000 WHALE_MAX=50000 TRIALS='1 2 3 4 5 6 7 8 9 10' PORT=8073 ./orchestrate/pesim/orchestrate_pesim_gate.sh
status "[7/7] done exit=$?"

status '=== AUTONOMOUS TASK A+B RUN FINISHED ==='
touch logs/AUTONOMOUS_TASKAB_ALLDONE
