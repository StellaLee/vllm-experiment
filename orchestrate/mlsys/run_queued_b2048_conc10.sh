#!/bin/bash
cd /root/pli/vllm-experiment
STAMP(){ date '+%Y-%m-%d %H:%M:%S'; }
LOG=logs/queued_b2048_conc10_status.log
echo "[$(STAMP)] waiting for convdiverse_conc10_gpu0 100-trial growth job (b16384+b512) to finish..." > $LOG

while [ ! -f logs/pesim_gate_convdiverse_conc10_gpu0_ALLDONE ]; do
  sleep 60
done

echo "[$(STAMP)] growth job done -- launching budget=2048, CONC=10, diverse-conv, 20 trials" >> $LOG
env GPU=0 TAG=convdiverse_conc10_gpu0 CONC=10 BUDGETS='2048' TRIALS='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20' PORT=8073 ./orchestrate/pesim/orchestrate_pesim_gate.sh >> $LOG 2>&1

echo "[$(STAMP)] budget=2048 run finished, exit=$?" >> $LOG
