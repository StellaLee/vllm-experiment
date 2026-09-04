#!/bin/bash
cd /root/pli/vllm-experiment
STAMP(){ date '+%Y-%m-%d %H:%M:%S'; }
LOG=logs/moe_tp2_conc20_status.log
echo "[$(STAMP)] MoE TP=2 test at higher concurrency (CONC=20, testing whether low activation needs more load to show ramp-reduction benefit), same widened whale setup, mono+chunk512, 3 trials" > $LOG
env MODEL=/data/pli/models/Qwen1.5-MoE-A2.7B-Chat GPUS=0,1 TAG=moe_wide_conc20 CONC=20 WHALE_MIN=18000 BUDGETS='16384 512' TRIALS='1 2 3' PORT=8073 ./orchestrate_pesim_gate_tp2.sh >> $LOG 2>&1
echo "[$(STAMP)] MoE TP=2 conc20 test done, exit=$?" >> $LOG
touch logs/moe_tp2_conc20_ALLDONE
