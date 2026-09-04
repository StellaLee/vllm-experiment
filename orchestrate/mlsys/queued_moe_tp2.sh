#!/bin/bash
cd /root/pli/vllm-experiment
STAMP(){ date '+%Y-%m-%d %H:%M:%S'; }
LOG=logs/queued_moe_tp2_status.log
echo "[$(STAMP)] waiting for interleaved_rerun (CONC=10/20 reserve libraries) to finish..." > $LOG

while [ ! -f logs/interleaved_rerun_ALLDONE ]; do
  sleep 60
done

echo "[$(STAMP)] interleaved_rerun done -- launching MoE TP=2 test (Qwen1.5-MoE-A2.7B-Chat), same setup as pesim_gate_tp2_14b_wide: GPUs 0,1 conc=4 widened whale, mono+chunk512, 3 trials" >> $LOG
env MODEL=/data/pli/models/Qwen1.5-MoE-A2.7B-Chat GPUS=0,1 TAG=moe_wide CONC=4 WHALE_MIN=18000 BUDGETS='16384 512' TRIALS='1 2 3' PORT=8073 ./orchestrate/pesim/orchestrate_pesim_gate_tp2.sh >> $LOG 2>&1
echo "[$(STAMP)] MoE TP=2 test done, exit=$?" >> $LOG
touch logs/queued_moe_tp2_ALLDONE
