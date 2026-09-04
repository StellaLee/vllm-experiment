#!/bin/bash
cd /root/pli/vllm-experiment
STAMP(){ date '+%Y-%m-%d %H:%M:%S'; }
LOG=logs/interleaved_conc6_status.log
echo "[$(STAMP)] CONC=6 widened interleaved: mono+chunk512, 100 trials (10 alternating 10-trial batches), for reserve-procurement prep" > $LOG

run_batch(){ # budget t0 t1
  local budget=$1 t0=$2 t1=$3
  local trials=$(seq -s' ' $t0 $t1)
  echo "[$(STAMP)] budget=$budget trials=$t0-$t1" >> $LOG
  env GPU=0 TAG=convdiverse_conc6_widened_interleaved_gpu0 CONC=6 WHALE_MIN=18000 BUDGETS="$budget" TRIALS="$trials" PORT=8073 ./orchestrate/pesim/orchestrate_pesim_gate.sh >> $LOG 2>&1
  local rc=$?
  echo "[$(STAMP)] budget=$budget trials=$t0-$t1 done, exit=$rc" >> $LOG
}

for i in 0 1 2 3 4 5 6 7 8 9; do
  t0=$((i*10+1)); t1=$((i*10+10))
  run_batch 16384 $t0 $t1
  run_batch 512 $t0 $t1
done

echo "[$(STAMP)] CONC=6 interleaved ALL DONE" >> $LOG
touch logs/interleaved_conc6_ALLDONE
