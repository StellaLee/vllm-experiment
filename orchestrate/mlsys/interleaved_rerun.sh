#!/bin/bash
cd /root/pli/vllm-experiment
STAMP(){ date '+%Y-%m-%d %H:%M:%S'; }
LOG=logs/interleaved_rerun_status.log
echo "[$(STAMP)] Interleaved rerun: CONC=10 and CONC=20 widened, mono+chunk512, batch-interleaved (10-trial batches alternating budget, 20 batches = 100 trials/arm each)" > $LOG

run_batch(){ # conc tag budget t0 t1
  local conc=$1 tag=$2 budget=$3 t0=$4 t1=$5
  local trials=$(seq -s' ' $t0 $t1)
  echo "[$(STAMP)] CONC=$conc budget=$budget trials=$t0-$t1" >> $LOG
  env GPU=0 TAG=$tag CONC=$conc WHALE_MIN=18000 BUDGETS="$budget" TRIALS="$trials" PORT=8073 ./orchestrate/pesim/orchestrate_pesim_gate.sh >> $LOG 2>&1
  local rc=$?
  echo "[$(STAMP)] CONC=$conc budget=$budget trials=$t0-$t1 done, exit=$rc" >> $LOG
}

echo "[$(STAMP)] === CONC=10 interleaved (10 alternating batches) ===" >> $LOG
for i in 0 1 2 3 4 5 6 7 8 9; do
  t0=$((i*10+1)); t1=$((i*10+10))
  run_batch 10 convdiverse_conc10_widened_interleaved_gpu0 16384 $t0 $t1
  run_batch 10 convdiverse_conc10_widened_interleaved_gpu0 512 $t0 $t1
done
touch logs/interleaved_conc10_ALLDONE
echo "[$(STAMP)] CONC=10 interleaved ALL DONE" >> $LOG

echo "[$(STAMP)] === CONC=20 interleaved (10 alternating batches) ===" >> $LOG
for i in 0 1 2 3 4 5 6 7 8 9; do
  t0=$((i*10+1)); t1=$((i*10+10))
  run_batch 20 convdiverse_conc20_widened_interleaved_gpu0 16384 $t0 $t1
  run_batch 20 convdiverse_conc20_widened_interleaved_gpu0 512 $t0 $t1
done
touch logs/interleaved_conc20_ALLDONE
echo "[$(STAMP)] CONC=20 interleaved ALL DONE" >> $LOG

echo "[$(STAMP)] ALL INTERLEAVED STAGES COMPLETE" >> $LOG
touch logs/interleaved_rerun_ALLDONE
