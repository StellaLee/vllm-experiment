#!/bin/bash
cd /root/pli/vllm-experiment
COMMON='CONC=20 MAX_SEQS=48 MAXTOK=256 NCONV=200 PAD_MEAN=800 PAD_CV2=0.5 PAD_MIN=100 PAD_MAX=8000 TRIALS=1 WHALE_FRAC=0.15 GPU=0'

wait_marker(){ # tag
  local n="pesim_gate_$1"
  while [ ! -e "logs/${n}_ALLDONE" ] && [ ! -e "logs/${n}_FAILED" ]; do sleep 15; done
}

echo "[queue] waiting for meansweep_a26000 (already in flight)"
wait_marker meansweep_a26000

echo "[queue] launching meansweep_a34000"
env $COMMON TAG=meansweep_a34000 WHALE_MIN=34000 WHALE_MAX=50000 MAX_PROMPT_CHARS=50000 bash orchestrate_pesim_gate.sh > logs/meansweep_a34000.launch.log 2>&1
wait_marker meansweep_a34000

echo "[queue] launching meansweep_a40000"
env $COMMON TAG=meansweep_a40000 WHALE_MIN=40000 WHALE_MAX=50000 MAX_PROMPT_CHARS=50000 bash orchestrate_pesim_gate.sh > logs/meansweep_a40000.launch.log 2>&1
wait_marker meansweep_a40000

echo "[queue] launching varsweep_w16000"
env $COMMON TAG=varsweep_w16000 WHALE_MIN=26000 WHALE_MAX=42000 MAX_PROMPT_CHARS=42000 bash orchestrate_pesim_gate.sh > logs/varsweep_w16000.launch.log 2>&1
wait_marker varsweep_w16000

echo "[queue] launching varsweep_w8000"
env $COMMON TAG=varsweep_w8000 WHALE_MIN=30000 WHALE_MAX=38000 MAX_PROMPT_CHARS=38000 bash orchestrate_pesim_gate.sh > logs/varsweep_w8000.launch.log 2>&1
wait_marker varsweep_w8000

echo "[queue] ALL DONE" > logs/whale_sweep_queue_ALLDONE
