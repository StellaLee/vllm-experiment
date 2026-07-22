#!/bin/bash
# pilot_lengthgate_rates.sh -- calibrate Phase-S / Phase-W rates BEFORE the 4-arm run.
# Goal: pick rate_S so the SHORT phase is prefill-bound (static-512 visibly pays TTFT vs static-2048)
# and rate_W so the WHALE phase stays sub-saturation. Runs static-512 and static-2048 servers over a
# candidate phase schedule and prints per-phase TTFT + a saturation flag. No dynamic arm here.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
DATE=$(date +%Y-%m-%d); PORT=8050
SCHED=${SCHED:-"12:0.0@45,3:0.2@45"}; DUR=${DUR:-180}
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 5; }
rm -f logs/lgatepilot_ALLDONE

run_static(){ # $1=budget
  local B=$1 FB="logs/${DATE}-lgatepilot-b${B}"
  echo ">>> static budget=$B  schedule=$SCHED"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-num-batched-tokens $B --max-model-len 16384 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  for i in $(seq 1 120); do sleep 5; grep -q "Application startup complete" ${FB}-server.log && break
    [ "$i" = 120 ] && { echo "  SERVER TIMEOUT (b=$B)"; kill $SV 2>/dev/null; kill_ours; return; }; done
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs 4000 --max-turns 1 --min-turns 1 --max-tokens 256 \
    --phase-schedule "$SCHED" --duration $DUR \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-frac 0.0 --whale-min-chars 44000 --whale-max-chars 50000 \
    --max-prompt-chars 50000 --pad-seed 1001 \
    --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
  local PREE=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)
  kill $SV 2>/dev/null; sleep 6; kill -9 $SV 2>/dev/null; kill_ours
  SCHEDULE="$SCHED" ARMS="$B" $PYTHON - "$B" <<'PY'
import os, sys, glob, json, statistics as st
sys.path.insert(0, "scripts"); from replay_timing import parse_phase_schedule, phase_type_at
B=sys.argv[1]; sched=parse_phase_schedule(os.environ["SCHEDULE"])
recs=[]
for f in glob.glob(f"logs/*-lgatepilot-b{B}-t1.jsonl"): recs+=[json.loads(l) for l in open(f) if l.strip()]
if not recs: print(f"  b={B}: no records"); raise SystemExit
t0=min(r["ts"]-r["latency"] for r in recs)
def ptype(t): return "S" if phase_type_at(sched, t)[2]==0.0 else "W"
S=[r["ttft"] for r in recs if r.get("ttft") is not None and ptype((r["ts"]-r["latency"])-t0)=="S"]
W=[r["ttft"] for r in recs if r.get("ttft") is not None and ptype((r["ts"]-r["latency"])-t0)=="W"]
def m(x): return 1000*st.mean(x) if x else float("nan")
sat="SATURATED?" if (m(S)>8000 or m(W)>8000) else "ok"
print(f"  b={B}:  S:TTFTmean={m(S):.0f}ms(n={len(S)})  W:TTFTmean={m(W):.0f}ms(n={len(W)})  [{sat}]")
PY
}

run_static 2048
run_static 512
echo "PICK: rate_S high enough that b=512 S:TTFT >> b=2048 S:TTFT (prefill-bound), both phases [ok]."
touch logs/lgatepilot_ALLDONE
echo "LGATE PILOT DONE"
