#!/bin/bash
# orchestrate_burstgpt_one.sh -- ONE-TRIAL initial reading of the 3-arm comparison on a real
# BurstGPT trace window. Slices a dense conv-log window, CAPS inter-arrival gaps (so we don't wait
# out the trace's sparse periods), rebases timestamps, and scales to a target wall-clock runtime.
# Drives mono/chunk/ours CONCURRENTLY on 3 GPU pairs against vLLM's NATIVE api_server (/generate,
# which burstgpt-bench speaks) -- same engine flags + scheduler patch as the synthetic sweep.
# Markers: logs/burstgpt_one_ALLDONE|FAILED.  Tunables via env: OFF NROWS GAPCAP TARGET_S.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
export PYTHONPATH=/root/pli/BurstGPT/example/src
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
BG=/root/pli/BurstGPT; DATA=$BG/example/preprocess_data/shareGPT.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/burstgpt_one_ALLDONE logs/burstgpt_one_FAILED
OFF=${OFF:-75000}; NROWS=${NROWS:-400}; GAPCAP=${GAPCAP:-3.0}; TARGET_S=${TARGET_S:-420}
WIN=logs/burstgpt_win.csv

log "slicing conv window off=$OFF n=$NROWS gapcap=${GAPCAP}s target=${TARGET_S}s"
$PYTHON - "$BG/data/BurstGPT_1.csv" "$WIN" "$OFF" "$NROWS" "$GAPCAP" "$TARGET_S" logs/burstgpt_scale.txt logs/burstgpt_n.txt <<'PY'
import csv, statistics as st, sys
src,win,off,n,gapcap,target,scaleout,nout = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7], sys.argv[8]
rows=[r for r in csv.reader(open(src))]; hdr=rows[0]
# conv-log rows with a NON-ZERO response (max_tokens=0 -> vLLM 500s)
conv=[r for r in rows[1:] if r[5]=="Conversation log" and int(r[3])>=1]
W=conv[off:off+n]; ts=[int(r[0]) for r in W]
capped=[0.0]
for i in range(1,len(ts)): capped.append(capped[-1]+min(ts[i]-ts[i-1], gapcap))
span=capped[-1] or 1.0
with open(win,"w",newline="") as f:
    w=csv.writer(f); w.writerow(hdr)
    for r,t in zip(W,capped): w.writerow([int(round(t))]+r[1:])
scale=span/target
open(scaleout,"w").write(f"{scale:.5f}"); open(nout,"w").write(str(len(W)))
req=[int(r[2]) for r in W]; m=st.mean(req); cs2=st.pvariance(req)/(m*m)
resp=[int(r[3]) for r in W]
sys.stderr.write(f"[slice] n={len(W)} capped_span={span:.0f}s scale={scale:.3f} "
                 f"realized_Cs2={cs2:.3f} mean_req={m:.0f} max_req={max(req)} "
                 f"mean_resp={st.mean(resp):.0f} p50_resp={st.median(resp):.0f}\n")
PY
SCALE=$(cat logs/burstgpt_scale.txt); NROWS=$(cat logs/burstgpt_n.txt)
log "scale=$SCALE  n=$NROWS  (window -> ~${TARGET_S}s wall clock)"

run_arm(){ # gpus port arm budget mode
  local gpus=$1 port=$2 arm=$3 budget=$4 mode=$5
  local EX="DYNAMIC_CHUNK=0"
  [ "$mode" = slocvar ] && EX="DYNAMIC_CHUNK=1 CHUNK_MODE=slocvar DYNAMIC_CHUNK_MIN=512 DYNAMIC_CHUNK_SLO_MS=50 DYNAMIC_CHUNK_CVAR_PCTL=90 DYNAMIC_CHUNK_START=512 DYNAMIC_CHUNK_TRACE=logs/burstgpt_one-${arm}-chunktrace.csv"
  log "  [$arm] server gpus=$gpus port=$port budget=$budget mode=$mode"
  env CUDA_VISIBLE_DEVICES=$gpus $EX $PYTHON -m vllm.entrypoints.api_server \
      --model "$MODEL" --port $port --max-num-seqs 32 --max-num-batched-tokens $budget \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/burstgpt_one-${arm}-server.log 2>&1 &
  local sv=$!
  for i in $(seq 1 100); do sleep 5
    grep -qiE "Application startup complete|Uvicorn running|Started server process" logs/burstgpt_one-${arm}-server.log && break
    [ "$i" = 100 ] && { log "  [$arm] SERVER TIMEOUT"; kill -9 $sv 2>/dev/null; return 1; }
  done
  timeout 1200 env PYTHONPATH=$PYTHONPATH $PYTHON -m burstgpt.cli --backend vllm --host localhost --port $port \
      --use_burstgpt --burstgpt_path "$WIN" --conv_or_api conv --prompt_num $NROWS --surplus_prompts_num $NROWS \
      --scale $SCALE --stream --ignore_eos --max_gen_len 1024 --data_path "$DATA" \
      --detail_log_path logs/burstgpt_one-${arm}-detail.jsonl --log_path logs/burstgpt_one-${arm}-sum.jsonl \
      > logs/burstgpt_one-${arm}-client.log 2>&1
  local rc=$?
  log "  [$arm] client done rc=$rc recs=$(grep -c . logs/burstgpt_one-${arm}-detail.jsonl 2>/dev/null || echo 0)"
  kill "$sv" 2>/dev/null; sleep 10; kill -9 "$sv" 2>/dev/null  # SIGTERM first so vLLM tears down its TP workers (SIGKILL orphans them)
  return $rc
}

log "launching 3 arms concurrently (mono@0-1 chunk@2-3 ours@4-5)"
run_arm 0,1 8021 mono  16384 static  & P1=$!
run_arm 2,3 8022 chunk 512   static  & P2=$!
run_arm 4,5 8023 ours  16384 slocvar & P3=$!
RC=0; wait $P1||RC=1; wait $P2||RC=1; wait $P3||RC=1
kill_ours
log "analyzing"
$PYTHON scripts/analyze_burstgpt.py > logs/burstgpt_one_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE (rc=$RC)" >> logs/burstgpt_one_ANALYSIS.txt
touch logs/burstgpt_one_ALLDONE
log "done -> logs/burstgpt_one_ANALYSIS.txt"
