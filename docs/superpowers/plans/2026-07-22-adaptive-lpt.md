# Adaptive Per-Request Prefill-Length Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--long-prefill-token-threshold` adaptive at runtime (protect=512 when a long prompt
is prefilling/waiting, off=0 otherwise) while the step-wide budget stays static at mono (16384),
then run it as a new arm (`adaptivelpt`) against the existing 8-arm lengthgate/threshold grid.

**Architecture:** A single-anchor hotpatch inserts a per-step gate into `Scheduler.schedule()`,
right after the unconditional `token_budget = self.max_num_scheduled_tokens` line, that mutates
`self.scheduler_config.long_prefill_token_threshold` directly. Both existing read sites in
`scheduler.py` already read that attribute fresh every step with no caching, so no other code
changes are needed. The gate is independent of `ChunkSizeController`/`DYNAMIC_CHUNK` — it runs
whether or not the chunk controller is active.

**Tech Stack:** Python 3.10, vLLM 0.23.0 (V1 scheduler), bash orchestration scripts, box
`183.147.142.123` (`/root/pli/vllm-experiment`, venv `/root/pli/venv-vllm023`).

## Global Constraints

- Anchor line (verbatim, confirmed against the live box file): `        token_budget = self.max_num_scheduled_tokens\n` in `vllm/v1/core/sched/scheduler.py` (8-space indent — inside `Scheduler.schedule()`).
- Env defaults: `ADAPTIVE_LPT_GATE=4096`, `ADAPTIVE_LPT_PROTECT=512`, `ADAPTIVE_LPT_OFF=0` — reuse lengthgate's already-validated gate/protect values verbatim; do not retune them in this plan.
- The patch must not touch `ChunkSizeController` or any `DYNAMIC_CHUNK`/`CHUNK_MODE` code path.
- Workload for the experiment arm must be byte-for-byte identical to every other arm this session: `--phase-schedule "6:0.0@60,1.0:0.2@60" --duration 360`, `--max-tokens 256`, `--pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000`, `--whale-min-chars 44000 --whale-max-chars 50000`, `--max-prompt-chars 50000`, `--pad-seed 1001`. Server: `--max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90`.
- `logs/*-lgate-b{arm}-t1.jsonl` naming convention (required by `scripts/analyze_lengthgate.py`'s `load()` glob) must be preserved for the new arm's output file.

---

### Task 1: Adaptive-LPT hotpatch + test

**Files:**
- Create: `scripts/hotpatch_adaptive_lpt.py`
- Create: `tests/test_hotpatch_adaptive_lpt.py`

**Interfaces:**
- Consumes: nothing from other tasks (reuses the `VLLM_SCHED_PATH` env-var override pattern already established in `scripts/hotpatch_lengthgate.py`, so tests never touch the real vLLM install).
- Produces: `scripts/hotpatch_adaptive_lpt.py`'s `main() -> int` (0 = success/no-op-if-already-patched, 1 = anchor not found), invoked as `python scripts/hotpatch_adaptive_lpt.py` by Task 2's orchestration script.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hotpatch_adaptive_lpt.py`:

```python
#!/usr/bin/env python3
"""Offline test of the adaptive-lpt patcher: apply to a scheduler FIXTURE (the anchor line the
real scheduler contains), verify the gate block lands right after it and re-applying is a no-op.
No vLLM/box needed."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import hotpatch_adaptive_lpt as H

FIXTURE = '''\
class Scheduler(SchedulerInterface):
    def schedule(self):
        token_budget = self.max_num_scheduled_tokens

        if self._chunk_ctrl is not None:
            _decode_depth = sum(
                1 for r in self.running if not r.is_prefill_chunk
            )
            token_budget = self._chunk_ctrl.step(
                _decode_depth, getattr(self, "_ff_last_tokens", None))
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0
'''


def _apply(src):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scheduler.py")
    open(p, "w").write(src)
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    return rc, open(p).read()


def test_patch_applies_gate_block():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert 'if os.getenv("ADAPTIVE_LPT"):' in out
    assert 'self.scheduler_config.long_prefill_token_threshold = _alpt_thr' in out
    idx_anchor = out.index("token_budget = self.max_num_scheduled_tokens")
    idx_gate = out.index('if os.getenv("ADAPTIVE_LPT"):')
    idx_chunk_ctrl = out.index("if self._chunk_ctrl is not None:")
    assert idx_anchor < idx_gate < idx_chunk_ctrl


def test_patch_idempotent():
    rc1, out1 = _apply(FIXTURE)
    open(os.environ["VLLM_SCHED_PATH"], "w").write(out1)
    rc2 = H.main()                       # second apply on already-patched file
    assert rc2 == 0
    assert open(os.environ["VLLM_SCHED_PATH"]).read() == out1   # no further change


def test_patch_missing_anchor_errors():
    rc, out = _apply("class Scheduler(SchedulerInterface):\n    def schedule(self):\n        pass\n")
    assert rc == 1


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print(f"ok {f.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_hotpatch_adaptive_lpt.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'hotpatch_adaptive_lpt'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/hotpatch_adaptive_lpt.py`:

```python
#!/usr/bin/env python3
"""Make --long-prefill-token-threshold adaptive: off (0) by default, protect (512) whenever a
long prompt is prefilling or waiting. Independent of ChunkSizeController/DYNAMIC_CHUNK -- the
step-wide budget is never touched, only scheduler_config.long_prefill_token_threshold, which both
existing read sites (scheduler.py running-loop line ~736, waiting-loop line ~1032) read fresh
every step with no caching.

Idempotent. One edit: insert a per-step gate block right after the unconditional
`token_budget = self.max_num_scheduled_tokens` line, so it runs every step regardless of whether
DYNAMIC_CHUNK/_chunk_ctrl is active.

Env: ADAPTIVE_LPT (set to enable), ADAPTIVE_LPT_GATE (tok, default 4096),
ADAPTIVE_LPT_PROTECT (512), ADAPTIVE_LPT_OFF (0), ADAPTIVE_LPT_TRACE (optional CSV path:
wall_s,pf_remaining,threshold per step).
"""
import os
import sys
from pathlib import Path


def _find_sched() -> Path:
    override = os.environ.get("VLLM_SCHED_PATH")
    if override:
        return Path(override)
    import vllm  # noqa: PLC0415
    return Path(vllm.__file__).parent / "v1" / "core" / "sched" / "scheduler.py"


ANCHOR = "        token_budget = self.max_num_scheduled_tokens\n"
NEW = ANCHOR + '''\
        if os.getenv("ADAPTIVE_LPT"):
            _alpt_pf = 0
            for _r in self.running:
                if getattr(_r, 'is_prefill_chunk', False):
                    _alpt_pf = max(_alpt_pf, _r.num_prompt_tokens - _r.num_computed_tokens)
            for _r in self.waiting:
                _alpt_pf = max(_alpt_pf, _r.num_prompt_tokens - _r.num_computed_tokens)
            _alpt_gate = int(os.getenv("ADAPTIVE_LPT_GATE", "4096"))
            _alpt_protect = int(os.getenv("ADAPTIVE_LPT_PROTECT", "512"))
            _alpt_off = int(os.getenv("ADAPTIVE_LPT_OFF", "0"))
            _alpt_thr = _alpt_protect if _alpt_pf > _alpt_gate else _alpt_off
            self.scheduler_config.long_prefill_token_threshold = _alpt_thr
            _alpt_trace = os.getenv("ADAPTIVE_LPT_TRACE")
            if _alpt_trace:
                import time as _alpt_time
                with open(_alpt_trace, "a") as _alpt_f:
                    _alpt_f.write(f"{_alpt_time.monotonic()},{_alpt_pf},{_alpt_thr}\\n")
'''


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()
    if "ADAPTIVE_LPT" in src:
        print("adaptive-lpt controller already present -- no changes.")
        return 0
    if ANCHOR not in src:
        print("ERROR: token_budget anchor not found", file=sys.stderr)
        return 1
    src = src.replace(ANCHOR, NEW, 1)
    sched.write_text(src)
    print(f"adaptive-lpt controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/test_hotpatch_adaptive_lpt.py`
Expected: `ok test_patch_applies_gate_block`, `ok test_patch_idempotent`,
`ok test_patch_missing_anchor_errors`, `3 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/hotpatch_adaptive_lpt.py tests/test_hotpatch_adaptive_lpt.py
git commit -m "feat: add adaptive long-prefill-token-threshold hotpatch"
```

---

### Task 2: Orchestration script for the `adaptivelpt` arm

**Files:**
- Create: `run_adaptive_lpt_arm.sh`

**Interfaces:**
- Consumes: `scripts/hotpatch_adaptive_lpt.py` (Task 1) via `python scripts/hotpatch_adaptive_lpt.py`; `scripts/analyze_lengthgate.py` (pre-existing, unmodified — its `ARMS` env var and `load()` glob already generalize to any arm label).
- Produces: `logs/2026-07-22-lgate-badaptivelpt-t1.jsonl` (client records), `logs/adaptivelpt-chunktrace.csv` (gate trace, via `ADAPTIVE_LPT_TRACE`), `logs/lgate_ANALYSIS_v5.txt` (final 9-arm table), markers `logs/lgate_adaptivelpt_ALLDONE` / `_FAILED`.

This task is an operational shell script (matches the project's established pattern for
`rerun_lpt_arm.sh`/`rerun_lpt_grid.sh` — these are not unit-tested; they're validated by running
them against the live box). Step 2's "test" is a local syntax check; the real validation is the
live run in Task 3.

- [ ] **Step 1: Write the script**

Create `run_adaptive_lpt_arm.sh`:

```bash
#!/bin/bash
# run_adaptive_lpt_arm.sh -- new arm: adaptive long-prefill-token-threshold (off=0 when calm,
# protect=512 when a long prompt is prefilling/waiting), step-wide budget static at mono (16384).
# Same phase-schedule workload as every other lgate arm. Re-analyzes all 9 arms into
# logs/lgate_ANALYSIS_v5.txt.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
DATE=$(date +%Y-%m-%d)
SCHED="6:0.0@60,1.0:0.2@60"; DUR=360
GATE=4096; PROTECT=512; OFF=0
PORT=8050
ARM=adaptivelpt
FB="logs/${DATE}-lgate-b${ARM}"

rm -f logs/lgate_adaptivelpt_ALLDONE logs/lgate_adaptivelpt_FAILED "${FB}-t1.jsonl" "${FB}-chunktrace.csv"

$PYTHON scripts/hotpatch_adaptive_lpt.py || { log "PATCH FAILED"; touch logs/lgate_adaptivelpt_FAILED; exit 1; }

log "arm=$ARM budget=16384 gate=$GATE protect=$PROTECT off=$OFF SCHED='$SCHED' DUR=${DUR}s"
env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
  ADAPTIVE_LPT=1 ADAPTIVE_LPT_GATE=$GATE ADAPTIVE_LPT_PROTECT=$PROTECT ADAPTIVE_LPT_OFF=$OFF \
  ADAPTIVE_LPT_TRACE="${FB}-chunktrace.csv" \
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
  --max-num-seqs 128 --max-num-batched-tokens 16384 --max-model-len 16384 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
SV=$!
for i in $(seq 1 120); do sleep 5
  grep -q "Application startup complete" ${FB}-server.log && break
  [ "$i" = 120 ] && { log "SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours; touch logs/lgate_adaptivelpt_FAILED; exit 1; }
done
$PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
  --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens 256 \
  --phase-schedule "$SCHED" --duration $DUR \
  --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
  --whale-min-chars 44000 --whale-max-chars 50000 \
  --max-prompt-chars 50000 --pad-seed 1001 \
  --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
log "[$ARM] recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours

log "re-analyzing all 9 arms"
SCHEDULE="$SCHED" \
  ARMS="16384 512 2048 lengthgate 16384lpt512 16384lpt256 16384lpt2048 2048lpt512 ${ARM}" \
  SLO_TBT_MS=500 $PYTHON scripts/analyze_lengthgate.py > logs/lgate_ANALYSIS_v5.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/lgate_ANALYSIS_v5.txt
touch logs/lgate_adaptivelpt_ALLDONE
log "done -> logs/lgate_ANALYSIS_v5.txt"
```

- [ ] **Step 2: Syntax-check the script**

Run: `bash -n run_adaptive_lpt_arm.sh`
Expected: no output (exit 0 = valid syntax)

- [ ] **Step 3: Commit**

```bash
git add run_adaptive_lpt_arm.sh
git commit -m "feat: add adaptivelpt arm orchestration script"
```

---

### Task 3: Run the experiment and review the trace

Not a code task — the live validation step. Do this after Tasks 1-2 are committed.

- [ ] **Step 1: Confirm the box GPUs are idle**

```bash
ssh root@183.147.142.123 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
```
Expected: all rows `1 MiB, 0 %` (or confirm any non-idle GPUs are not 0/1, the pair this script uses).

- [ ] **Step 2: Sync the two new files and launch**

```bash
rsync -avz scripts/hotpatch_adaptive_lpt.py root@183.147.142.123:/root/pli/vllm-experiment/scripts/hotpatch_adaptive_lpt.py
rsync -avz run_adaptive_lpt_arm.sh root@183.147.142.123:/root/pli/vllm-experiment/run_adaptive_lpt_arm.sh
ssh root@183.147.142.123 "cd /root/pli/vllm-experiment && chmod +x run_adaptive_lpt_arm.sh && nohup bash run_adaptive_lpt_arm.sh > logs/lgate_adaptivelpt_run.log 2>&1 & disown"
```

- [ ] **Step 3: Wait for `logs/lgate_adaptivelpt_ALLDONE`, then review the trace BEFORE reading the analysis table**

```bash
ssh root@183.147.142.123 "cd /root/pli/vllm-experiment && python3 -c \"
import csv
rows = [r for r in csv.reader(open('logs/2026-07-22-lgate-badaptivelpt-chunktrace.csv'))]
print('total steps:', len(rows))
protect_steps = [r for r in rows if r[2] == '512']
print('protect-mode steps:', len(protect_steps))
# check: any protect-mode step where pf_remaining was actually 0 (would indicate a gate bug)?
bad = [r for r in protect_steps if float(r[1]) <= 4096]
print('protect steps with pf_remaining <= gate (should be 0):', len(bad))
\""
```

Expected: `bad` count is 0. If not, this is the same class of bug lengthgate had (bug 1) — do not
proceed to reporting results until root-caused, per the mandatory-trace-review non-goal in the
design spec.

- [ ] **Step 4: Read the analysis table**

```bash
ssh root@183.147.142.123 "cat /root/pli/vllm-experiment/logs/lgate_ANALYSIS_v5.txt"
```

Compare the `adaptivelpt` row's S:TTFT against mono (250) and `16384lpt512` (263), and its
W:max/W:goodput against `16384lpt512`'s (490.4 / 100.0%), per the design spec's success criteria.

- [ ] **Step 5: Update findings**

Append the result to `findings/2026-07-22-lengthgate-and-per-request-cap.md` (same file the rest
of this line of work lives in), following its existing table/section conventions.

---

## Self-Review Notes

- **Spec coverage:** Mechanism (Task 1), components 1-2-4 from the spec's Components list (Task 1
  + Task 2), the trace-review requirement (Task 3 Step 3), setup/success-criteria (Task 3 Steps
  2-4), findings update (Task 3 Step 5). Spec's "Components" item 3 (`ADAPTIVE_LPT_TRACE`) is
  folded into Task 1's single code block rather than a separate task, since it's a few lines
  inside the same anchor block, not an independently testable unit.
- **Placeholder scan:** no TBD/TODO; all steps have complete code or exact commands with expected
  output.
- **Type consistency:** `hotpatch_adaptive_lpt.main() -> int` matches `hotpatch_lengthgate.py`'s
  existing convention (0/1 return); env var names (`ADAPTIVE_LPT`, `ADAPTIVE_LPT_GATE`,
  `ADAPTIVE_LPT_PROTECT`, `ADAPTIVE_LPT_OFF`, `ADAPTIVE_LPT_TRACE`) are identical across the spec,
  Task 1's implementation, and Task 2's launch command.
