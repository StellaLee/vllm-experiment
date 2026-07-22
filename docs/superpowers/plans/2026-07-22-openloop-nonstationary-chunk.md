# Open-loop regime check + non-stationary dynamic-chunking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the harness + analysis to (E1) re-check chunk's decode-protection win under open-loop Poisson arrivals and (E2) demonstrate a *moving* prefill-budget optimum under non-stationary concurrency that `hslo` tracks while a static budget cannot.

**Architecture:** Factor all timing/geometry into one pure, unit-tested stdlib module (`scripts/replay_timing.py`) shared by the replay driver and a new per-phase analyzer. Add a fourth driver branch to the replay for a wall-clock concurrency schedule. Two isolated-sequential orchestrators run the arms on the box; no per-token logging change (token emission times are reconstructed from existing `ts`/`latency`/`ttft`/`tbt_ms`).

**Tech Stack:** Python 3.9 stdlib only (Mac `python3` for tests/derivation; box `venv-vllm023` for runs); bash orchestrators; vLLM 0.23.0 V1.

## Global Constraints

- **Box:** 8×4090 `183.147.142.123`, vLLM 0.23.0, Qwen2.5-Coder-14B-Instruct, TP=2, GPUs 0,1, port 8050. Server flags: `--max-num-batched-tokens 16384 --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 48`.
- **Workload (both experiments):** single-turn, real ShareGPT first-turns, `--whale-frac 0.15 --whale-min-chars 44000 --whale-max-chars 50000 --max-prompt-chars 50000 --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 --pad-seed 1001 --max-tokens 256`, MODEL `/data/pli/models/Qwen2.5-Coder-14B-Instruct`, DATASET `data/sharegpt_v3.json`.
- **Arms are ISOLATED SEQUENTIAL** — each alone on GPUs 0,1.
- **Metric:** pooled P99 / max TBT + TTFT.
- **Kill discipline:** `kill_ours()` may only kill processes whose `/proc/PID/cmdline` contains `venv-vllm023`. Confirm box idle (only venv-vllm023 present) before any launch.
- **Code home:** edit locally on the Mac (source of truth), `rsync` up to `/root/pli/vllm-experiment`, commit/push from the Mac. GitHub is blocked from the box.
- **Stdlib only** — no new pip dependencies. Mac Python is 3.9.6; box python is `/root/pli/venv-vllm023/bin/python`.
- **E1 validity rule (one-directional):** a clean-mono result (P99 TBT ≈ decode baseline) means "raise the rate and re-run," NEVER a negative result.

---

### Task 1: Shared timing/geometry helpers (`scripts/replay_timing.py`)

Pure, stdlib-only, no side effects → fully unit-testable on the Mac with no box. This is the DRY core both the driver and the analyzer import.

**Files:**
- Create: `scripts/replay_timing.py`
- Test: `tests/test_replay_timing.py`

**Interfaces:**
- Produces:
  - `parse_schedule(s: str) -> list[tuple[int, float]]` — `"8@50,40@50"` → `[(8, 50.0), (40, 50.0)]`
  - `phase_at(sched: list, t: float) -> tuple[int, int, int]` — elapsed `t` → `(phase_idx, concurrency, cycle)`, cycling the schedule
  - `token_times(rec: dict) -> list[float]` — absolute wall-clock emission time of each output token
  - `realized_concurrency(recs: list[dict]) -> tuple[float, int]` — `(time-weighted mean, max)` in-flight
  - `realized_throughput(recs: list[dict]) -> float` — completions / wall-span (conv/s)
  - `bucket_by_phase(recs: list[dict], sched: list) -> dict[int, dict]` — `phase_idx → {"tbt": [ms...], "ttft": [s...], "conc": int}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_replay_timing.py`:

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import replay_timing as rt


def test_parse_schedule():
    assert rt.parse_schedule("8@50,40@50") == [(8, 50.0), (40, 50.0)]
    assert rt.parse_schedule(" 8@50 , 40@50 ") == [(8, 50.0), (40, 50.0)]


def test_phase_at_cycles():
    s = rt.parse_schedule("8@50,40@50")
    assert rt.phase_at(s, 0.0)   == (0, 8, 0)
    assert rt.phase_at(s, 49.9)  == (0, 8, 0)
    assert rt.phase_at(s, 50.0)  == (1, 40, 0)
    assert rt.phase_at(s, 99.9)  == (1, 40, 0)
    assert rt.phase_at(s, 100.0) == (0, 8, 1)     # second cycle, low phase
    assert rt.phase_at(s, 150.0) == (1, 40, 1)


def test_token_times():
    # start = ts - latency = 100.0; ttft = 2.0; two ITL gaps of 30ms, 50ms => 3 tokens
    rec = {"ts": 105.0, "latency": 5.0, "ttft": 2.0, "tbt_ms": [30.0, 50.0]}
    tt = rt.token_times(rec)
    assert len(tt) == 3
    assert abs(tt[0] - 102.0) < 1e-9      # 100 + ttft
    assert abs(tt[1] - 102.03) < 1e-9     # + 30ms
    assert abs(tt[2] - 102.08) < 1e-9     # + 50ms
    assert rt.token_times({"ts": None}) == []


def test_realized_concurrency_and_throughput():
    # three overlapping requests: [0,10], [5,15], [12,20]
    recs = [
        {"ts": 10.0, "latency": 10.0},
        {"ts": 15.0, "latency": 10.0},
        {"ts": 20.0, "latency": 8.0},
    ]
    mean, mx = rt.realized_concurrency(recs)
    assert mx == 2                        # never 3 at once (3rd starts at 12, 1st ends at 10)
    assert 0.0 < mean < 2.0
    tp = rt.realized_throughput(recs)     # 3 completions over span [0,20]
    assert abs(tp - 3 / 20.0) < 1e-9


def test_bucket_by_phase():
    s = rt.parse_schedule("8@50,40@50")
    # one request: start_wall=0 (ttft in phase 0), tokens stretch to t=60 (phase 1)
    # ts - latency = 0 => ts=latency. ttft=1s. many 500ms ITLs walking across the boundary.
    rec = {"ts": 61.0, "latency": 61.0, "ttft": 1.0, "tbt_ms": [500.0] * 118}
    b = rt.bucket_by_phase([rec], s)
    assert b[0]["conc"] == 8 and b[1]["conc"] == 40
    assert len(b[0]["ttft"]) == 1 and len(b[1]["ttft"]) == 0   # ttft at start (phase 0)
    assert len(b[0]["tbt"]) > 0 and len(b[1]["tbt"]) > 0        # tokens span both phases


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 tests/test_replay_timing.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'replay_timing'`

- [ ] **Step 3: Write the implementation**

Create `scripts/replay_timing.py`:

```python
#!/usr/bin/env python3
"""Pure timing/geometry helpers shared by the replay driver (src/replay_sharegpt.py) and the
non-stationary analyzer (scripts/analyze_nonstationary.py). Stdlib only, no side effects, so it is
unit-testable on any machine without the box (tests/test_replay_timing.py).

Per-token emission times are RECONSTRUCTED from the existing log fields (ts, latency, ttft, tbt_ms)
so no change to the replay's logging is needed:
    start_wall   = ts - latency
    token[1]     = start_wall + ttft
    token[k>=2]  = start_wall + ttft + sum(tbt_ms[:k-1]) / 1000
"""


def parse_schedule(s):
    """'8@50,40@50' -> [(8, 50.0), (40, 50.0)]  (concurrency, phase_seconds)."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        n, sec = part.split("@")
        out.append((int(n), float(sec)))
    if not out:
        raise ValueError(f"empty schedule: {s!r}")
    return out


def phase_at(sched, t):
    """Elapsed time t (s) -> (phase_idx, concurrency, cycle). Schedule cycles forever."""
    period = sum(sec for _, sec in sched)
    if period <= 0:
        raise ValueError("schedule period must be > 0")
    cycle = int(t // period)
    off = t - cycle * period
    for idx, (n, sec) in enumerate(sched):
        if off < sec:
            return idx, n, cycle
        off -= sec
    idx = len(sched) - 1           # floating-point edge -> last phase
    return idx, sched[idx][0], cycle


def token_times(rec):
    """Absolute wall-clock emission time of each output token (len == output token count)."""
    ts, lat, ttft = rec.get("ts"), rec.get("latency"), rec.get("ttft")
    tbt = rec.get("tbt_ms") or []
    if ts is None or lat is None or ttft is None:
        return []
    start = ts - lat
    times = [start + ttft]
    acc = ttft
    for ms in tbt:
        acc += ms / 1000.0
        times.append(start + acc)
    return times


def _intervals(recs):
    iv = []
    for r in recs:
        ts, lat = r.get("ts"), r.get("latency")
        if ts is not None and lat is not None:
            iv.append((ts - lat, ts))
    return iv


def realized_concurrency(recs):
    """(time-weighted mean, peak) count of overlapping [start, ts] request intervals."""
    iv = _intervals(recs)
    if not iv:
        return 0.0, 0
    events = []
    for s, e in iv:
        events.append((s, +1))
        events.append((e, -1))
    events.sort(key=lambda x: (x[0], -x[1]))   # at a tie, opens (+1) before closes (-1) => peak
    cur = mx = 0
    area = 0.0
    prev_t = events[0][0]
    for t, d in events:
        area += cur * (t - prev_t)
        prev_t = t
        cur += d
        mx = max(mx, cur)
    span = events[-1][0] - events[0][0]
    mean = area / span if span > 0 else float(mx)
    return mean, mx


def realized_throughput(recs):
    """Completions per wall-second over the run span (conv/s)."""
    iv = _intervals(recs)
    if not iv:
        return 0.0
    span = max(e for _, e in iv) - min(s for s, _ in iv)
    return len(iv) / span if span > 0 else 0.0


def bucket_by_phase(recs, sched):
    """phase_idx -> {'tbt': [ms...], 'ttft': [s...], 'conc': int}. t0 = earliest start_wall.
    Each request's TTFT is bucketed by its start_wall; each per-token TBT by that token's emission
    time. Phases with the same index across cycles aggregate together (all low phases pooled, etc.)."""
    starts = [ts - lat for ts, lat in
              ((r.get("ts"), r.get("latency")) for r in recs) if ts is not None and lat is not None]
    if not starts:
        return {}
    t0 = min(starts)
    out = {}

    def _bucket(idx, conc):
        return out.setdefault(idx, {"tbt": [], "ttft": [], "conc": conc})

    for r in recs:
        ts, lat, ttft = r.get("ts"), r.get("latency"), r.get("ttft")
        if ts is None or lat is None:
            continue
        start = ts - lat
        if ttft is not None:
            idx, conc, _ = phase_at(sched, start - t0)
            _bucket(idx, conc)["ttft"].append(ttft)
        tt = token_times(r)                     # tt[0] is token1 (at ttft); tt[j+1] is token j+2
        for j, ms in enumerate(r.get("tbt_ms") or []):
            emit = tt[j + 1] if (j + 1) < len(tt) else start
            idx, conc, _ = phase_at(sched, emit - t0)
            _bucket(idx, conc)["tbt"].append(ms)
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 tests/test_replay_timing.py`
Expected: `All 5 tests passed.`

- [ ] **Step 5: Commit**

```bash
git add scripts/replay_timing.py tests/test_replay_timing.py
git commit -m "feat: pure timing/geometry helpers for replay + phase analysis (TDD)"
```

---

### Task 2: Open-loop (E1) orchestrator + rate derivation

No harness code — `--rate` already exists. This task adds the orchestrator and grounds the Poisson rate in the closed-loop realized throughput, with the validity gate.

**Files:**
- Create: `orchestrate_longprompt_openloop.sh`

**Interfaces:**
- Consumes: `scripts/replay_timing.realized_throughput` (Task 1), `scripts/analyze_longprompt.py` (existing, unchanged).
- Produces: `logs/longpol_ANALYSIS.txt`, marker `logs/longpol_ALLDONE`.

- [ ] **Step 1: Write the orchestrator**

Create `orchestrate_longprompt_openloop.sh`:

```bash
#!/bin/bash
# orchestrate_longprompt_openloop.sh -- E1: open-loop Poisson version of the whale regime check.
# Identical bimodal-whale workload to orchestrate_longprompt.sh, but drives arrivals with --rate
# (open-loop Poisson) instead of --concurrency. RATE is derived from the realized throughput of the
# 07-21 closed-loop conc-20 chunk-2048 run (Little's law: same mean in-flight population that produced
# the original signal), unless overridden by env RATE. VALIDITY GATE: a clean-mono result (P99 TBT ~
# decode baseline) means "raise RATE and re-run", never a negative result -- see analysis footer.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/longpol_ALLDONE logs/longpol_FAILED
DATE=$(date +%Y-%m-%d)
MAX_SEQS=${MAX_SEQS:-48}; MAXTOK=${MAXTOK:-256}; NCONV=${NCONV:-200}
WHALE_FRAC=${WHALE_FRAC:-0.15}; WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}; PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
REF=${REF:-logs/2026-07-21-longp-b2048-t1.jsonl}
PORT=8050
BUDGETS_RUN="16384 2048 512"

# Derive RATE from the reference closed-loop run's realized throughput (conv/s) unless overridden.
if [ -z "${RATE:-}" ]; then
  RATE=$($PYTHON - "$REF" <<'PY'
import sys, json, os
sys.path.insert(0, "scripts"); import replay_timing as rt
p = sys.argv[1]
recs = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
tp = rt.realized_throughput(recs)
print(f"{tp:.3f}" if tp > 0 else "1.5")
PY
)
fi
log "E1 open-loop: RATE=${RATE} conv/s (ref=$REF)  arms: $BUDGETS_RUN"

for B in $BUDGETS_RUN; do
  ARM=b${B}ol
  log "  [$ARM] server GPUs 0,1 port=$PORT budget=$B"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
      $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $PORT --max-num-seqs $MAX_SEQS --max-num-batched-tokens $B \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-longp-${ARM}-server.log 2>&1 &
  SV=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-longp-${ARM}-server.log && break
    [ "$i" = 120 ] && { log "  [$ARM] SERVER TIMEOUT"; kill "$SV" 2>/dev/null; touch logs/longpol_FAILED; exit 1; }
  done
  out="logs/${DATE}-longp-${ARM}-t1.jsonl"
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
    --max-tokens $MAXTOK --rate $RATE \
    --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
    --whale-frac $WHALE_FRAC --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
    --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed 1001 \
    --output "$out" > "${out%.jsonl}.client.log" 2>&1 || true
  log "  [$ARM] done recs=$(grep -c . "$out" 2>/dev/null || echo 0) preempt=$(grep -c -i preempt logs/${DATE}-longp-${ARM}-server.log 2>/dev/null || echo 0)"
  kill "$SV" 2>/dev/null; sleep 8; kill -9 "$SV" 2>/dev/null; kill_ours
done

log "analyzing (open-loop; mono baseline + realized concurrency validity)"
# core metric via the existing analyzer, over the -ol arm files
BUDGETS="16384ol 2048ol 512ol" $PYTHON scripts/analyze_longprompt.py > logs/longpol_ANALYSIS.txt 2>&1
# validity gate: realized mean/max concurrency per arm (must be non-trivial for a valid run)
$PYTHON - "$DATE" >> logs/longpol_ANALYSIS.txt 2>&1 <<'PY'
import sys, json, glob
sys.path.insert(0, "scripts"); import replay_timing as rt
date = sys.argv[1]
print("\n=== validity gate: realized concurrency (Little's law check) ===")
print(f"driven RATE derived from reference throughput; a live decode batch is required for signal.")
for arm in ("16384ol", "2048ol", "512ol"):
    recs = []
    for f in glob.glob(f"logs/{date}-longp-b{arm}-t1.jsonl"):
        recs += [json.loads(l) for l in open(f) if l.strip()]
    mean, mx = rt.realized_concurrency(recs)
    tp = rt.realized_throughput(recs)
    print(f"  {arm:8s}: n={len(recs):4d}  realized_conc mean={mean:5.1f} max={mx:3d}  throughput={tp:.3f}/s")
print("\nVALIDITY: if the mono (16384ol) P99 TBT above is ~ its decode baseline (no multi-second tail),")
print("the rate was too low (no decoders to freeze) -> RAISE RATE and re-run; do NOT read as a null.")
PY
echo "[$(STAMP)] DONE" >> logs/longpol_ANALYSIS.txt
touch logs/longpol_ALLDONE
log "done -> logs/longpol_ANALYSIS.txt"
```

- [ ] **Step 2: Syntax-check the orchestrator**

Run: `bash -n orchestrate_longprompt_openloop.sh`
Expected: no output (exit 0).

- [ ] **Step 3: Commit**

```bash
git add orchestrate_longprompt_openloop.sh
git commit -m "feat: E1 open-loop Poisson whale regime orchestrator + rate derivation + validity gate"
```

---

### Task 3: Concurrency-schedule driver in the replay (`--concurrency-schedule`, `--duration`)

Adds a fourth, isolated driver branch. Existing closed-loop / open-loop / stagger branches are untouched.

**Files:**
- Modify: `src/replay_sharegpt.py` (argparse block near the other `--rate`/`--concurrency` args; driver dispatch near the `if args.rate:` branch)

**Interfaces:**
- Consumes: `scripts/replay_timing.parse_schedule`, `phase_at` (Task 1); `replay_conversation(ci, conv, args, records, records_lock, print_lock)` (existing).
- Produces: a `--concurrency-schedule "N1@S1,N2@S2,..."` + `--duration SECONDS` closed-loop phase driver.

- [ ] **Step 1: Add the import shim (top of `src/replay_sharegpt.py`, after existing imports)**

Find the existing import block (it uses `import argparse, json, os, random, threading, time, urllib...`). Immediately after it, add:

```python
# Shared pure timing/geometry helpers live in scripts/ (imported by the analyzers too).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from replay_timing import parse_schedule, phase_at  # noqa: E402
```

(If `sys` is not already imported, add `import sys` to the import block.)

- [ ] **Step 2: Add the argparse flags**

Find the `--rate` argument definition (`ap.add_argument("--rate", ...)`). Immediately after the `--min-turns` argument block, add:

```python
    ap.add_argument("--concurrency-schedule", default=None,
                    help='Non-stationary closed-loop schedule "N1@S1,N2@S2,..." (concurrency@seconds), '
                         'cycled until --duration elapses. Overrides --concurrency/--rate/--stagger.')
    ap.add_argument("--duration", type=float, default=None,
                    help="Total wall-clock seconds for --concurrency-schedule (required with it).")
```

- [ ] **Step 3: Add the driver branch**

Find the driver dispatch — it begins with `if args.rate:` (open-loop) followed by `elif args.stagger_window is not None:` and `else:` (closed-loop). Change the **first** condition from `if args.rate:` to check the schedule first, by inserting a new branch ABOVE it:

```python
    if args.concurrency_schedule:
        # Non-stationary closed-loop: maintain in_flight ~= N(t) where N steps through the phase
        # schedule over wall-clock. Rising phase -> launch immediately (crisp boundary); falling
        # phase -> stop launching, let running convs drain (no kills, no runaway queue). The conv
        # list is recycled with a per-launch unique id (conv_seq) so refill prompts stay
        # un-cacheable and whale-frac holds in expectation.
        sched = parse_schedule(args.concurrency_schedule)
        if not args.duration:
            raise SystemExit("ERROR: --concurrency-schedule requires --duration")
        in_flight = {"n": 0}
        lock2 = threading.Lock()
        threads = []
        conv_seq = 0
        t_start = time.monotonic()

        def _run_one(seq, conv):
            try:
                replay_conversation(seq, conv, args, records, records_lock, print_lock)
            finally:
                with lock2:
                    in_flight["n"] -= 1

        while True:
            elapsed = time.monotonic() - t_start
            if elapsed >= args.duration:
                break
            _, target, _ = phase_at(sched, elapsed)
            with lock2:
                room = target - in_flight["n"]
            for _ in range(max(0, room)):
                conv = convs[conv_seq % len(convs)]
                with lock2:
                    in_flight["n"] += 1
                th = threading.Thread(target=_run_one, args=(conv_seq, conv), daemon=True)
                th.start()
                threads.append(th)
                conv_seq += 1
            threads = [t for t in threads if t.is_alive()]   # prune finished
            time.sleep(0.5)
        for t in threads:
            t.join(timeout=130)
    elif args.rate:
```

(The line `elif args.rate:` replaces the former `if args.rate:` — verify the remaining `elif args.stagger_window` / `else` chain is unchanged.)

- [ ] **Step 4: Smoke-test schedule parsing wiring (Mac, no server)**

Run:
```bash
python3 -c "import sys, os; sys.path.insert(0,'scripts'); from replay_timing import parse_schedule, phase_at; s=parse_schedule('8@50,40@50'); print(phase_at(s, 75.0))"
```
Expected: `(1, 40, 0)`

- [ ] **Step 5: Byte-check the existing branches are intact**

Run: `grep -n "if args.concurrency_schedule:\|elif args.rate:\|elif args.stagger_window is not None:" src/replay_sharegpt.py`
Expected: three lines, in that order.

- [ ] **Step 6: Commit**

```bash
git add src/replay_sharegpt.py
git commit -m "feat: --concurrency-schedule/--duration non-stationary closed-loop driver"
```

---

### Task 4: Per-phase analyzer (`scripts/analyze_nonstationary.py`)

**Files:**
- Create: `scripts/analyze_nonstationary.py`
- Test: extend `tests/test_replay_timing.py` (bucketing is already covered; add a trace-bucketing test)

**Interfaces:**
- Consumes: `scripts/replay_timing.bucket_by_phase`, `parse_schedule`, `phase_at` (Task 1).
- Produces: `logs/longpns_ANALYSIS.txt` with a per-phase `phase | conc | arm | TTFT | TBT-p99 | TBT-max` table + hslo per-phase budget median/p90.

- [ ] **Step 1: Write the failing test (trace bucketing)**

Append to `tests/test_replay_timing.py` (before the `if __name__` runner):

```python
def test_bucket_trace_rows():
    # emulate chunktrace rows: wall_s crossing the 50s boundary; depth>0 marks activity start
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import analyze_nonstationary as an
    s = rt.parse_schedule("8@50,40@50")
    rows = [
        {"wall_s": "1000.0", "depth": "0",  "chunk": "16384"},  # idle, ignored for t0
        {"wall_s": "1002.0", "depth": "5",  "chunk": "2000"},   # t0 here -> phase 0
        {"wall_s": "1055.0", "depth": "30", "chunk": "900"},    # +53s -> phase 1
    ]
    b = an.bucket_trace(rows, s)
    assert 2000 in b[0] and 900 in b[1]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/test_replay_timing.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'analyze_nonstationary'`

- [ ] **Step 3: Write the analyzer**

Create `scripts/analyze_nonstationary.py`:

```python
#!/usr/bin/env python3
"""Per-phase analysis for the non-stationary (concurrency-schedule) whale experiment.

Buckets pooled P99/max TBT and TTFT into concurrency phases (all cycles of a phase pooled), so we can
show: a static budget wins one phase and loses another, while hslo tracks the moving optimum. Phase
membership is reconstructed from the existing log fields (no per-token logging change).

Env:
  SCHEDULE  e.g. "8@50,40@50"   (required)
  ARMS      e.g. "16384 2048 512 hslo400ns"   (space-separated arm labels; first = mono baseline)
Reads logs/*-longp-b{arm}-t1.jsonl and, for hslo* arms, logs/*-longp-b{arm}-chunktrace.csv.
"""
import csv, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_timing import bucket_by_phase, parse_schedule, phase_at

SCHEDULE = os.environ["SCHEDULE"]
ARMS = os.environ.get("ARMS", "16384 2048 512 hslo400ns").split()


def pctl(x, p):
    if not x:
        return 0.0
    y = sorted(x)
    k = (len(y) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(y) - 1)
    return y[lo] + (y[hi] - y[lo]) * (k - lo)


def load(arm):
    recs = []
    for f in glob.glob(f"logs/*-longp-b{arm}-t1.jsonl"):
        recs += [json.loads(l) for l in open(f) if l.strip()]
    return recs


def bucket_trace(rows, sched):
    """chunktrace rows (dicts with wall_s, depth, chunk) -> phase_idx -> [chunk...].
    t0 aligned to the first row with depth>0 (~ first request activity)."""
    parsed = []
    for r in rows:
        try:
            parsed.append((float(r["wall_s"]), int(float(r["depth"])), int(float(r["chunk"]))))
        except (KeyError, ValueError):
            continue
    active = [w for w, d, _ in parsed if d > 0]
    if not active:
        return {}
    t0 = min(active)
    out = {}
    for w, _, ch in parsed:
        if w < t0:
            continue
        idx, _, _ = phase_at(sched, w - t0)
        out.setdefault(idx, []).append(ch)
    return out


def main():
    sched = parse_schedule(SCHEDULE)
    nphase = len(sched)
    print(f"schedule={SCHEDULE}  phases={[(n, s) for n, s in sched]}  arms={ARMS}\n")
    header = f"{'phase':>5} {'conc':>5} {'arm':>10} {'n_tok':>7} {'TTFT_ms':>8} {'TBTp99':>8} {'TBTmax':>8}"
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        recs = load(arm)
        b = bucket_by_phase(recs, sched)
        for idx in range(nphase):
            ph = b.get(idx, {"tbt": [], "ttft": [], "conc": sched[idx][0]})
            ttft_ms = 1000.0 * (sum(ph["ttft"]) / len(ph["ttft"])) if ph["ttft"] else 0.0
            print(f"{idx:>5} {ph['conc']:>5} {arm:>10} {len(ph['tbt']):>7} "
                  f"{ttft_ms:>8.0f} {pctl(ph['tbt'], 99):>8.1f} {(max(ph['tbt']) if ph['tbt'] else 0):>8.1f}")
        print()

    # hslo per-phase budget (evidence the controller tracks the phase)
    for arm in ARMS:
        if not arm.startswith("hslo"):
            continue
        for f in glob.glob(f"logs/*-longp-b{arm}-chunktrace.csv"):
            rows = list(csv.DictReader(open(f)))
            bt = bucket_trace(rows, sched)
            print(f"hslo budget by phase  [{os.path.basename(f)}]:")
            for idx in range(nphase):
                ch = bt.get(idx, [])
                if ch:
                    print(f"  phase {idx} (conc {sched[idx][0]:>3}): budget median={pctl(ch,50):.0f} "
                          f"p90={pctl(ch,90):.0f} min={min(ch)} max={max(ch)} steps={len(ch)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 tests/test_replay_timing.py`
Expected: `All 6 tests passed.`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_nonstationary.py tests/test_replay_timing.py
git commit -m "feat: per-phase non-stationary analyzer (TBT/TTFT + hslo budget by phase)"
```

---

### Task 5: E2 de-risk probe + non-stationary orchestrator

The de-risk probe confirms the optimum actually moves before committing the phase params; then the orchestrator runs the four arms.

**Files:**
- Create: `orchestrate_longprompt_nonstationary.sh`

**Interfaces:**
- Consumes: `scripts/hotpatch_hslo.py`, `scripts/hotpatch_hslo_alphafloor.py` (existing); `src/replay_sharegpt.py` `--concurrency-schedule`/`--duration` (Task 3); `scripts/analyze_nonstationary.py` (Task 4).
- Produces: `logs/longpns_ANALYSIS.txt`, marker `logs/longpns_ALLDONE`.

- [ ] **Step 1: Write the de-risk probe note into the orchestrator header + write the orchestrator**

Create `orchestrate_longprompt_nonstationary.sh`:

```bash
#!/bin/bash
# orchestrate_longprompt_nonstationary.sh -- E2 (Option A): non-stationary concurrency phases.
# Whales present throughout; concurrency alternates LOW<->HIGH over wall-clock phases (SCHEDULE), so
# the SLO-correct prefill budget MOVES: static-2048 is clean in the low phase but violates the SLO in
# the high phase (db_high + 2048*alpha > SLO); static-512 pays TTFT in the low phase; hslo tracks
# both. Four arms, isolated sequential, paired workload (pad-seed 1001), per-phase analysis.
#
# DE-RISK (run BEFORE trusting phase params): confirm decode_baseline moves a meaningful fraction of
# the 400ms SLO across the concurrency range, else the optimum barely shifts. Quick check from the
# existing hslo trace:
#   /root/pli/venv-vllm023/bin/python scripts/plot_chunk_trace.py logs/2026-07-22-longp-bhslo400af-chunktrace.csv
# (inspect signal_ms = db vs depth). If db(conc40) is not a large fraction of the SLO, raise HIGH
# toward 48 and/or WHALE_FRAC before running. Default SCHEDULE below assumes the separation holds.
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/longpns_ALLDONE logs/longpns_FAILED
DATE=$(date +%Y-%m-%d)
MAX_SEQS=${MAX_SEQS:-48}; MAXTOK=${MAXTOK:-256}; NCONV=${NCONV:-200}
WHALE_FRAC=${WHALE_FRAC:-0.15}; WHALE_MIN=${WHALE_MIN:-44000}; WHALE_MAX=${WHALE_MAX:-50000}
MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-50000}; PAD_MEAN=${PAD_MEAN:-800}; PAD_CV2=${PAD_CV2:-0.5}
SCHEDULE=${SCHEDULE:-"8@50,40@50"}; DURATION=${DURATION:-300}
FLOOR=${FLOOR:-512}; START=${START:-512}; SLO_MS=${SLO_MS:-400}; ALPHA_MIN=${ALPHA_MIN:-256}; ALPHA_HW_MS=${ALPHA_HW_MS:-0.18}
PORT=8050

$PYTHON scripts/hotpatch_hslo.py || { log "PATCH(base) FAILED"; touch logs/longpns_FAILED; exit 1; }
$PYTHON scripts/hotpatch_hslo_alphafloor.py || { log "PATCH(alphafloor) FAILED"; touch logs/longpns_FAILED; exit 1; }

log "E2 non-stationary: SCHEDULE='$SCHEDULE' DURATION=${DURATION}s whale_frac=$WHALE_FRAC"

# arm spec: label | budget-or-hslo. For hslo we pass mode env; budget arms just set --max-num-batched-tokens.
run_arm(){ # $1=label $2=mode(static|hslo) $3=budget(for static)
  local ARM=$1 MODE=$2 BUD=$3
  log "  [$ARM] server GPUs 0,1 port=$PORT mode=$MODE budget=$BUD"
  local EXTRA="DYNAMIC_CHUNK=0"
  if [ "$MODE" = "hslo" ]; then
    EXTRA="DYNAMIC_CHUNK=1 CHUNK_MODE=hslo DYNAMIC_CHUNK_MIN=$FLOOR DYNAMIC_CHUNK_START=$START DYNAMIC_CHUNK_SLO_MS=$SLO_MS DYNAMIC_CHUNK_ALPHA_MIN_PREFILL=$ALPHA_MIN DYNAMIC_CHUNK_ALPHA_MIN=$ALPHA_HW_MS DYNAMIC_CHUNK_TRACE=logs/${DATE}-longp-${ARM}-chunktrace.csv"
    BUD=16384   # hslo controls the budget dynamically; server ceiling stays 16384
  fi
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 $EXTRA \
      $PYTHON -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port $PORT --max-num-seqs $MAX_SEQS --max-num-batched-tokens $BUD \
      --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
      > logs/${DATE}-longp-${ARM}-server.log 2>&1 &
  local SV=$!
  for i in $(seq 1 120); do sleep 5
    grep -q "Application startup complete" logs/${DATE}-longp-${ARM}-server.log && break
    [ "$i" = 120 ] && { log "  [$ARM] SERVER TIMEOUT"; kill "$SV" 2>/dev/null; touch logs/longpns_FAILED; exit 1; }
  done
  local out="logs/${DATE}-longp-${ARM}-t1.jsonl"
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs $NCONV --max-turns 1 --min-turns 1 \
    --max-tokens $MAXTOK --concurrency-schedule "$SCHEDULE" --duration $DURATION \
    --pad-mean-chars $PAD_MEAN --pad-cv2 $PAD_CV2 --pad-min 100 --pad-max 8000 \
    --whale-frac $WHALE_FRAC --whale-min-chars $WHALE_MIN --whale-max-chars $WHALE_MAX \
    --max-prompt-chars $MAX_PROMPT_CHARS --pad-seed 1001 \
    --output "$out" > "${out%.jsonl}.client.log" 2>&1 || true
  log "  [$ARM] done recs=$(grep -c . "$out" 2>/dev/null || echo 0) preempt=$(grep -c -i preempt logs/${DATE}-longp-${ARM}-server.log 2>/dev/null || echo 0)"
  kill "$SV" 2>/dev/null; sleep 8; kill -9 "$SV" 2>/dev/null; kill_ours
}

run_arm 16384ns   static 16384
run_arm 2048ns    static 2048
run_arm 512ns     static 512
run_arm hslo400ns hslo   16384

log "analyzing (per-phase TBT/TTFT + hslo budget by phase)"
SCHEDULE="$SCHEDULE" ARMS="16384ns 2048ns 512ns hslo400ns" \
  $PYTHON scripts/analyze_nonstationary.py > logs/longpns_ANALYSIS.txt 2>&1
echo "[$(STAMP)] DONE" >> logs/longpns_ANALYSIS.txt
touch logs/longpns_ALLDONE
log "done -> logs/longpns_ANALYSIS.txt"
```

- [ ] **Step 2: Syntax-check**

Run: `bash -n orchestrate_longprompt_nonstationary.sh`
Expected: no output (exit 0).

- [ ] **Step 3: Commit**

```bash
git add orchestrate_longprompt_nonstationary.sh
git commit -m "feat: E2 non-stationary concurrency-phase orchestrator (4 arms + per-phase analysis)"
```

---

### Task 6: Deploy, run on the box, analyze

Runs are integration — validated by markers + analysis output, not unit tests. Confirm box idle first.

- [ ] **Step 1: Confirm box idle**

Run: `ssh 183.147.142.123 "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; echo ---; for p in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do tr '\0' ' ' </proc/\$p/cmdline; echo; done"`
Expected: no `venv-vllm023` api_server processes running (or GPUs 0,1 free). If another user's process is present, do NOT kill it — wait.

- [ ] **Step 2: rsync code up**

Run: `rsync -avz --exclude 'logs/' --exclude '.git/' --exclude 'data/' /Users/li/Documents/vllm-experiment/ root@183.147.142.123:/root/pli/vllm-experiment/`
Expected: transfers `scripts/replay_timing.py`, `scripts/analyze_nonstationary.py`, `src/replay_sharegpt.py`, both new orchestrators, `tests/`.

- [ ] **Step 3: Verify helpers import under the box python**

Run: `ssh 183.147.142.123 "cd /root/pli/vllm-experiment && /root/pli/venv-vllm023/bin/python tests/test_replay_timing.py"`
Expected: `All 6 tests passed.`

- [ ] **Step 4: Run E1 (open-loop) in the background**

Run: `ssh 183.147.142.123 "cd /root/pli/vllm-experiment && nohup bash orchestrate_longprompt_openloop.sh > logs/longpol_run.log 2>&1 &"; then poll for `logs/longpol_ALLDONE` (each arm ~ a few min; 3 arms).
Expected marker: `logs/longpol_ALLDONE`. Read `logs/longpol_ANALYSIS.txt`.

- [ ] **Step 5: E1 validity check**

Read `logs/longpol_ANALYSIS.txt`. Confirm mono (`16384ol`) P99 TBT shows the multi-second freeze AND realized_conc mean is non-trivial (roughly matches the closed-loop ~20). If mono is clean / concurrency is tiny → re-run with `RATE=$(higher)` per the one-directional rule. Do not proceed to interpret as a null.

- [ ] **Step 6: E2 de-risk probe**

Run: `ssh 183.147.142.123 "cd /root/pli/vllm-experiment && /root/pli/venv-vllm023/bin/python scripts/plot_chunk_trace.py logs/2026-07-22-longp-bhslo400af-chunktrace.csv"`
Inspect `signal_ms` (= decode_baseline) range. If `db` at high depth is not a meaningful fraction of the 400ms SLO (target: high-phase budget ≲1024 vs low-phase ≳2048), set `HIGH`→48 and/or raise `WHALE_FRAC` via env before Step 7. Record the chosen `SCHEDULE`.

- [ ] **Step 7: Run E2 (non-stationary) in the background**

Run: `ssh 183.147.142.123 "cd /root/pli/vllm-experiment && SCHEDULE='8@50,40@50' DURATION=300 nohup bash orchestrate_longprompt_nonstationary.sh > logs/longpns_run.log 2>&1 &"`; poll for `logs/longpns_ALLDONE` (4 arms × ~5 min each).
Expected marker: `logs/longpns_ALLDONE`. Read `logs/longpns_ANALYSIS.txt`.

- [ ] **Step 8: E2 success check**

In `logs/longpns_ANALYSIS.txt` confirm: static-2048 shows the per-phase inversion (low-phase TTFT win, high-phase TBT-p99 crossing the SLO), and hslo holds P99 TBT ≈ SLO in *both* phases with the budget-by-phase trace showing it shrinks in the high phase and grows in the low phase. If not, note the observed behavior for interpretation (a genuine null here — static wins even non-stationary — is a real result, unlike E1's rate-null).

- [ ] **Step 9: Write findings + commit**

Write `findings/2026-07-22-openloop-nonstationary.md` (E1 regime-persistence result + E2 per-phase moving-optimum result), update `MEMORY.md` index line, `git add -A && git commit && git push` from the Mac.

---

## Self-Review

**Spec coverage:**
- E1 open-loop regime check → Task 2 (orchestrator + rate derivation) + Task 6 Steps 4–5. ✓
- E1 rate discipline / validity gate → Task 2 validity footer + Task 6 Step 5 one-directional rule. ✓
- Per-token wall-clock reconstruction (no logging change) → Task 1 `token_times`/`bucket_by_phase`. ✓
- E2 hypothesis + moving optimum → Task 5 orchestrator, Task 4 per-phase analyzer. ✓
- E2 de-risk `db(concurrency)` probe gating phase params → Task 5 header + Task 6 Step 6. ✓
- Harness addition 1 (concurrency-schedule driver, isolated branch) → Task 3. ✓
- Harness addition 2 (phase-aware analyzer) → Task 4. ✓
- Per-cycle uniqueness salt on recycle → Task 3 (`conv_seq` unique id per launch). ✓
- Isolated-sequential arms, kill discipline, rsync/commit-from-Mac → Global Constraints + Task 6. ✓

**Placeholder scan:** none — all code blocks are complete; phase params have concrete defaults with an explicit de-risk gate to adjust them.

**Type consistency:** `parse_schedule`→`list[(int,float)]` consumed by `phase_at`/`bucket_by_phase`/`bucket_trace`; `phase_at` returns `(idx, conc, cycle)` used consistently (driver ignores idx/cycle, analyzer uses idx); `bucket_by_phase` returns `{idx: {"tbt","ttft","conc"}}` matched in the analyzer; arm labels (`b{label}ol`, `{label}ns`) consistent between orchestrators and analyzer `ARMS`/glob. ✓
