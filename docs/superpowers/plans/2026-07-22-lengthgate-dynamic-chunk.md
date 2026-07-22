# Length-Gated Dynamic Chunking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a prompt-length-aware dynamic chunker (`CHUNK_MODE=lengthgate`) and a non-stationary whale-fraction workload, then compare it against mono/512/2048 to show it wins throughput in all-short phases *and* tail in whale phases — the dynamic win no static chunk can match.

**Architecture:** Extend the existing ShareGPT replay driver with an open-loop **phase schedule** `(rate, whale_frac, duration)`; add a scheduler hotpatch that gates the prefill budget on *prompt length + decoder presence*; drive four arms through an orchestrator; analyze per phase-type. Reuses the project's whale-pad, hotpatch, and phase-reconstruction infrastructure.

**Tech Stack:** Python 3.10 (stdlib only for the testable helpers), vLLM 0.23.0 (V1 scheduler), bash orchestration over SSH to the 8×4090 box.

## Global Constraints

- **Local is source of truth.** Edit files locally, `scp` up to `183.147.142.123:/root/pli/vllm-experiment/`, run on the box, commit from the Mac. GitHub is blocked from the box.
- **Paired comparison:** all four arms MUST see the identical arrival sequence and whale positions — the arrival schedule is generated from a fixed seed (`--pad-seed 1001`), never from wall-clock randomness.
- **kill_ours safety:** may only kill processes whose `/proc/PID/cmdline` contains `venv-vllm023` (TERM the api_server pattern, then `-9` any GPU compute-app matching that substring). Never touch another user's process.
- **Sub-saturation gate:** reject any operating point where per-phase TTFT is queue-dominated (pilot enforces a TTFT ceiling); a saturated reading is invalid, not a result.
- **Box/server invariants:** vLLM 0.23.0 V1, model Qwen2.5-Coder-14B-Instruct, `--tensor-parallel-size 2` on GPUs 0,1, `--gpu-memory-utilization 0.90`, `--max-model-len 16384`, `--max-num-seqs 128`, `--max-num-batched-tokens 16384`. Whales: pad uniform `[44000,50000]` chars, `--max-prompt-chars 50000`. Short: `--pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000`. `--max-tokens 256`.
- **Tests:** no pytest. Test functions named `test_*` in `tests/test_replay_timing.py`, run via `python3 tests/test_replay_timing.py` (the existing auto-discovering runner). Patcher/analyzer tests get their own `tests/test_*.py` with the same self-runner pattern.
- **Controller depends on the base `ChunkSizeController`** already present in the box's patched `scheduler.py` (modes depth/slo/feedforward/slocvar/slotail/hslo exist). The lengthgate hotpatch only *adds* a mode + the call-site prefill signal.

## File Structure

- `scripts/replay_timing.py` (MODIFY) — add `parse_phase_schedule`, `phase_type_at`, `generate_phase_arrivals` (pure, stdlib, testable).
- `tests/test_replay_timing.py` (MODIFY) — add tests for the three new helpers.
- `src/replay_sharegpt.py` (MODIFY) — add `--phase-schedule`, `force_whale` plumbing, and the open-loop phase-schedule driver branch.
- `scripts/hotpatch_lengthgate.py` (CREATE) — idempotent 4-edit patch adding `CHUNK_MODE=lengthgate` + the prefill-remaining signal at the call site.
- `tests/test_hotpatch_lengthgate.py` (CREATE) — offline test of the patcher (anchors present, idempotent) against a scheduler fixture.
- `scripts/analyze_lengthgate.py` (CREATE) — per-phase-type analysis (Phase S: TTFT/throughput; Phase W: P99/max TBT + request-goodput).
- `tests/test_analyze_lengthgate.py` (CREATE) — bucketing test on synthetic records.
- `pilot_lengthgate_rates.sh` (CREATE) — rate-calibration pilot (box).
- `orchestrate_lengthgate.sh` (CREATE) — four-arm run + analysis (box).

---

### Task 1: Phase-schedule helpers (parse, phase-type, arrival generation)

**Files:**
- Modify: `scripts/replay_timing.py`
- Test: `tests/test_replay_timing.py`

**Interfaces:**
- Produces:
  - `parse_phase_schedule(s: str) -> list[tuple[float,float,float]]` — `"10:0.0@60,3:0.2@60"` → `[(10.0,0.0,60.0),(3.0,0.2,60.0)]` (rate conv/s, whale_frac, phase_seconds).
  - `phase_type_at(sched, t) -> (idx:int, rate:float, frac:float, cycle:int)` — schedule cycles forever.
  - `generate_phase_arrivals(sched, duration, seed) -> list[tuple[float,bool,int]]` — deterministic `(arrival_s, is_whale, seq)`, sorted by arrival, one entry per Poisson arrival within `[0,duration)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_replay_timing.py`, above the `if __name__` runner):

```python
def test_parse_phase_schedule():
    from replay_timing import parse_phase_schedule
    assert parse_phase_schedule("10:0.0@60,3:0.2@45") == [(10.0, 0.0, 60.0), (3.0, 0.2, 45.0)]

def test_parse_phase_schedule_empty_rejected():
    from replay_timing import parse_phase_schedule
    try:
        parse_phase_schedule("")
        assert False, "expected ValueError"
    except ValueError:
        pass

def test_phase_type_at_cycles():
    from replay_timing import parse_phase_schedule, phase_type_at
    s = parse_phase_schedule("10:0.0@60,3:0.2@60")
    assert phase_type_at(s, 0.0)[1:3] == (10.0, 0.0)     # phase 0 (S)
    assert phase_type_at(s, 59.9)[0] == 0
    assert phase_type_at(s, 60.0)[1:3] == (3.0, 0.2)     # phase 1 (W)
    assert phase_type_at(s, 120.0)[0] == 0               # cycled back to S
    assert phase_type_at(s, 120.0)[3] == 1               # cycle index 1

def test_generate_phase_arrivals_deterministic_and_bounded():
    from replay_timing import parse_phase_schedule, generate_phase_arrivals
    s = parse_phase_schedule("20:0.0@30,5:0.5@30")
    a = generate_phase_arrivals(s, 120.0, seed=1001)
    b = generate_phase_arrivals(s, 120.0, seed=1001)
    assert a == b                                        # deterministic (paired arms)
    assert all(0.0 <= t < 120.0 for t, _, _ in a)        # bounded by duration
    assert [seq for _, _, seq in a] == list(range(len(a)))  # seq is 0..n-1 in arrival order
    # S phases (frac 0.0) produce no whales; W phases (frac 0.5) produce some
    from replay_timing import phase_type_at
    s_whales = sum(w for t, w, _ in a if phase_type_at(s, t)[2] == 0.0)
    w_whales = sum(w for t, w, _ in a if phase_type_at(s, t)[2] == 0.5)
    assert s_whales == 0 and w_whales > 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 tests/test_replay_timing.py`
Expected: FAIL (ImportError / AttributeError: `parse_phase_schedule` not defined).

- [ ] **Step 3: Implement the helpers** (append to `scripts/replay_timing.py`):

```python
def parse_phase_schedule(s):
    """'10:0.0@60,3:0.2@45' -> [(rate, whale_frac, seconds), ...] for the open-loop phase driver."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        rf, sec = part.split("@")
        rate, frac = rf.split(":")
        out.append((float(rate), float(frac), float(sec)))
    if not out:
        raise ValueError(f"empty phase schedule: {s!r}")
    return out


def phase_type_at(sched, t):
    """Elapsed t (s) -> (phase_idx, rate, whale_frac, cycle). Schedule cycles forever."""
    period = sum(sec for _, _, sec in sched)
    if period <= 0:
        raise ValueError("phase schedule period must be > 0")
    cycle = int(t // period)
    off = t - cycle * period
    for idx, (rate, frac, sec) in enumerate(sched):
        if off < sec:
            return idx, rate, frac, cycle
        off -= sec
    idx = len(sched) - 1
    return idx, sched[idx][0], sched[idx][1], cycle


def generate_phase_arrivals(sched, duration, seed):
    """Deterministic open-loop arrivals for a (rate, whale_frac, seconds) phase schedule.
    Piecewise-homogeneous Poisson: at time t use the current phase's rate for the next
    inter-arrival; the arrival's whale flag uses the phase's frac at its own time. Returns
    a sorted list of (arrival_s, is_whale, seq). Seeded so every arm replays identically."""
    import random as _random
    rng = _random.Random(f"phasearr-{seed}")
    out = []
    t = 0.0
    seq = 0
    while True:
        rate = phase_type_at(sched, t)[1]
        if rate <= 0:
            break
        t += rng.expovariate(rate)
        if t >= duration:
            break
        frac = phase_type_at(sched, t)[2]
        is_whale = rng.random() < frac
        out.append((t, is_whale, seq))
        seq += 1
    return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 tests/test_replay_timing.py`
Expected: PASS (all tests, including the pre-existing ones).

- [ ] **Step 5: Commit**

```bash
git add scripts/replay_timing.py tests/test_replay_timing.py
git commit -m "feat: phase-schedule parse/phase-type/arrival helpers for open-loop whale-fraction driver"
```

---

### Task 2: Open-loop phase-schedule driver branch

**Files:**
- Modify: `src/replay_sharegpt.py`
- Test: `tests/test_replay_timing.py` (the `force_whale` pad path is the unit-testable slice; the live driver is smoke-tested by the Task 5 pilot)

**Interfaces:**
- Consumes: `parse_phase_schedule`, `generate_phase_arrivals` (Task 1); existing `sample_pad_len`, `replay_conversation`.
- Produces: `--phase-schedule "R:F@D,..."` flag (open-loop, whale-fraction varies per phase); `sample_pad_len(..., force_whale=None)` and `replay_conversation(..., force_whale=None)` extended so the driver forces the whale decision per the deterministic arrival schedule.

- [ ] **Step 1: Write the failing test** (append to `tests/test_replay_timing.py`; it exercises the pure `force_whale` pad logic by importing the driver's `sample_pad_len`):

```python
def test_sample_pad_len_force_whale():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from replay_sharegpt import sample_pad_len
    class A:  # minimal args stand-in
        pad_seed = 1001; whale_frac = 0.0; whale_min_chars = 44000; whale_max_chars = 50000
        pad_mean_chars = 800; pad_cv2 = 0.5; pad_min = 100; pad_max = 8000; pad_chars = 0
    a = A()
    forced = sample_pad_len(0, 0, a, force_whale=True)
    assert 44000 <= forced <= 50000            # forced whale -> whale-range pad
    short = sample_pad_len(0, 0, a, force_whale=False)
    assert short <= 8000                        # forced non-whale -> normal pad path
    legacy = sample_pad_len(0, 0, a)            # force_whale=None -> unchanged behavior (wf=0 -> normal)
    assert legacy <= 8000
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/test_replay_timing.py`
Expected: FAIL (`sample_pad_len() got an unexpected keyword argument 'force_whale'`).

- [ ] **Step 3: Add `force_whale` to `sample_pad_len`** — replace the whale block at the top of `sample_pad_len` (lines 47-53) with:

```python
    wf = float(getattr(args, "whale_frac", 0.0) or 0.0)
    if force_whale is True:
        rc = random.Random(f"whale-{args.pad_seed}-{ci}-{turn_num}")
        lo = float(getattr(args, "whale_min_chars", 48000))
        hi = float(getattr(args, "whale_max_chars", 60000))
        return int(rc.uniform(lo, hi))
    if force_whale is None and wf > 0.0:
        rc = random.Random(f"whale-{args.pad_seed}-{ci}-{turn_num}")
        if rc.random() < wf:
            lo = float(getattr(args, "whale_min_chars", 48000))
            hi = float(getattr(args, "whale_max_chars", 60000))
            return int(rc.uniform(lo, hi))
    # force_whale is False -> skip whale, fall through to the normal pad path below.
```

and change the signature (line 29) to:

```python
def sample_pad_len(ci, turn_num, args, force_whale=None):
```

- [ ] **Step 4: Thread `force_whale` through `replay_conversation`** — change its signature (line 145) to:

```python
def replay_conversation(ci, conv, args, records, records_lock, print_lock, force_whale=None):
```

and its `sample_pad_len` call (line 159) to:

```python
        pad_len = sample_pad_len(ci, turn_num, args, force_whale=force_whale)
```

- [ ] **Step 5: Add the `--phase-schedule` arg** — after the `--duration` arg (line 246), add:

```python
    ap.add_argument("--phase-schedule", default=None,
                    help='Open-loop non-stationary whale-fraction schedule "R:F@S,..." '
                         '(rate_conv_per_s : whale_frac @ seconds), cycled until --duration. '
                         'Arrivals are Poisson, seeded by --pad-seed so all arms are paired. '
                         'Overrides --concurrency-schedule/--rate/--concurrency.')
```

- [ ] **Step 6: Import the new helpers** — change line 26 to:

```python
from replay_timing import (parse_schedule, phase_at,  # noqa: E402
                           parse_phase_schedule, generate_phase_arrivals)
```

- [ ] **Step 7: Add the driver branch** — insert this as the FIRST branch, immediately before `if args.concurrency_schedule:` (line 320). Also add its banner before the `if args.concurrency_schedule:` banner (line 301):

Banner (insert before line 301's `if args.concurrency_schedule:`):

```python
    if args.phase_schedule:
        print(f"[replay] {len(convs)} conversations | max_turns={args.max_turns} | "
              f"max_tokens={args.max_tokens} | phase-schedule={args.phase_schedule} "
              f"duration={args.duration}s (open-loop, whale-fraction non-stationary)")
    elif args.concurrency_schedule:
```

(i.e. demote the existing `if args.concurrency_schedule:` banner to `elif`.)

Driver branch (insert before line 320's `if args.concurrency_schedule:`, and demote that to `elif`):

```python
    if args.phase_schedule:
        # Open-loop non-stationary whale-fraction: a deterministic, seeded arrival schedule
        # (paired across arms) whose per-arrival whale flag follows the current phase's frac.
        # Each arrival launches one conversation thread after sleeping to its arrival time; the
        # whale decision is FORCED so the phase's frac holds exactly, not just in expectation.
        if not args.duration:
            raise SystemExit("ERROR: --phase-schedule requires --duration")
        sched = parse_phase_schedule(args.phase_schedule)
        arrivals = generate_phase_arrivals(sched, args.duration, args.pad_seed)
        print(f"[replay] phase-schedule: {len(arrivals)} arrivals over {args.duration}s "
              f"({sum(1 for _, w, _ in arrivals if w)} whales)")
        threads = []
        t_start = time.monotonic()
        for t_arr, is_whale, seq in arrivals:
            dt = t_arr - (time.monotonic() - t_start)
            if dt > 0:
                time.sleep(dt)
            conv = convs[seq % len(convs)]
            th = threading.Thread(
                target=replay_conversation,
                args=(seq, conv, args, records, records_lock, print_lock),
                kwargs={"force_whale": is_whale},
                daemon=True,
            )
            threads.append(th)
            th.start()
        for th in threads:
            th.join()
    elif args.concurrency_schedule:
```

- [ ] **Step 8: Run the pad-path test to verify it passes**

Run: `python3 tests/test_replay_timing.py`
Expected: PASS (including `test_sample_pad_len_force_whale`).

- [ ] **Step 9: Byte-compile the driver to catch syntax/indentation errors**

Run: `python3 -c "import ast; ast.parse(open('src/replay_sharegpt.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 10: Commit**

```bash
git add src/replay_sharegpt.py tests/test_replay_timing.py
git commit -m "feat: open-loop --phase-schedule driver (per-phase whale-fraction, forced whale, paired arrivals)"
```

---

### Task 3: Length-gated controller hotpatch

**Files:**
- Create: `scripts/hotpatch_lengthgate.py`
- Test: `tests/test_hotpatch_lengthgate.py`

**Interfaces:**
- Produces: `CHUNK_MODE=lengthgate` in the vLLM scheduler. Env: `LENGTHGATE_THRESHOLD` (default 4096 tok), `LENGTHGATE_PROTECT` (512), `LENGTHGATE_BLAST` (16384). Emits the standard `DYNAMIC_CHUNK_TRACE` CSV (`chunk` = gated budget, `signal_ms` = prefill-remaining).
- Consumes: the base `ChunkSizeController` and the call site in `schedule()` (both already present in the patched box scheduler).

The patch makes four idempotent string edits to `scheduler.py`:
1. `step()` signature — add `pf_remaining=None`.
2. dispatch — route `mode=="lengthgate"` to `_step_lengthgate`.
3. method — insert `_step_lengthgate` before `class Scheduler(SchedulerInterface):`.
4. call site — compute `_pf_remaining` (max prefill-tokens-remaining over running prefill-chunks + waiting requests) and pass it.

- [ ] **Step 1: Write the failing test** (`tests/test_hotpatch_lengthgate.py`):

```python
#!/usr/bin/env python3
"""Offline test of the lengthgate patcher: apply to a scheduler FIXTURE (the anchors the real
scheduler contains), verify all four edits land and re-applying is a no-op. No vLLM/box needed."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import hotpatch_lengthgate as H

FIXTURE = '''\
class ChunkSizeController:
    def step(self, decode_depth: int, last_tokens=None) -> int:
        if self.mode == "hslo":
            return self._step_hslo(decode_depth, last_tokens)
        if self.mode == "slo":
            return self._step_slo(decode_depth)
        return self._step_depth(decode_depth)


class Scheduler(SchedulerInterface):
    def schedule(self):
        token_budget = self.max_num_scheduled_tokens
        if self._chunk_ctrl is not None:
            _decode_depth = sum(
                1 for r in self.running if not r.is_prefill_chunk
            )
            token_budget = self._chunk_ctrl.step(
                _decode_depth, getattr(self, "_ff_last_tokens", None))
'''

def _apply(src):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scheduler.py")
    open(p, "w").write(src)
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    return rc, open(p).read()

def test_patch_applies_all_four_edits():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert 'def step(self, decode_depth: int, last_tokens=None, pf_remaining=None)' in out
    assert 'if self.mode == "lengthgate":' in out
    assert 'def _step_lengthgate' in out
    assert '_pf_remaining' in out and 'pf_remaining=_pf_remaining' in out

def test_patch_idempotent():
    rc1, out1 = _apply(FIXTURE)
    d = os.path.dirname(os.environ["VLLM_SCHED_PATH"])
    open(os.environ["VLLM_SCHED_PATH"], "w").write(out1)
    rc2 = H.main()                       # second apply on already-patched file
    assert rc2 == 0
    assert open(os.environ["VLLM_SCHED_PATH"]).read() == out1   # no further change

def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print(f"ok {f.__name__}")
    print(f"{len(fns)} passed")

if __name__ == "__main__":
    _run()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/test_hotpatch_lengthgate.py`
Expected: FAIL (ImportError: no module `hotpatch_lengthgate`).

- [ ] **Step 3: Implement the patcher** (`scripts/hotpatch_lengthgate.py`):

```python
#!/usr/bin/env python3
"""Add CHUNK_MODE=lengthgate to vLLM's ChunkSizeController: blast a large prefill budget by default,
shrink to a small one whenever a LONG prompt is prefilling AND decoders are present. Length-driven,
never db-driven -- the freeze it protects against exists at any decode depth (no deep batch needed).

Idempotent. Four edits to scheduler.py:
  1. step() signature: add pf_remaining=None
  2. dispatch: route mode=="lengthgate" to _step_lengthgate
  3. method: insert _step_lengthgate before `class Scheduler(SchedulerInterface):`
  4. call site: compute _pf_remaining (max prefill-tokens-remaining over running prefill-chunks +
     waiting) and pass it to step(), so a queued whale is caught BEFORE it is admitted.

Env: LENGTHGATE_THRESHOLD (tok, default 4096), LENGTHGATE_PROTECT (512), LENGTHGATE_BLAST (16384).
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


SIG_ANCHOR = "    def step(self, decode_depth: int, last_tokens=None) -> int:\n"
SIG_NEW = "    def step(self, decode_depth: int, last_tokens=None, pf_remaining=None) -> int:\n"

DISPATCH_ANCHOR = (
    '        if self.mode == "slo":\n'
    '            return self._step_slo(decode_depth)\n'
)
DISPATCH_NEW = (
    '        if self.mode == "lengthgate":\n'
    '            return self._step_lengthgate(decode_depth, pf_remaining)\n'
    '        if self.mode == "slo":\n'
    '            return self._step_slo(decode_depth)\n'
)

METHOD_ANCHOR = "\nclass Scheduler(SchedulerInterface):\n"
METHOD = '''
    def _step_lengthgate(self, decode_depth: int, pf_remaining=None) -> int:
        # Prompt-length gate. Blast a large chunk by default (packs many short prefills / fast whale
        # TTFT when nothing is decoding); shrink to PROTECT the moment a long prompt is prefilling
        # while decoders are live, so the whale is sliced instead of freezing them. Reacts in time
        # because pf_remaining includes WAITING requests -> a queued whale is caught before admission.
        thr = int(os.getenv("LENGTHGATE_THRESHOLD", "4096"))
        protect = int(os.getenv("LENGTHGATE_PROTECT", "512"))
        blast = int(os.getenv("LENGTHGATE_BLAST", "16384"))
        pf = int(pf_remaining or 0)
        if pf > thr and decode_depth > 0:
            self.chunk = int(max(self.min, min(self.max, protect)))
        else:
            self.chunk = int(max(self.min, min(self.max, blast)))
        self._trace(decode_depth, float(pf))   # signal_ms column carries pf_remaining (diagnostics)
        if self._step_count % 50 == 0:
            import logging
            logging.getLogger(__name__).info(
                "ChunkCtrl[lengthgate] step=%d depth=%d pf_remaining=%d chunk=%d",
                self._step_count, decode_depth, pf, self.chunk)
        return self.chunk

'''

CALL_ANCHOR = (
    "            token_budget = self._chunk_ctrl.step(\n"
    '                _decode_depth, getattr(self, "_ff_last_tokens", None))\n'
)
CALL_NEW = (
    "            _pf_remaining = 0\n"
    "            for _r in self.running:\n"
    "                if getattr(_r, 'is_prefill_chunk', False):\n"
    "                    _pf_remaining = max(_pf_remaining, _r.num_prompt_tokens - _r.num_computed_tokens)\n"
    "            for _r in self.waiting:\n"
    "                _pf_remaining = max(_pf_remaining, _r.num_prompt_tokens - _r.num_computed_tokens)\n"
    "            token_budget = self._chunk_ctrl.step(\n"
    '                _decode_depth, getattr(self, "_ff_last_tokens", None), _pf_remaining)\n'
)


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()
    if "def _step_lengthgate" in src:
        print("lengthgate controller already present -- no changes.")
        return 0
    for name, anchor in (("step() signature", SIG_ANCHOR),
                         ("dispatch (mode=='slo')", DISPATCH_ANCHOR),
                         ("class Scheduler anchor", METHOD_ANCHOR),
                         ("controller.step call site", CALL_ANCHOR)):
        if anchor not in src:
            print(f"ERROR: {name} anchor not found", file=sys.stderr)
            return 1
    src = src.replace(SIG_ANCHOR, SIG_NEW, 1)
    src = src.replace(DISPATCH_ANCHOR, DISPATCH_NEW, 1)
    src = src.replace(METHOD_ANCHOR, METHOD + METHOD_ANCHOR, 1)
    src = src.replace(CALL_ANCHOR, CALL_NEW, 1)
    sched.write_text(src)
    print(f"lengthgate controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `python3 tests/test_hotpatch_lengthgate.py`
Expected: PASS (`test_patch_applies_all_four_edits`, `test_patch_idempotent`).

- [ ] **Step 5: Commit**

```bash
git add scripts/hotpatch_lengthgate.py tests/test_hotpatch_lengthgate.py
git commit -m "feat: lengthgate controller hotpatch (prompt-length + decoder-presence budget gate)"
```

---

### Task 4: Per-phase-type analyzer

**Files:**
- Create: `scripts/analyze_lengthgate.py`
- Test: `tests/test_analyze_lengthgate.py`

**Interfaces:**
- Consumes: `parse_phase_schedule`, `phase_type_at`, `token_times` (Task 1 / existing). Reads `logs/*-lgate-b{arm}-t1.jsonl`.
- Produces: `bucket_lengthgate(recs, sched) -> {"S": {...}, "W": {...}}` with per-phase-type `ttft` (list, s), `tbt` (list, ms), `n` (arrivals in phase), `span` (s). Phase type = "S" when frac==0 else "W". TTFT bucketed by arrival phase; TBT by token-emission phase.
- Env for the CLI: `SCHEDULE` (phase schedule string), `ARMS` (default `"16384 512 2048 lengthgate"`), `SLO_TBT_MS` (default 500).

- [ ] **Step 1: Write the failing test** (`tests/test_analyze_lengthgate.py`):

```python
#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from replay_timing import parse_phase_schedule
import analyze_lengthgate as AL

def _rec(start, ttft, tbt_ms):
    lat = ttft + sum(tbt_ms) / 1000.0
    return {"ts": start + lat, "latency": lat, "ttft": ttft, "tbt_ms": tbt_ms}

def test_bucket_splits_S_and_W():
    sched = parse_phase_schedule("100:0.0@10,100:0.5@10")   # S = [0,10), W = [10,20)
    recs = [
        _rec(1.0, 0.20, [30.0, 30.0]),   # arrives in S
        _rec(12.0, 0.40, [400.0, 30.0]), # arrives in W (a frozen decoder tail)
    ]
    b = AL.bucket_lengthgate(recs, sched)
    assert b["S"]["n"] == 1 and b["W"]["n"] == 1
    assert abs(b["S"]["ttft"][0] - 0.20) < 1e-9
    assert 400.0 in b["W"]["tbt"]        # the big gap lands in W

def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print(f"ok {f.__name__}")
    print(f"{len(fns)} passed")

if __name__ == "__main__":
    _run()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/test_analyze_lengthgate.py`
Expected: FAIL (no module `analyze_lengthgate`).

- [ ] **Step 3: Implement the analyzer** (`scripts/analyze_lengthgate.py`):

```python
#!/usr/bin/env python3
"""Per-phase-type analysis for the length-gated dynamic-chunk experiment.
Splits each arm's records into phase types S (whale_frac==0, short burst) and W (whale present),
reconstructing phase from arrival wall-time vs the schedule (no new per-record logging). Reports:
  Phase S: TTFT mean/p95 + throughput      (large chunk should win -> mono ~ lengthgate < 512)
  Phase W: P99/max TBT + request-goodput   (small chunk should win -> 512 ~ lengthgate < 2048 << mono)
Win = lengthgate ~ mono in S AND ~ 512 in W, beating static-2048 in both.

Env: SCHEDULE (required), ARMS (default "16384 512 2048 lengthgate"), SLO_TBT_MS (default 500).
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_timing import parse_phase_schedule, phase_type_at, token_times


def _ptype(sched, t):
    return "S" if phase_type_at(sched, t)[2] == 0.0 else "W"


def bucket_lengthgate(recs, sched):
    starts = [r["ts"] - r["latency"] for r in recs
              if r.get("ts") is not None and r.get("latency") is not None]
    out = {"S": {"ttft": [], "tbt": [], "n": 0, "span": 0.0},
           "W": {"ttft": [], "tbt": [], "n": 0, "span": 0.0}}
    if not starts:
        return out
    t0 = min(starts)
    for r in recs:
        ts, lat, ttft = r.get("ts"), r.get("latency"), r.get("ttft")
        if ts is None or lat is None:
            continue
        start = ts - lat
        pt = _ptype(sched, start - t0)
        out[pt]["n"] += 1
        if ttft is not None:
            out[pt]["ttft"].append(ttft)
        tt = token_times(r)
        for j, ms in enumerate(r.get("tbt_ms") or []):
            emit = tt[j + 1] if (j + 1) < len(tt) else start
            out[_ptype(sched, emit - t0)]["tbt"].append(ms)
    # per-type wall span (for throughput) = arrivals' start range within that type
    for r in recs:
        ts, lat = r.get("ts"), r.get("latency")
        if ts is None or lat is None:
            continue
        start = ts - lat
        pt = _ptype(sched, start - t0)
        out[pt]["span"] = max(out[pt]["span"], start - t0)
    return out


def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def main():
    import statistics as st
    SCHEDULE = os.environ["SCHEDULE"]
    ARMS = os.environ.get("ARMS", "16384 512 2048 lengthgate").split()
    SLO = float(os.environ.get("SLO_TBT_MS", "500"))
    sched = parse_phase_schedule(SCHEDULE)
    print(f"length-gate experiment  schedule={SCHEDULE}  arms={ARMS}  SLO_TBT={SLO:.0f}ms\n")

    def load(arm):
        R = []
        for f in glob.glob(f"logs/*-lgate-b{arm}-t1.jsonl"):
            R += [json.loads(l) for l in open(f) if l.strip()]
        return R

    tbl = {}
    hdr = (f"{'arm':>10} | {'S:TTFTmean':>10} {'S:TTFTp95':>10} {'S:n':>5} | "
           f"{'W:TBTp99':>9} {'W:TBTmax':>9} {'W:gpReq%':>8} {'W:tbtN':>7}")
    print(hdr); print("-" * len(hdr))
    for arm in ARMS:
        recs = load(arm)
        b = bucket_lengthgate(recs, sched)
        S, W = b["S"], b["W"]
        s_ttm = 1000.0 * st.mean(S["ttft"]) if S["ttft"] else float("nan")
        s_p95 = 1000.0 * pctl(S["ttft"], 95) if S["ttft"] else float("nan")
        w_p99 = pctl(W["tbt"], 99)
        w_max = max(W["tbt"]) if W["tbt"] else float("nan")
        # request-goodput in W: fraction of W-arriving requests whose worst gap <= SLO
        w_recs = [r for r in recs if r.get("tbt_ms") and
                  _ptype(sched, (r["ts"] - r["latency"]) - min(x["ts"] - x["latency"] for x in recs
                  if x.get("ts") is not None)) == "W"]
        w_ok = sum(1 for r in w_recs if max(r["tbt_ms"]) <= SLO)
        w_gp = 100.0 * w_ok / len(w_recs) if w_recs else float("nan")
        tbl[arm] = (s_ttm, w_p99, w_gp)
        print(f"{arm:>10} | {s_ttm:>10.0f} {s_p95:>10.0f} {S['n']:>5} | "
              f"{w_p99:>9.1f} {w_max:>9.1f} {w_gp:>8.1f} {len(W['tbt']):>7}")

    print("\n=== verdict (lengthgate should ~match mono in S and ~512 in W, beating 2048 in both) ===")
    if all(a in tbl for a in ("16384", "512", "2048", "lengthgate")):
        lg = tbl["lengthgate"]
        print(f"  Phase S TTFT:  mono={tbl['16384'][0]:.0f}  2048={tbl['2048'][0]:.0f}  "
              f"512={tbl['512'][0]:.0f}  lengthgate={lg[0]:.0f}ms")
        print(f"  Phase W P99TBT: mono={tbl['16384'][1]:.0f}  2048={tbl['2048'][1]:.0f}  "
              f"512={tbl['512'][1]:.0f}  lengthgate={lg[1]:.0f}ms")
        s_win = lg[0] <= tbl["2048"][0] * 1.10
        w_win = lg[1] <= tbl["2048"][1] * 1.10
        print(f"  lengthgate beats/ties 2048 in S (TTFT): {s_win};  in W (P99 TBT): {w_win}  "
              f"-> DYNAMIC WIN: {s_win and w_win}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the test passes**

Run: `python3 tests/test_analyze_lengthgate.py`
Expected: PASS (`test_bucket_splits_S_and_W`).

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_lengthgate.py tests/test_analyze_lengthgate.py
git commit -m "feat: per-phase-type analyzer for length-gate experiment (S:TTFT, W:tail+goodput, verdict)"
```

---

### Task 5: Rate-calibration pilot (box)

**Files:**
- Create: `pilot_lengthgate_rates.sh`

**Interfaces:**
- Consumes: `--phase-schedule` driver (Task 2), the whale workload. Runs static-512 and static-2048 servers at candidate rates.
- Produces: per-phase TTFT/throughput + a saturation flag, so the operator picks `rate_S` (Phase S prefill-bound: 512 pays TTFT vs 2048) and `rate_W` (sub-saturation). Prints a recommended `--phase-schedule`.

- [ ] **Step 1: Write the pilot script** (`pilot_lengthgate_rates.sh`):

```bash
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
```

- [ ] **Step 2: Verify it parses**

Run: `bash -n pilot_lengthgate_rates.sh && echo ok`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add pilot_lengthgate_rates.sh
git commit -m "feat: length-gate rate-calibration pilot (S prefill-bound, both phases sub-saturation)"
```

---

### Task 6: Four-arm orchestrator + box run

**Files:**
- Create: `orchestrate_lengthgate.sh`

**Interfaces:**
- Consumes: all prior tasks. Runs mono/512/2048/lengthgate over the pilot-calibrated `--phase-schedule`; analyzes per phase-type.
- Produces: `logs/lgate_ANALYSIS.txt`, markers `logs/lgate_ALLDONE` / `_FAILED`.

- [ ] **Step 1: Write the orchestrator** (`orchestrate_lengthgate.sh`):

```bash
#!/bin/bash
# orchestrate_lengthgate.sh -- length-gated dynamic chunking, four arms over a non-stationary
# whale-fraction schedule (open-loop, paired arrivals). Arms: mono 16384 / static 512 / static 2048
# / lengthgate (dynamic). Per-phase-type analysis (S: TTFT/throughput, W: tail/goodput). The SCHEDULE
# must be the pilot-calibrated one (Phase S prefill-bound, both phases sub-saturation).
set -uo pipefail
cd /root/pli/vllm-experiment
source scripts/env.sh >/dev/null 2>&1
export PYTHON="$(command -v python)"
MODEL=/data/pli/models/Qwen2.5-Coder-14B-Instruct
DATASET=data/sharegpt_v3.json
STAMP(){ date +%H:%M:%S; }; log(){ echo "[$(STAMP)] $*" >&2; }
kill_ours(){ pkill -TERM -f "venv-vllm023.*api_server" 2>/dev/null; sleep 10
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null | grep -q venv-vllm023 && kill -9 $pid 2>/dev/null; done; sleep 6; }
mkdir -p logs; rm -f logs/lgate_ALLDONE logs/lgate_FAILED
DATE=$(date +%Y-%m-%d)
SCHED=${SCHED:-"12:0.0@45,3:0.2@45"}; DUR=${DUR:-360}   # pilot-calibrated; >=2 full S<->W cycles
THRESH=${THRESH:-4096}; PROTECT=${PROTECT:-512}; BLAST=${BLAST:-16384}
PORT=8050

$PYTHON scripts/hotpatch_hslo.py || { log "PATCH(hslo base) FAILED"; touch logs/lgate_FAILED; exit 1; }
$PYTHON scripts/hotpatch_lengthgate.py || { log "PATCH(lengthgate) FAILED"; touch logs/lgate_FAILED; exit 1; }

log "lengthgate: SCHED='$SCHED' DUR=${DUR}s thresh=$THRESH protect=$PROTECT blast=$BLAST"

run_arm(){ # $1=label $2=mode(static|lengthgate) $3=budget
  local ARM=$1 MODE=$2 BUD=$3 FB="logs/${DATE}-lgate-b$1"
  local EXTRA="DYNAMIC_CHUNK=0"
  if [ "$MODE" = "lengthgate" ]; then
    EXTRA="DYNAMIC_CHUNK=1 CHUNK_MODE=lengthgate DYNAMIC_CHUNK_MIN=$PROTECT DYNAMIC_CHUNK_START=$BLAST LENGTHGATE_THRESHOLD=$THRESH LENGTHGATE_PROTECT=$PROTECT LENGTHGATE_BLAST=$BLAST DYNAMIC_CHUNK_TRACE=${FB}-chunktrace.csv"
    BUD=16384
  fi
  log "  [$ARM] server mode=$MODE budget=$BUD"
  env CUDA_VISIBLE_DEVICES=0,1 PREFIX_REORDER=0 $EXTRA \
    $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT \
    --max-num-seqs 128 --max-num-batched-tokens $BUD --max-model-len 16384 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 > ${FB}-server.log 2>&1 &
  local SV=$!
  for i in $(seq 1 120); do sleep 5; grep -q "Application startup complete" ${FB}-server.log && break
    [ "$i" = 120 ] && { log "  [$ARM] SERVER TIMEOUT"; kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours; touch logs/lgate_FAILED; exit 1; }; done
  $PYTHON src/replay_sharegpt.py --host localhost --port $PORT --model "$MODEL" \
    --dataset "$DATASET" --num-convs 8000 --max-turns 1 --min-turns 1 --max-tokens 256 \
    --phase-schedule "$SCHED" --duration $DUR \
    --pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000 \
    --whale-min-chars 44000 --whale-max-chars 50000 \
    --max-prompt-chars 50000 --pad-seed 1001 \
    --output ${FB}-t1.jsonl > ${FB}.client.log 2>&1 || true
  log "  [$ARM] recs=$(grep -c . ${FB}-t1.jsonl 2>/dev/null || echo 0) preempt=$(grep -c -i preempt ${FB}-server.log 2>/dev/null || echo 0)"
  kill $SV 2>/dev/null; sleep 8; kill -9 $SV 2>/dev/null; kill_ours
}

run_arm 16384      static     16384
run_arm 512        static     512
run_arm 2048       static     2048
run_arm lengthgate lengthgate 16384

log "analyzing per phase-type"
if ! SCHEDULE="$SCHED" ARMS="16384 512 2048 lengthgate" SLO_TBT_MS=500 \
     $PYTHON scripts/analyze_lengthgate.py > logs/lgate_ANALYSIS.txt 2>&1; then
  log "ANALYSIS FAILED"; touch logs/lgate_FAILED; exit 1
fi
echo "[$(STAMP)] DONE" >> logs/lgate_ANALYSIS.txt
touch logs/lgate_ALLDONE
log "done -> logs/lgate_ANALYSIS.txt"
```

Note: the whale-frac for arms comes from the `--phase-schedule` (which forces whales per phase), so the client omits `--whale-frac` (defaults 0, overridden by the forced per-arrival decision). Every arm uses the same `--pad-seed 1001` → identical arrivals + whale positions (paired).

- [ ] **Step 2: Verify it parses**

Run: `bash -n orchestrate_lengthgate.sh && echo ok`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add orchestrate_lengthgate.sh
git commit -m "feat: four-arm length-gate orchestrator (mono/512/2048/lengthgate, per-phase analysis)"
```

- [ ] **Step 4: Ship to box, calibrate, run** (operator step — box execution, after confirming all 8 GPUs idle):

```bash
# from Mac: sync the changed files up
scp scripts/replay_timing.py scripts/hotpatch_lengthgate.py scripts/analyze_lengthgate.py \
    183.147.142.123:/root/pli/vllm-experiment/scripts/
scp src/replay_sharegpt.py 183.147.142.123:/root/pli/vllm-experiment/src/
scp pilot_lengthgate_rates.sh orchestrate_lengthgate.sh 183.147.142.123:/root/pli/vllm-experiment/
# on box: calibrate rates first (pick rate_S so 512's S:TTFT >> 2048's; keep both [ok])
ssh 183.147.142.123 'cd /root/pli/vllm-experiment && nohup bash pilot_lengthgate_rates.sh > logs/lgatepilot_run.log 2>&1 &'
# inspect logs/lgatepilot_run.log; set SCHED accordingly, then run the four arms:
ssh 183.147.142.123 'cd /root/pli/vllm-experiment && SCHED="<calibrated>" DUR=360 nohup bash orchestrate_lengthgate.sh > logs/lgate_run.log 2>&1 &'
# read logs/lgate_ANALYSIS.txt for the verdict.
```

Expected: `logs/lgate_ALLDONE` present; `lgate_ANALYSIS.txt` shows the per-arm phase table and the DYNAMIC WIN line.

---

## Self-Review Notes

- **Spec coverage:** driver phase-schedule (Task 2), lengthgate controller (Task 3), four arms + per-phase analysis (Tasks 4,6), calibration pilot (Task 5), all present. Success criteria encoded in the analyzer verdict.
- **Type consistency:** `parse_phase_schedule` → `[(rate,frac,sec)]` consumed identically by `phase_type_at`, `generate_phase_arrivals`, and the analyzer. `pf_remaining` flows call-site → `step(pf_remaining=)` → `_step_lengthgate`.
- **Paired arms:** arrivals + whale positions come from `generate_phase_arrivals(..., seed=pad_seed)` and `sample_pad_len`'s per-`(seed,ci,turn)` RNG; identical across arms.
- **Known risk (carried from spec):** if the pilot cannot make Phase S prefill-bound while sub-saturation, the short-phase win is genuinely small on this hardware — report honestly, do not force it.
