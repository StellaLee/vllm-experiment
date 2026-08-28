# Congestion-Aware Adaptive Prefill-Length Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, independent congestion gate to the existing `adaptivelpt2` controller so
the step-wide budget degrades toward `static-512`-like behavior when the system is congested,
closing the WRATE=2.0 collapse found in this session's rate sweep without regressing WRATE=1.0's
validated tie-and-match or WRATE=0.5's behavior.

**Architecture:** Extend the single-anchor hotpatch already in `scripts/hotpatch_adaptive_lpt.py`
with a second hysteresis gate on `len(self.running) / max_num_running_reqs` (load as a fraction of
configured concurrency, not an absolute count). When both the existing `pf_remaining` gate
(protect mode) and the new congestion gate are active simultaneously, shrink the step-wide
`token_budget` local variable (not just the per-request threshold) to a small value. Feature-flagged
off by default (`ADAPTIVE_LPT_CONGEST_FRAC=0`) so today's `adaptivelpt2` behavior is preserved
exactly unless explicitly enabled.

**Tech Stack:** Python 3.10, vLLM 0.23.0 (V1 scheduler), bash orchestration scripts, box
`183.147.142.123` (`/root/pli/vllm-experiment`, venv `/root/pli/venv-vllm023`).

## Global Constraints

- Anchor and existing hysteresis block are unchanged; the new block is inserted between the
  existing `self.scheduler_config.long_prefill_token_threshold = _alpt_thr` line and the existing
  `_alpt_trace = os.getenv(...)` line.
- `ADAPTIVE_LPT_CONGEST_FRAC=0` must be a full no-op (byte-identical scheduling decisions to today's
  `adaptivelpt2`) — this is the regression contract the tests must enforce.
- Do not touch `ChunkSizeController`/`DYNAMIC_CHUNK`/`CHUNK_MODE` code paths.
- Workload for every experiment arm must stay byte-for-byte identical to the existing rate-sweep
  arms at the same `WRATE`: `--phase-schedule "6:0.0@60,${WRATE}:0.2@60" --duration 360`,
  `--max-tokens 256`, `--pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000`,
  `--whale-min-chars 44000 --whale-max-chars 50000`, `--max-prompt-chars 50000`,
  `--pad-seed 1001`. Server: `--max-num-seqs 128 --max-num-batched-tokens 16384
  --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.90`.
- `logs/*-lgate-b{arm}-t1.jsonl` naming convention (required by `analyze_lengthgate.py`'s `load()`
  glob) must be preserved for the new arm's output files.

---

### Task 1: Upgrade the hotpatch + tests

**Files:**
- Modify: `scripts/hotpatch_adaptive_lpt.py`
- Modify: `tests/test_hotpatch_adaptive_lpt.py`

**Interfaces:**
- Consumes: nothing new from other tasks; reuses the existing `VLLM_SCHED_PATH` override pattern.
- Produces: `main() -> int` unchanged in signature; the installed block gains the congestion gate.
  Migration-aware: detects the currently-installed hysteresis block and upgrades it in place (same
  pattern the file already uses to upgrade the pre-hysteresis `adaptivelpt` block to `adaptivelpt2`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_hotpatch_adaptive_lpt.py` (alongside the existing hysteresis-block tests):

```python
def test_patch_installs_congestion_block():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert 'ADAPTIVE_LPT_CONGEST_FRAC' in out
    assert '_alpt_in_congest' in out
    idx_thr = out.index('self.scheduler_config.long_prefill_token_threshold = _alpt_thr')
    idx_congest = out.index('ADAPTIVE_LPT_CONGEST_FRAC')
    idx_trace = out.index('_alpt_trace = os.getenv')
    assert idx_thr < idx_congest < idx_trace


def test_patch_migrates_from_hysteresis_block():
    # Apply the OLD hysteresis-only block first (simulating an already-patched box from before
    # this change), then re-run main() and confirm it upgrades in place rather than no-op'ing
    # or erroring.
    rc1, out1 = _apply(FIXTURE)  # installs the NEWEST block directly in a fresh test — also
    # exercise the actual migration path using the file's own pre-upgrade OLD_BLOCK constant:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scheduler.py")
    # Build the fixture with the CURRENT (pre-congestion) hysteresis block already applied,
    # using the module's own HYSTERESIS_BLOCK constant so this test tracks the real prior state.
    open(p, "w").write(FIXTURE.replace(
        "        token_budget = self.max_num_scheduled_tokens\n",
        H.HYSTERESIS_BLOCK))
    os.environ["VLLM_SCHED_PATH"] = p
    rc2 = H.main()
    out2 = open(p).read()
    assert rc2 == 0
    assert 'ADAPTIVE_LPT_CONGEST_FRAC' in out2


def test_patch_idempotent_with_congestion_block():
    rc1, out1 = _apply(FIXTURE)
    open(os.environ["VLLM_SCHED_PATH"], "w").write(out1)
    rc2 = H.main()
    assert rc2 == 0
    assert open(os.environ["VLLM_SCHED_PATH"]).read() == out1
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python3 tests/test_hotpatch_adaptive_lpt.py`
Expected: `test_patch_installs_congestion_block` and `test_patch_migrates_from_hysteresis_block`
FAIL (`AttributeError: module 'hotpatch_adaptive_lpt' has no attribute 'HYSTERESIS_BLOCK'` or
assertion failures); existing hysteresis tests still pass.

- [ ] **Step 3: Rename the current `NEW_BLOCK` constant to `HYSTERESIS_BLOCK`, add `CONGEST_BLOCK`, update `main()`**

In `scripts/hotpatch_adaptive_lpt.py`:
1. Rename the existing `NEW_BLOCK` constant to `HYSTERESIS_BLOCK` (keep `OLD_BLOCK` as-is — the
   pre-hysteresis version, still needed for the two-step migration chain from the very first
   installed version).
2. Add a new `CONGEST_BLOCK` constant:

```python
CONGEST_BLOCK = ANCHOR + '''\
        if os.getenv("ADAPTIVE_LPT"):
            _alpt_pf = 0
            for _r in self.running:
                if getattr(_r, 'is_prefill_chunk', False):
                    _alpt_pf = max(_alpt_pf, _r.num_prompt_tokens - _r.num_computed_tokens)
            for _r in self.waiting:
                _alpt_pf = max(_alpt_pf, _r.num_prompt_tokens - _r.num_computed_tokens)
            _alpt_gate = int(os.getenv("ADAPTIVE_LPT_GATE", "4096"))
            _alpt_exit_gate = int(os.getenv("ADAPTIVE_LPT_EXIT_GATE", "0"))
            _alpt_protect = int(os.getenv("ADAPTIVE_LPT_PROTECT", "512"))
            _alpt_off = int(os.getenv("ADAPTIVE_LPT_OFF", "0"))
            _alpt_was_protect = getattr(self, "_alpt_in_protect", False)
            _alpt_enter = _alpt_pf > _alpt_gate
            _alpt_stay = _alpt_was_protect and _alpt_pf > _alpt_exit_gate
            _alpt_in_protect = _alpt_enter or _alpt_stay
            self._alpt_in_protect = _alpt_in_protect
            _alpt_thr = _alpt_protect if _alpt_in_protect else _alpt_off
            self.scheduler_config.long_prefill_token_threshold = _alpt_thr
            _alpt_congest_frac = float(os.getenv("ADAPTIVE_LPT_CONGEST_FRAC", "0"))
            _alpt_running = len(self.running)
            _alpt_cap = max(1, self.max_num_running_reqs)
            _alpt_load = _alpt_running / _alpt_cap
            _alpt_in_congest = False
            if _alpt_congest_frac > 0:
                _alpt_congest_exit_frac = float(os.getenv("ADAPTIVE_LPT_CONGEST_EXIT_FRAC", str(_alpt_congest_frac)))
                _alpt_was_congest = getattr(self, "_alpt_in_congest", False)
                _alpt_c_enter = _alpt_load > _alpt_congest_frac
                _alpt_c_stay = _alpt_was_congest and _alpt_load > _alpt_congest_exit_frac
                _alpt_in_congest = _alpt_c_enter or _alpt_c_stay
                self._alpt_in_congest = _alpt_in_congest
                if _alpt_in_protect and _alpt_in_congest:
                    _alpt_congest_budget = int(os.getenv("ADAPTIVE_LPT_CONGEST_BUDGET", "512"))
                    token_budget = min(token_budget, _alpt_congest_budget)
            _alpt_trace = os.getenv("ADAPTIVE_LPT_TRACE")
            if _alpt_trace:
                import time as _alpt_time
                with open(_alpt_trace, "a") as _alpt_f:
                    _alpt_f.write(
                        f"{_alpt_time.monotonic()},{_alpt_pf},{_alpt_thr},"
                        f"{_alpt_running},{_alpt_load:.4f},{int(_alpt_in_congest)},{token_budget}\\n")
'''
```

3. Update `main()` to a 3-step migration chain (already-newest / migrate-from-hysteresis /
   migrate-from-old / fresh-install):

```python
def main() -> int:
    sched = _find_sched()
    src = sched.read_text()
    if CONGEST_BLOCK in src:
        print("adaptive-lpt controller (congestion-aware) already present -- no changes.")
        return 0
    if HYSTERESIS_BLOCK in src:
        src = src.replace(HYSTERESIS_BLOCK, CONGEST_BLOCK, 1)
        sched.write_text(src)
        print(f"adaptive-lpt controller upgraded to congestion-aware version in {sched}")
        return 0
    if OLD_BLOCK in src:
        src = src.replace(OLD_BLOCK, CONGEST_BLOCK, 1)
        sched.write_text(src)
        print(f"adaptive-lpt controller upgraded (old->congestion-aware) in {sched}")
        return 0
    if ANCHOR not in src:
        print("ERROR: token_budget anchor not found", file=sys.stderr)
        return 1
    src = src.replace(ANCHOR, CONGEST_BLOCK, 1)
    sched.write_text(src)
    print(f"adaptive-lpt controller installed in {sched}")
    return 0
```

Update the module docstring's `Env:` line to document the three new variables:
`ADAPTIVE_LPT_CONGEST_FRAC` (load-fraction entry threshold, i.e. `len(self.running) /
max_num_running_reqs`, 0 = feature off, default 0), `ADAPTIVE_LPT_CONGEST_EXIT_FRAC`
(load-fraction exit threshold, default = entry value), `ADAPTIVE_LPT_CONGEST_BUDGET` (degraded
step-wide token budget, default 512). Fraction, not an absolute running-sequence count, so the
gate is expressed relative to `--max-num-seqs` and doesn't need recalibrating if that changes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 tests/test_hotpatch_adaptive_lpt.py`
Expected: all tests pass, including the pre-existing hysteresis-block tests (regression check) and
the three new ones.

- [ ] **Step 5: Commit**

```bash
git add scripts/hotpatch_adaptive_lpt.py tests/test_hotpatch_adaptive_lpt.py
git commit -m "feat: add congestion-aware budget degradation to adaptive-lpt controller"
```

---

### Task 2: Calibration pass — find where `running` sits at WRATE=2.0 vs WRATE=0.5/1.0

Not a code task — an instrumented data-collection run, required before picking
`ADAPTIVE_LPT_CONGEST_FRAC`/`EXIT_GATE` values. Do this after Task 1 is committed and synced.

- [ ] **Step 1: Sync the upgraded hotpatch**

```bash
rsync -avz scripts/hotpatch_adaptive_lpt.py 183.147.142.123:/root/pli/vllm-experiment/scripts/hotpatch_adaptive_lpt.py
rsync -avz tests/test_hotpatch_adaptive_lpt.py 183.147.142.123:/root/pli/vllm-experiment/tests/test_hotpatch_adaptive_lpt.py
```

- [ ] **Step 2: Run one instrumented pass per rate point with `ADAPTIVE_LPT_CONGEST_FRAC=0` (feature off, trace-only)**

Reuse the existing `run_adaptive()` shape from `orchestrate/mlsys/rate_sweep_lpt.sh` but only the
`adaptivelpt` arm, one launch per rate, trace enabled (already default). At each of WRATE=0.5, 1.0,
2.0: start the server with `ADAPTIVE_LPT=1 ADAPTIVE_LPT_CONGEST_FRAC=0 ADAPTIVE_LPT_TRACE=<path>`,
replay the same workload as before, then:

```bash
python3 -c "
import csv
rows = list(csv.reader(open('<trace-path>')))
running = [int(r[3]) for r in rows]  # new column index after the upgrade
print('n steps:', len(running))
print('running: min/p50/p90/p99/max:', min(running), sorted(running)[len(running)//2],
      sorted(running)[int(0.9*len(running))], sorted(running)[int(0.99*len(running))], max(running))
"
```

- [ ] **Step 3: Pick `CONGEST_FRAC`/`CONGEST_EXIT_FRAC` from the three distributions**

Expected shape (to confirm, not assume): WRATE=2.0's `running` distribution should sit
substantially higher than WRATE=1.0's, which should sit higher than WRATE=0.5's, with a gap wide
enough to place a gate between "WRATE=1.0's p90" and "WRATE=2.0's p10" without misclassifying
either regime. If the distributions overlap too much to find such a gap, `len(self.running)` is
the wrong proxy — stop and reconsider (see design spec's Non-goals: decode-token volume or
measured step wall-time as fallback proxies), do not force a gate value that doesn't cleanly
separate the two known regimes.

- [ ] **Step 4: Record the chosen values in this plan's findings update (Task 4, Step 3)**

---

### Task 3: Orchestration script for the congestion-aware arm across all 3 rate points

**Files:**
- Create: `orchestrate/mlsys/rate_sweep_lpt_congest.sh`

**Interfaces:**
- Consumes: `scripts/hotpatch_adaptive_lpt.py` (Task 1), `CONGEST_FRAC`/`CONGEST_EXIT_FRAC`
  values chosen in Task 2.
- Produces: `logs/<date>-lgate-badaptivelptcongest{w05,w10,w20}-t1.jsonl` +
  `-chunktrace.csv`, `logs/ratesweep_congest_{w05,w10,w20}_ANALYSIS.txt`.

- [ ] **Step 1: Write the script**

Base it on `orchestrate/mlsys/rate_sweep_lpt_v2.sh` (already parameterized on `GPUS`/`PORT`/
`WRATE`/`TAG`), adding a 5th arm alongside the existing 4 (`run_adaptive` for plain `adaptivelpt`,
plus a new `run_adaptive_congest` for `adaptivelptcongest` using the same server/replay shape but
adding `ADAPTIVE_LPT_CONGEST_FRAC=$CGATE ADAPTIVE_LPT_CONGEST_EXIT_FRAC=$CEXIT
ADAPTIVE_LPT_CONGEST_BUDGET=$CBUDGET` to the env block). `CGATE`/`CEXIT` default to the values
chosen in Task 2 (do not hardcode without the calibration data). `CBUDGET` defaults to 512.
Analysis call extends `ARMS` with the 5th label.

- [ ] **Step 2: Syntax-check**

Run: `bash -n orchestrate/mlsys/rate_sweep_lpt_congest.sh`
Expected: no output (exit 0).

- [ ] **Step 3: Commit**

```bash
git add orchestrate/mlsys/rate_sweep_lpt_congest.sh
git commit -m "feat: add congestion-aware arm to the rate-sweep orchestration"
```

---

### Task 4: Run the 3-rate validation sweep, review traces, update findings

Not a code task — live validation. Do this after Tasks 1-3 are committed and synced.

- [ ] **Step 1: Confirm GPUs idle, sync, launch**

```bash
ssh 183.147.142.123 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
rsync -avz orchestrate/mlsys/rate_sweep_lpt_congest.sh 183.147.142.123:/root/pli/vllm-experiment/orchestrate/mlsys/
ssh 183.147.142.123 "cd /root/pli/vllm-experiment && chmod +x orchestrate/mlsys/rate_sweep_lpt_congest.sh && \
  nohup bash -c 'for w in 0.5 1.0 2.0; do env WRATE=\$w TAG=congest\$(echo \$w | tr -d .) bash orchestrate/mlsys/rate_sweep_lpt_congest.sh; done' > logs/ratesweep_congest_chain.log 2>&1 & disown"
```

- [ ] **Step 2: Wait for all 3 `ratesweep_congest_*_ANALYSIS.txt` files, then review each trace BEFORE reading the analysis tables**

For each rate point, check (mirroring Task 3 Step 3's expected-shape check from the original
`adaptivelpt` design):
- No congestion-gate flapping (count of `in_congest` transitions should be low relative to episode
  length, not toggling every step).
- At WRATE=1.0 specifically: confirm the congestion gate rarely or never fires (`in_congest` mostly
  0) — if it fires here, the WRATE=1.0 regression risk is real and must be resolved before trusting
  the analysis table.

- [ ] **Step 3: Read the analysis tables, compare against success criteria from the design spec**

Compare `adaptivelptcongest`'s row at each rate against: WRATE=2.0 → `static-512`'s S:TTFT
(7574ms)/W:max (397.4ms)/goodput(100%); WRATE=1.0 → `adaptivelpt2`'s validated S:TTFT (≈248ms)/
W:max (≈489ms); WRATE=0.5 → today's `adaptivelptw05` (S:TTFT 284ms/W:max 486.6ms), not required to
improve, just not regress further.

- [ ] **Step 4: Update findings**

Write results into a new findings file (`findings/2026-08-27-congestion-aware-adaptive-lpt.md`,
following the project's existing table/section conventions) — do not overwrite
`2026-07-22-lengthgate-and-per-request-cap.md`, which documents the original (uncalibrated-for-load)
result and should stay as the historical record. Cross-link both directions.

- [ ] **Step 5: Decide paper impact**

If success criteria are met: rewrite `paper-mlsys/tex/sections/057-adaptive-cap.tex`/
`058-tradeoff.tex` around the congestion-aware result as the headline controller claim. If not met
within the time-box (see design spec): keep the plain `adaptivelpt2` result but reframe it
explicitly as a bounded, single-operating-point case study supporting the paper's methodology
argument (mock review Pivot A, `docs/2026-08-27-mock-mlsys-review.md`), and state the load-
sensitivity finding as an honest limitation rather than omitting it.

---

## Self-Review Notes

- **Spec coverage:** Mechanism + migration path (Task 1), calibration requirement (Task 2, per the
  spec's Components item 4 — explicitly not guessing the gate value), orchestration (Task 3),
  success-criteria validation + trace-review discipline + findings/paper update (Task 4).
- **Placeholder scan:** Task 2's exact `CGATE`/`CEXIT` values are intentionally left for the
  calibration data, not guessed — flagged explicitly in both the spec (Components item 4) and here,
  not a silently-skipped TODO.
- **Consistency with prior work's conventions:** migration-aware patcher (matches the file's
  existing pre-hysteresis→hysteresis upgrade pattern), mandatory trace review before trusting
  results (matches the `adaptivelpt`/`lengthgate` bug-discovery discipline), `token_budget =
  min(token_budget, ...)` composability with `_chunk_ctrl` (matches the anchor's existing ordering
  contract), separate findings file rather than overwriting history (matches how this session's own
  rate-sweep work was kept distinct from the original `2026-07-22` finding).
