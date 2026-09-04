# DRF-Based Power-Aware Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal async reverse-proxy router in front of N vLLM replicas on the
8×4090 server, implementing three request-routing conditions (vLLM-default round-robin,
LMETRIC, and a Dominant-Resource-Fairness power-aware extension) so the ACM e-Energy
routing experiment can be run.

**Architecture:** Pure decision logic (cache-state mirroring, load tracking, ramp-rate math,
the three scoring policies, power-pressure-window classification) lives in small, dependency-free,
fully unit-tested modules under `scripts/eenergy/router/`. A thin aiohttp layer
(`proxy_server.py`) wires that logic to the network: it terminates the OpenAI-compatible
streaming `/v1/completions` request, calls the pure `Router.route()` decision, and forwards
bytes to whichever vLLM replica was chosen — the existing benchmark harness
(`src/replay_sharegpt.py`) needs zero changes, it just points at the proxy's host:port.

**Tech Stack:** Python 3.11, `aiohttp` (proxy server + upstream client), `transformers`
(tokenizer, both already in `requirements.txt`), `pynvml` (GPU power telemetry, isolated to
one module), stdlib `dataclasses`. No new vLLM hotpatches — all new logic lives outside
`scheduler.py`.

**Spec:** `docs/superpowers/specs/2026-08-30-eenergy-routing-design.md`

## Global Constraints

- The proxy MUST be a transparent passthrough for the OpenAI-compatible streaming
  `/v1/completions` API — `src/replay_sharegpt.py` gets zero code changes, only a
  host:port change (spec §3.1).
- No hand-tuned weight anywhere between the three DRF shares (compute/load/power) — each
  share is normalized to its own replica-configured capacity (spec §3.1, §1 "design-space
  dead ends").
- `scripts/pesim/power_logger.py` stays unmodified and is run as an independent sidecar
  during experiments for the ground-truth trace; the router's own live NVML reads (for
  routing decisions) are a separate, in-process concern (spec §3.1 implementation
  components).
- Follow existing repo conventions: `scripts/eenergy/`, `orchestrate/eenergy/`,
  `tests/test_eenergy_*.py` (already scaffolded, currently empty except README stubs).
- Pure logic modules stay dependency-free (stdlib only) so they're unit-testable without
  `aiohttp`/`pynvml`/`transformers` installed — matches this repo's existing convention of
  hotpatch tests needing "no vLLM/box" (see `tests/test_hotpatch_static_reserve.py`).

---

## File Structure

```
scripts/eenergy/router/
  __init__.py            # empty, makes this a package
  replica_state.py       # ReplicaConfig, ReplicaState dataclasses
  cache_mirror.py         # rolling block-hash prefix-cache simulator -> P-token
  load_tracker.py          # in-flight (BS) counters from dispatch/complete events
  ramp.py                   # pure ramp-rate (W/s) math from power samples
  power_nvml.py              # thin NVML read wrapper (only module importing pynvml)
  scoring.py                  # the three routing policies: round_robin, lmetric, drf
  router_core.py               # Router class: ties the above into route()/complete()
  proxy_server.py                # aiohttp app: HTTP wiring + streaming passthrough
scripts/eenergy/
  run_router.py           # CLI entrypoint
  classify_power_windows.py  # power-pressure vs normal window classifier (for eval)
orchestrate/eenergy/
  launch_router_experiment.sh  # launches N replicas + power_logger.py + router
tests/
  test_eenergy_replica_state.py
  test_eenergy_cache_mirror.py
  test_eenergy_load_tracker.py
  test_eenergy_ramp.py
  test_eenergy_power_nvml.py
  test_eenergy_scoring.py
  test_eenergy_router_core.py
  test_eenergy_proxy_server.py
  test_eenergy_classify_power_windows.py
```

`scripts/eenergy/router/*.py` modules are imported directly by test files via `sys.path`
manipulation, matching how `tests/test_hotpatch_static_reserve.py` imports from
`scripts/mlsys/`.

---

### Task 1: Replica config/state dataclasses

**Files:**
- Create: `scripts/eenergy/router/__init__.py` (empty)
- Create: `scripts/eenergy/router/replica_state.py`
- Test: `tests/test_eenergy_replica_state.py`

**Interfaces:**
- Produces: `ReplicaConfig(replica_id: str, host: str, port: int, gpu_index: int, token_budget: int, max_num_seqs: int, ramp_ceiling_w_per_s: float)`, `ReplicaState(config: ReplicaConfig, in_flight: int = 0, cached_block_hashes: set = <factory>, last_power_w: float = 0.0, last_power_ts: float = 0.0, ramp_rate_w_per_s: float = 0.0)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eenergy_replica_state.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from replica_state import ReplicaConfig, ReplicaState  # noqa: E402


def test_replica_config_holds_fields():
    cfg = ReplicaConfig(replica_id="r0", host="127.0.0.1", port=8001, gpu_index=0,
                         token_budget=16384, max_num_seqs=64, ramp_ceiling_w_per_s=100.0)
    assert cfg.replica_id == "r0"
    assert cfg.port == 8001
    assert cfg.ramp_ceiling_w_per_s == 100.0


def test_replica_state_defaults_and_independent_mutable_fields():
    cfg = ReplicaConfig(replica_id="r0", host="h", port=1, gpu_index=0,
                         token_budget=1, max_num_seqs=1, ramp_ceiling_w_per_s=1.0)
    s1 = ReplicaState(config=cfg)
    s2 = ReplicaState(config=cfg)
    assert s1.in_flight == 0
    assert s1.cached_block_hashes == set()
    s1.cached_block_hashes.add(123)
    assert s2.cached_block_hashes == set(), "default set must not be shared across instances"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_eenergy_replica_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'replica_state'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eenergy/router/replica_state.py
"""Per-replica configuration and mutable runtime state for the e-Energy router. Kept as
plain dataclasses with no I/O so every other module in this package can be unit tested
without a real vLLM replica or GPU present."""
from dataclasses import dataclass, field


@dataclass
class ReplicaConfig:
    replica_id: str
    host: str
    port: int
    gpu_index: int
    token_budget: int           # vLLM's max_num_scheduled_tokens for this replica
    max_num_seqs: int           # vLLM's max_num_seqs for this replica
    ramp_ceiling_w_per_s: float  # calibrated ramp-rate ceiling for Share_power normalization


@dataclass
class ReplicaState:
    config: ReplicaConfig
    in_flight: int = 0                                  # BS proxy: dispatched, not yet complete
    cached_block_hashes: set = field(default_factory=set)
    last_power_w: float = 0.0
    last_power_ts: float = 0.0
    ramp_rate_w_per_s: float = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_eenergy_replica_state.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/eenergy/router/__init__.py scripts/eenergy/router/replica_state.py tests/test_eenergy_replica_state.py
git commit -m "eenergy router: add ReplicaConfig/ReplicaState dataclasses"
```

---

### Task 2: Prefix-cache mirror (P-token computation)

**Files:**
- Create: `scripts/eenergy/router/cache_mirror.py`
- Test: `tests/test_eenergy_cache_mirror.py`

**Interfaces:**
- Consumes: nothing from other eenergy modules
- Produces: `BLOCK_SIZE: int`, `block_hashes(token_ids: list[int], block_size: int = BLOCK_SIZE) -> list[int]`, `new_tokens_if_routed(token_ids: list[int], cached_hashes: set[int], block_size: int = BLOCK_SIZE) -> int`, `record_cached(token_ids: list[int], cached_hashes: set[int], block_size: int = BLOCK_SIZE) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eenergy_cache_mirror.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from cache_mirror import block_hashes, new_tokens_if_routed, record_cached, BLOCK_SIZE  # noqa: E402


def test_block_hashes_ignores_partial_trailing_block():
    ids = list(range(BLOCK_SIZE + 3))  # one full block + 3 leftover tokens
    hashes = block_hashes(ids)
    assert len(hashes) == 1


def test_new_tokens_with_empty_cache_is_full_prompt_length():
    ids = list(range(BLOCK_SIZE * 2 + 1))
    assert new_tokens_if_routed(ids, cached_hashes=set()) == len(ids)


def test_record_cached_then_identical_prefix_reduces_new_tokens():
    cache = set()
    first = list(range(BLOCK_SIZE * 2))       # exactly 2 full blocks
    record_cached(first, cache)
    second = first + list(range(BLOCK_SIZE * 2, BLOCK_SIZE * 2 + 5))  # same prefix + 5 more
    # 2 full blocks already cached (2*BLOCK_SIZE tokens); only the trailing 5 are new
    assert new_tokens_if_routed(second, cache) == 5


def test_diverging_prefix_after_first_block_only_credits_matching_prefix():
    cache = set()
    first = list(range(BLOCK_SIZE * 2))
    record_cached(first, cache)
    # second request shares block 0 exactly, diverges inside block 1
    diverged = first[:BLOCK_SIZE] + [-1] * BLOCK_SIZE
    result = new_tokens_if_routed(diverged, cache)
    assert result == BLOCK_SIZE  # only block 1 (the diverging one) is new
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_eenergy_cache_mirror.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cache_mirror'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eenergy/router/cache_mirror.py
"""Per-replica prefix-cache simulator: lets the router compute how many NEW tokens a
request would need if routed to a given replica, without querying that replica -- mirrors
how LMETRIC and prefix-aware routers generally avoid a round-trip per routing decision.

Uses a rolling block-hash chain (hash of block k depends on the hash of block k-1 and
block k's own tokens), the same structural idea vLLM's own prefix-caching uses: a hash
match at block k implies the ENTIRE prefix up to and including block k matches, so the
router only needs to walk forward from block 0 until the first miss."""

BLOCK_SIZE = 16  # tokens per cache block


def block_hashes(token_ids: list[int], block_size: int = BLOCK_SIZE) -> list[int]:
    """Rolling prefix-block hashes. A partial trailing block (fewer than block_size tokens)
    is never included -- it can't be a cache hit target for a later, longer request until
    it's actually completed to a full block by more tokens."""
    hashes = []
    prev = 0
    for start in range(0, len(token_ids), block_size):
        block = tuple(token_ids[start:start + block_size])
        if len(block) < block_size:
            break
        prev = hash((prev, block))
        hashes.append(prev)
    return hashes


def new_tokens_if_routed(token_ids: list[int], cached_hashes: set,
                          block_size: int = BLOCK_SIZE) -> int:
    """How many of token_ids are NOT covered by cached_hashes if routed to the replica that
    owns cached_hashes. Walks the hash chain from the start; stops at the first block whose
    hash isn't in cached_hashes (by construction, everything before that block matched)."""
    hashes = block_hashes(token_ids, block_size)
    matched_blocks = 0
    for h in hashes:
        if h in cached_hashes:
            matched_blocks += 1
        else:
            break
    return len(token_ids) - matched_blocks * block_size


def record_cached(token_ids: list[int], cached_hashes: set,
                   block_size: int = BLOCK_SIZE) -> None:
    """Mark this request's full blocks as cached on the replica. Call once the router has
    dispatched the request there."""
    cached_hashes.update(block_hashes(token_ids, block_size))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_eenergy_cache_mirror.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/eenergy/router/cache_mirror.py tests/test_eenergy_cache_mirror.py
git commit -m "eenergy router: add prefix-cache mirror for P-token computation"
```

---

### Task 3: Load tracker (BS computation)

**Files:**
- Create: `scripts/eenergy/router/load_tracker.py`
- Test: `tests/test_eenergy_load_tracker.py`

**Interfaces:**
- Produces: `LoadTracker` class with `on_dispatch(replica_id: str) -> None`, `on_complete(replica_id: str) -> None`, `in_flight(replica_id: str) -> int`, `in_flight_if_dispatched(replica_id: str) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eenergy_load_tracker.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from load_tracker import LoadTracker  # noqa: E402


def test_unknown_replica_starts_at_zero():
    t = LoadTracker()
    assert t.in_flight("r0") == 0


def test_dispatch_increments_and_complete_decrements():
    t = LoadTracker()
    t.on_dispatch("r0")
    t.on_dispatch("r0")
    assert t.in_flight("r0") == 2
    t.on_complete("r0")
    assert t.in_flight("r0") == 1


def test_complete_never_goes_negative():
    t = LoadTracker()
    t.on_complete("r0")
    assert t.in_flight("r0") == 0


def test_in_flight_if_dispatched_is_whatif_not_mutating():
    t = LoadTracker()
    t.on_dispatch("r0")
    assert t.in_flight_if_dispatched("r0") == 2
    assert t.in_flight("r0") == 1  # the what-if call must not mutate real state


def test_replicas_are_independent():
    t = LoadTracker()
    t.on_dispatch("r0")
    assert t.in_flight("r1") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_eenergy_load_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'load_tracker'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eenergy/router/load_tracker.py
"""Tracks in-flight (running+queued, from the router's own vantage point) request counts
per replica, updated purely from the request lifecycle events the router already observes
(dispatch when it picks a replica, complete when that request's response finishes) -- no
polling of the replica needed. This is LMETRIC's BS term."""


class LoadTracker:
    def __init__(self):
        self._counts: dict = {}

    def on_dispatch(self, replica_id: str) -> None:
        self._counts[replica_id] = self._counts.get(replica_id, 0) + 1

    def on_complete(self, replica_id: str) -> None:
        self._counts[replica_id] = max(0, self._counts.get(replica_id, 0) - 1)

    def in_flight(self, replica_id: str) -> int:
        return self._counts.get(replica_id, 0)

    def in_flight_if_dispatched(self, replica_id: str) -> int:
        """What-if: in-flight count if one more request were dispatched here. Does not
        mutate tracker state -- used to score candidates before a decision is made."""
        return self.in_flight(replica_id) + 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_eenergy_load_tracker.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/eenergy/router/load_tracker.py tests/test_eenergy_load_tracker.py
git commit -m "eenergy router: add LoadTracker for BS (running+queued) computation"
```

---

### Task 4: Ramp-rate math

**Files:**
- Create: `scripts/eenergy/router/ramp.py`
- Test: `tests/test_eenergy_ramp.py`

**Interfaces:**
- Consumes: `ReplicaState` (Task 1) — only reads/writes its `last_power_w`, `last_power_ts`, `ramp_rate_w_per_s` fields
- Produces: `compute_ramp_rate(prev_power_w: float, prev_ts: float, cur_power_w: float, cur_ts: float) -> float`, `update_ramp_state(state: ReplicaState, power_w: float, ts: float) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eenergy_ramp.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from ramp import compute_ramp_rate, update_ramp_state  # noqa: E402
from replica_state import ReplicaConfig, ReplicaState  # noqa: E402


def _state():
    cfg = ReplicaConfig(replica_id="r0", host="h", port=1, gpu_index=0,
                         token_budget=1, max_num_seqs=1, ramp_ceiling_w_per_s=1.0)
    return ReplicaState(config=cfg)


def test_compute_ramp_rate_basic():
    assert compute_ramp_rate(100.0, 0.0, 150.0, 2.0) == 25.0


def test_compute_ramp_rate_zero_or_negative_dt_returns_zero():
    assert compute_ramp_rate(100.0, 5.0, 150.0, 5.0) == 0.0
    assert compute_ramp_rate(100.0, 5.0, 150.0, 4.0) == 0.0


def test_update_ramp_state_first_sample_has_no_prior_so_ramp_is_zero():
    s = _state()
    update_ramp_state(s, power_w=200.0, ts=10.0)
    assert s.ramp_rate_w_per_s == 0.0
    assert s.last_power_w == 200.0
    assert s.last_power_ts == 10.0


def test_update_ramp_state_second_sample_computes_real_ramp():
    s = _state()
    update_ramp_state(s, power_w=200.0, ts=10.0)
    update_ramp_state(s, power_w=210.0, ts=11.0)
    assert s.ramp_rate_w_per_s == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_eenergy_ramp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ramp'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eenergy/router/ramp.py
"""Pure ramp-rate (W/s) arithmetic, kept separate from the actual NVML I/O (power_nvml.py)
so the math is unit-testable without real hardware or pynvml installed."""


def compute_ramp_rate(prev_power_w: float, prev_ts: float,
                       cur_power_w: float, cur_ts: float) -> float:
    """Instantaneous ramp rate between two power samples, W/s. Returns 0.0 for a
    zero/negative interval (out-of-order or duplicate timestamps) rather than dividing by
    zero or reporting a nonsensical sign flip."""
    dt = cur_ts - prev_ts
    if dt <= 0:
        return 0.0
    return (cur_power_w - prev_power_w) / dt


def update_ramp_state(state, power_w: float, ts: float) -> None:
    """Mutates a ReplicaState in place: computes ramp rate from the previous sample (0.0 if
    this is the first sample ever, i.e. last_power_ts is still its default 0.0), then
    stores the new sample as the previous one for next time."""
    if state.last_power_ts > 0:
        state.ramp_rate_w_per_s = compute_ramp_rate(
            state.last_power_w, state.last_power_ts, power_w, ts)
    state.last_power_w = power_w
    state.last_power_ts = ts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_eenergy_ramp.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/eenergy/router/ramp.py tests/test_eenergy_ramp.py
git commit -m "eenergy router: add pure ramp-rate math for Share_power"
```

---

### Task 5: NVML power reader

**Files:**
- Create: `scripts/eenergy/router/power_nvml.py`
- Test: `tests/test_eenergy_power_nvml.py`

**Interfaces:**
- Produces: `NvmlPowerReader(gpu_indices: list[int])` with `.read(gpu_index: int) -> tuple[float, float]` (power_w, timestamp) and `.shutdown() -> None`

**Note:** This is the only module in the package that imports `pynvml`. The test injects a
fake `pynvml` module via `sys.modules` so it runs without real hardware or the `pynvml`
package installed — same reasoning the repo already applies to keep hotpatch tests
box-free.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eenergy_power_nvml.py
import importlib
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))


def _install_fake_pynvml():
    fake = types.ModuleType("pynvml")
    calls = {"init": 0, "shutdown": 0}

    def nvmlInit():
        calls["init"] += 1

    def nvmlDeviceGetHandleByIndex(i):
        return f"handle-{i}"

    def nvmlDeviceGetPowerUsage(handle):
        return 250000  # milliwatts

    def nvmlShutdown():
        calls["shutdown"] += 1

    fake.nvmlInit = nvmlInit
    fake.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
    fake.nvmlDeviceGetPowerUsage = nvmlDeviceGetPowerUsage
    fake.nvmlShutdown = nvmlShutdown
    sys.modules["pynvml"] = fake
    return calls


def test_read_converts_milliwatts_to_watts_and_inits_once():
    calls = _install_fake_pynvml()
    import power_nvml
    importlib.reload(power_nvml)
    reader = power_nvml.NvmlPowerReader([0, 1])
    power_w, ts = reader.read(0)
    assert power_w == 250.0
    assert ts > 0
    assert calls["init"] == 1


def test_shutdown_calls_nvml_shutdown():
    calls = _install_fake_pynvml()
    import power_nvml
    importlib.reload(power_nvml)
    reader = power_nvml.NvmlPowerReader([0])
    reader.shutdown()
    assert calls["shutdown"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_eenergy_power_nvml.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'power_nvml'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eenergy/router/power_nvml.py
"""Thin NVML wrapper -- the only module in this package that imports pynvml, so every
other module (and its tests) stays hardware-free. Mirrors scripts/pesim/power_logger.py's
sampling approach but exposes live per-GPU reads in-process for routing decisions, rather
than only writing a CSV trace. power_logger.py itself stays unmodified and keeps running as
an independent sidecar for the ground-truth experiment trace (see spec S3.1)."""
import time

import pynvml


class NvmlPowerReader:
    def __init__(self, gpu_indices: list[int]):
        pynvml.nvmlInit()
        self._handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices}

    def read(self, gpu_index: int) -> tuple:
        """Returns (power_w, timestamp)."""
        power_w = pynvml.nvmlDeviceGetPowerUsage(self._handles[gpu_index]) / 1000.0
        return power_w, time.time()

    def shutdown(self) -> None:
        pynvml.nvmlShutdown()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_eenergy_power_nvml.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/eenergy/router/power_nvml.py tests/test_eenergy_power_nvml.py
git commit -m "eenergy router: add NVML power reader (only module importing pynvml)"
```

---

### Task 6: Scoring policies (round-robin, LMETRIC, DRF)

**Files:**
- Create: `scripts/eenergy/router/scoring.py`
- Test: `tests/test_eenergy_scoring.py`

**Interfaces:**
- Produces: `Candidate(replica_id: str, new_tokens: int, in_flight_after: int, token_budget: int, max_num_seqs: int, ramp_rate_w_per_s: float, ramp_ceiling_w_per_s: float)`, `pick_round_robin(candidates: list[Candidate], last_index: int) -> tuple[str, int]`, `pick_lmetric(candidates: list[Candidate]) -> str`, `dominant_share(c: Candidate) -> float`, `pick_drf(candidates: list[Candidate]) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eenergy_scoring.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from scoring import Candidate, pick_round_robin, pick_lmetric, dominant_share, pick_drf  # noqa: E402


def _cand(replica_id, new_tokens=0, in_flight_after=1, token_budget=100,
           max_num_seqs=10, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0):
    return Candidate(replica_id, new_tokens, in_flight_after, token_budget,
                      max_num_seqs, ramp_rate_w_per_s, ramp_ceiling_w_per_s)


def test_round_robin_cycles_through_candidates_in_order():
    cands = [_cand("r0"), _cand("r1"), _cand("r2")]
    chosen, idx = pick_round_robin(cands, last_index=-1)
    assert (chosen, idx) == ("r0", 0)
    chosen, idx = pick_round_robin(cands, last_index=idx)
    assert (chosen, idx) == ("r1", 1)
    chosen, idx = pick_round_robin(cands, last_index=idx)
    assert (chosen, idx) == ("r2", 2)
    chosen, idx = pick_round_robin(cands, last_index=idx)
    assert (chosen, idx) == ("r0", 0)  # wraps around


def test_lmetric_picks_minimum_new_tokens_times_in_flight():
    cands = [
        _cand("r0", new_tokens=1000, in_flight_after=5),   # score 5000
        _cand("r1", new_tokens=10, in_flight_after=2),     # score 20 (best)
        _cand("r2", new_tokens=50, in_flight_after=50),    # score 2500
    ]
    assert pick_lmetric(cands) == "r1"


def test_dominant_share_is_max_of_three_normalized_shares():
    # compute-dominant: 80/100=0.8, load 1/10=0.1, power 0/10=0.0
    c = _cand("r0", new_tokens=80, in_flight_after=1, token_budget=100,
              max_num_seqs=10, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0)
    assert dominant_share(c) == 0.8


def test_dominant_share_negative_ramp_does_not_count_as_pressure():
    c = _cand("r0", new_tokens=0, in_flight_after=0, token_budget=100,
              max_num_seqs=10, ramp_rate_w_per_s=-50.0, ramp_ceiling_w_per_s=10.0)
    assert dominant_share(c) == 0.0


def test_drf_picks_replica_with_lowest_dominant_share_even_if_worse_on_other_axes():
    # r0 is best on compute+load but its power ramp is already near the ceiling
    r0 = _cand("r0", new_tokens=1, in_flight_after=1, token_budget=100,
               max_num_seqs=10, ramp_rate_w_per_s=95.0, ramp_ceiling_w_per_s=100.0)  # dom=0.95
    # r1 is worse on compute+load but has power headroom
    r1 = _cand("r1", new_tokens=50, in_flight_after=5, token_budget=100,
               max_num_seqs=10, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)   # dom=0.5
    assert pick_drf([r0, r1]) == "r1"
    # for comparison: LMETRIC (power-blind) would have picked r0 here
    assert pick_lmetric([r0, r1]) == "r0"


def test_pick_functions_raise_on_empty_candidates():
    import pytest
    with pytest.raises(ValueError):
        pick_lmetric([])
    with pytest.raises(ValueError):
        pick_drf([])
    with pytest.raises(ValueError):
        pick_round_robin([], -1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_eenergy_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scoring'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eenergy/router/scoring.py
"""Pure routing-decision functions for the three evaluation conditions (spec S3.3). Each
takes already-computed per-candidate metrics and returns a routing decision -- no I/O, no
network, no tokenizer -- so these are the most heavily unit-tested module in the package;
this is the actual intellectual contribution being evaluated."""
from dataclasses import dataclass


@dataclass
class Candidate:
    replica_id: str
    new_tokens: int           # P-token: new prefill tokens needed if routed here
    in_flight_after: int      # BS: in-flight count if this request were dispatched here
    token_budget: int
    max_num_seqs: int
    ramp_rate_w_per_s: float
    ramp_ceiling_w_per_s: float


def pick_round_robin(candidates: list, last_index: int):
    """Condition 1 (vLLM default -- vLLM ships no built-in multi-replica router, so
    round-robin is the honest content-/state-blind baseline; see spec S3.3). Ignores every
    per-candidate metric; cycles through the given candidate order. Returns (chosen
    replica_id, new last_index to store for the next call)."""
    if not candidates:
        raise ValueError("no candidates to route to")
    next_index = (last_index + 1) % len(candidates)
    return candidates[next_index].replica_id, next_index


def pick_lmetric(candidates: list) -> str:
    """Condition 2. Score = P-token x BS, route to the minimum (Zhang et al., "Simple is
    Better: Multiplication May Be All You Need for LLM Request Scheduling", OSDI'26,
    arXiv:2603.15202)."""
    if not candidates:
        raise ValueError("no candidates to route to")
    best = min(candidates, key=lambda c: c.new_tokens * c.in_flight_after)
    return best.replica_id


def dominant_share(c) -> float:
    """Share_compute, Share_load, Share_power for one candidate, each normalized to that
    replica's own configured capacity (no cross-resource weight); returns the max, i.e. the
    dominant share (spec S3.1). A negative ramp rate (power decreasing) never counts as
    pressure."""
    share_compute = c.new_tokens / c.token_budget
    share_load = c.in_flight_after / c.max_num_seqs
    share_power = max(c.ramp_rate_w_per_s, 0.0) / c.ramp_ceiling_w_per_s
    return max(share_compute, share_load, share_power)


def pick_drf(candidates: list) -> str:
    """Condition 3 (ours). Route to the replica with the lowest resulting dominant share
    across the three independently-normalized resources -- Dominant Resource Fairness
    (Ghodsi et al., NSDI 2011), adapted to an online per-request routing setting (spec
    S3.1, with the honest scope caveat that DRF's original theorem is proven for a static
    allocation game, not this streaming setting)."""
    if not candidates:
        raise ValueError("no candidates to route to")
    best = min(candidates, key=dominant_share)
    return best.replica_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_eenergy_scoring.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/eenergy/router/scoring.py tests/test_eenergy_scoring.py
git commit -m "eenergy router: add round_robin/lmetric/drf scoring policies"
```

---

### Task 7: Router core (orchestration, still pure)

**Files:**
- Create: `scripts/eenergy/router/router_core.py`
- Test: `tests/test_eenergy_router_core.py`

**Interfaces:**
- Consumes: `ReplicaState` (Task 1), `new_tokens_if_routed`/`record_cached` (Task 2), `LoadTracker` (Task 3), `Candidate`/`pick_round_robin`/`pick_lmetric`/`pick_drf` (Task 6)
- Produces: `Router(replica_states: list[ReplicaState], policy: str)` with `.route(token_ids: list[int]) -> str` and `.complete(replica_id: str) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eenergy_router_core.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from router_core import Router  # noqa: E402
from replica_state import ReplicaConfig, ReplicaState  # noqa: E402


def _states():
    cfgs = [
        ReplicaConfig(replica_id="r0", host="h", port=1, gpu_index=0,
                       token_budget=1000, max_num_seqs=10, ramp_ceiling_w_per_s=100.0),
        ReplicaConfig(replica_id="r1", host="h", port=2, gpu_index=1,
                       token_budget=1000, max_num_seqs=10, ramp_ceiling_w_per_s=100.0),
    ]
    return [ReplicaState(config=c) for c in cfgs]


def test_unknown_policy_raises():
    import pytest
    with pytest.raises(ValueError):
        Router(_states(), policy="not_a_real_policy")


def test_round_robin_alternates_across_two_calls():
    r = Router(_states(), policy="round_robin")
    first = r.route(token_ids=[1, 2, 3])
    second = r.route(token_ids=[4, 5, 6])
    assert {first, second} == {"r0", "r1"}
    assert first != second


def test_lmetric_prefers_replica_with_cached_prefix():
    states = _states()
    r = Router(states, policy="lmetric")
    long_prompt = list(range(64))  # 4 full 16-token blocks
    r.route(long_prompt)  # warms whichever replica gets picked first (round-robin tiebreak N/A: lmetric picks by score)
    # both replicas start with empty cache and 0 in-flight, so the first call is a tie;
    # min() picks the first candidate in list order (r0) deterministically
    first_chosen = states  # placeholder to keep flake8 quiet about unused var below
    del first_chosen
    r.complete("r0")
    # route the SAME prompt again: r0 now has it cached (0 new tokens), r1 doesn't
    second = r.route(long_prompt)
    assert second == "r0"


def test_drf_routes_away_from_replica_with_high_ramp_even_if_cache_favors_it():
    states = _states()
    states[0].cached_block_hashes = set()  # r0: no cache advantage
    states[0].ramp_rate_w_per_s = 95.0     # but r0 is near its ramp ceiling (dom share 0.95)
    states[1].ramp_rate_w_per_s = 0.0      # r1 has full power headroom
    r = Router(states, policy="drf")
    chosen = r.route(token_ids=[1, 2, 3])
    assert chosen == "r1"


def test_route_then_complete_updates_load_tracker_round_trip():
    r = Router(_states(), policy="round_robin")
    chosen = r.route(token_ids=[1])
    assert r.load_tracker.in_flight(chosen) == 1
    r.complete(chosen)
    assert r.load_tracker.in_flight(chosen) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_eenergy_router_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'router_core'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eenergy/router/router_core.py
"""Pure orchestration: given already-tokenized request token_ids, decide which replica to
route to under one of the three policies. No network I/O, no tokenizer, no pynvml here --
those live in proxy_server.py / power_nvml.py and are called before/around this class,
keeping Router itself fully unit-testable."""
from cache_mirror import new_tokens_if_routed, record_cached
from load_tracker import LoadTracker
from scoring import Candidate, pick_round_robin, pick_lmetric, pick_drf

_POLICIES = ("round_robin", "lmetric", "drf")


class Router:
    def __init__(self, replica_states: list, policy: str):
        if policy not in _POLICIES:
            raise ValueError(f"unknown policy: {policy!r}, expected one of {_POLICIES}")
        self.replica_states = replica_states
        self.policy = policy
        self.load_tracker = LoadTracker()
        self._rr_index = -1

    def _build_candidates(self, token_ids: list) -> list:
        candidates = []
        for state in self.replica_states:
            new_tokens = new_tokens_if_routed(token_ids, state.cached_block_hashes)
            in_flight_after = self.load_tracker.in_flight_if_dispatched(state.config.replica_id)
            candidates.append(Candidate(
                replica_id=state.config.replica_id,
                new_tokens=new_tokens,
                in_flight_after=in_flight_after,
                token_budget=state.config.token_budget,
                max_num_seqs=state.config.max_num_seqs,
                ramp_rate_w_per_s=state.ramp_rate_w_per_s,
                ramp_ceiling_w_per_s=state.config.ramp_ceiling_w_per_s,
            ))
        return candidates

    def route(self, token_ids: list) -> str:
        candidates = self._build_candidates(token_ids)
        if self.policy == "round_robin":
            replica_id, self._rr_index = pick_round_robin(candidates, self._rr_index)
        elif self.policy == "lmetric":
            replica_id = pick_lmetric(candidates)
        else:
            replica_id = pick_drf(candidates)

        self.load_tracker.on_dispatch(replica_id)
        for state in self.replica_states:
            if state.config.replica_id == replica_id:
                record_cached(token_ids, state.cached_block_hashes)
                break
        return replica_id

    def complete(self, replica_id: str) -> None:
        self.load_tracker.on_complete(replica_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_eenergy_router_core.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/eenergy/router/router_core.py tests/test_eenergy_router_core.py
git commit -m "eenergy router: add Router orchestration class"
```

---

### Task 8: Power-pressure-window classifier

**Files:**
- Create: `scripts/eenergy/classify_power_windows.py`
- Test: `tests/test_eenergy_classify_power_windows.py`

**Interfaces:**
- Produces: `Window(start_ts: float, end_ts: float, label: str)`, `classify_windows(power_samples: list[tuple[float, float]], ramp_ceiling_w_per_s: float, merge_gap_s: float = 1.0) -> list[Window]`, `label_at(windows: list[Window], ts: float) -> str`

This directly implements the spec's §3.2 structural prediction ("report latency metrics
conditioned on power-pressure windows vs. normal windows") as concrete, testable code.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eenergy_classify_power_windows.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy"))
from classify_power_windows import classify_windows, label_at  # noqa: E402


def test_flat_power_trace_is_all_normal():
    samples = [(0.0, 100.0), (1.0, 100.0), (2.0, 100.0), (3.0, 100.0)]
    windows = classify_windows(samples, ramp_ceiling_w_per_s=10.0)
    assert all(w.label == "normal" for w in windows)


def test_sharp_ramp_is_labeled_power_pressure():
    samples = [(0.0, 100.0), (1.0, 100.0), (2.0, 300.0), (3.0, 300.0)]
    windows = classify_windows(samples, ramp_ceiling_w_per_s=10.0)
    labels = [w.label for w in windows]
    assert "power_pressure" in labels
    assert labels[0] == "normal"       # 100->100 over [0,1]
    assert labels[1] == "power_pressure"  # 100->300 over [1,2], ramp=200 > ceiling
    assert labels[2] == "normal"       # 300->300 over [2,3]


def test_adjacent_same_label_windows_are_merged():
    # two consecutive power_pressure intervals back to back should merge into one window
    samples = [(0.0, 0.0), (1.0, 100.0), (2.0, 200.0), (3.0, 200.0)]
    windows = classify_windows(samples, ramp_ceiling_w_per_s=10.0)
    pressure_windows = [w for w in windows if w.label == "power_pressure"]
    assert len(pressure_windows) == 1
    assert pressure_windows[0].start_ts == 0.0
    assert pressure_windows[0].end_ts == 2.0


def test_label_at_finds_covering_window_and_defaults_to_normal():
    samples = [(0.0, 100.0), (1.0, 300.0), (2.0, 300.0)]
    windows = classify_windows(samples, ramp_ceiling_w_per_s=10.0)
    assert label_at(windows, 0.5) == "power_pressure"
    assert label_at(windows, 1.5) == "normal"
    assert label_at(windows, 999.0) == "normal"  # outside any window


def test_fewer_than_two_samples_returns_empty():
    assert classify_windows([], ramp_ceiling_w_per_s=10.0) == []
    assert classify_windows([(0.0, 100.0)], ramp_ceiling_w_per_s=10.0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_eenergy_classify_power_windows.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'classify_power_windows'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eenergy/classify_power_windows.py
"""Classifies time windows in a power trace as 'power_pressure' (ramp rate exceeds the
configured ceiling) or 'normal', for the spec's conditional TTFT/TBT reporting
(docs/superpowers/specs/2026-08-30-eenergy-routing-design.md S3.2): the design predicts
that DRF-3way's latency cost relative to LMETRIC-alone should be concentrated in
power_pressure windows, not spread uniformly."""
from dataclasses import dataclass


@dataclass
class Window:
    start_ts: float
    end_ts: float
    label: str  # "power_pressure" or "normal"


def classify_windows(power_samples: list, ramp_ceiling_w_per_s: float,
                      merge_gap_s: float = 1.0) -> list:
    """power_samples: list of (timestamp, power_w) from a SINGLE replica's trace, sorted by
    timestamp. Computes the ramp rate between each consecutive pair; any interval whose
    |ramp rate| exceeds ramp_ceiling_w_per_s is labeled power_pressure. Adjacent
    same-label intervals within merge_gap_s of each other are merged into one window."""
    if len(power_samples) < 2:
        return []

    raw = []
    for (t0, p0), (t1, p1) in zip(power_samples, power_samples[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        ramp = abs(p1 - p0) / dt
        label = "power_pressure" if ramp > ramp_ceiling_w_per_s else "normal"
        raw.append(Window(t0, t1, label))

    merged = []
    for w in raw:
        if merged and merged[-1].label == w.label and w.start_ts - merged[-1].end_ts <= merge_gap_s:
            merged[-1] = Window(merged[-1].start_ts, w.end_ts, w.label)
        else:
            merged.append(w)
    return merged


def label_at(windows: list, ts: float) -> str:
    """Which window's label covers timestamp ts; 'normal' if none cover it (e.g. before the
    first sample or after the last)."""
    for w in windows:
        if w.start_ts <= ts <= w.end_ts:
            return w.label
    return "normal"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_eenergy_classify_power_windows.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/eenergy/classify_power_windows.py tests/test_eenergy_classify_power_windows.py
git commit -m "eenergy: add power-pressure-window classifier for conditional latency reporting"
```

---

### Task 9: Proxy server (aiohttp wiring)

**Files:**
- Create: `scripts/eenergy/router/proxy_server.py`
- Create: `scripts/eenergy/run_router.py`
- Test: `tests/test_eenergy_proxy_server.py`

**Interfaces:**
- Consumes: `ReplicaConfig`/`ReplicaState` (Task 1), `Router` (Task 7), `NvmlPowerReader` (Task 5), `update_ramp_state` (Task 4)
- Produces: `build_replica_states(replica_specs: list[dict]) -> list[ReplicaState]`, `make_app(states: list[ReplicaState], policy: str, model_name: str) -> aiohttp.web.Application`, `run(replica_specs: list[dict], policy: str, model_name: str, host: str, port: int, power_interval_s: float = 0.5) -> None`

**Note on scope:** `build_replica_states` is pure (plain dataclasses) and gets a real unit
test. The actual HTTP request-handling coroutine (`handle_completions`) needs `aiohttp` and
a live/mocked upstream to exercise meaningfully; this repo's existing convention is to
leave network-facing server/orchestration code (`start_server.sh`,
`scripts/pesim/power_logger.py`) without a pytest file and validate it by running it — Task
10's orchestration script plus the manual smoke-test below follow that same convention
rather than forcing an awkward mock.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eenergy_proxy_server.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from proxy_server import build_replica_states  # noqa: E402


def test_build_replica_states_from_specs():
    specs = [
        dict(replica_id="r0", host="127.0.0.1", port=8001, gpu_index=0,
             token_budget=16384, max_num_seqs=64, ramp_ceiling_w_per_s=100.0),
        dict(replica_id="r1", host="127.0.0.1", port=8002, gpu_index=1,
             token_budget=16384, max_num_seqs=64, ramp_ceiling_w_per_s=100.0),
    ]
    states = build_replica_states(specs)
    assert len(states) == 2
    assert states[0].config.replica_id == "r0"
    assert states[1].config.port == 8002
    assert states[0].in_flight == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_eenergy_proxy_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proxy_server'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eenergy/router/proxy_server.py
"""aiohttp reverse proxy: the router's network-facing service. Wires Router (routing
decisions), NvmlPowerReader (power telemetry), and a tokenizer together, and transparently
forwards the OpenAI-compatible streaming /v1/completions request/response so the existing
benchmark harness (src/replay_sharegpt.py) needs no changes -- it just points at this
proxy's host:port instead of a replica's directly (spec S3.1, Global Constraints)."""
import asyncio

from aiohttp import web, ClientSession, ClientTimeout
from transformers import AutoTokenizer

from replica_state import ReplicaConfig, ReplicaState
from router_core import Router
from power_nvml import NvmlPowerReader
from ramp import update_ramp_state


def build_replica_states(replica_specs: list) -> list:
    """replica_specs: list of dicts with keys matching ReplicaConfig's fields."""
    return [ReplicaState(config=ReplicaConfig(**spec)) for spec in replica_specs]


async def power_poll_loop(states: list, reader: NvmlPowerReader, interval_s: float):
    while True:
        for state in states:
            power_w, ts = reader.read(state.config.gpu_index)
            update_ramp_state(state, power_w, ts)
        await asyncio.sleep(interval_s)


def make_app(states: list, policy: str, model_name: str):
    router = Router(states, policy)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    by_id = {s.config.replica_id: s for s in states}

    async def handle_completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        token_ids = tokenizer.encode(body["prompt"])
        replica_id = router.route(token_ids)
        target = by_id[replica_id].config

        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        url = f"http://{target.host}:{target.port}/v1/completions"
        try:
            async with ClientSession(timeout=ClientTimeout(total=None)) as session:
                async with session.post(url, json=body) as upstream:
                    async for chunk in upstream.content.iter_any():
                        await resp.write(chunk)
        finally:
            router.complete(replica_id)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/completions", handle_completions)
    app["states"] = states
    app["router"] = router
    return app


def run(replica_specs: list, policy: str, model_name: str, host: str, port: int,
        power_interval_s: float = 0.5) -> None:
    states = build_replica_states(replica_specs)
    gpu_indices = [s.config.gpu_index for s in states]
    reader = NvmlPowerReader(gpu_indices)
    app = make_app(states, policy, model_name)

    async def _on_startup(app):
        app["power_task"] = asyncio.create_task(
            power_poll_loop(states, reader, power_interval_s))

    app.on_startup.append(_on_startup)
    web.run_app(app, host=host, port=port)
```

```python
# scripts/eenergy/run_router.py
#!/usr/bin/env python3
"""CLI entrypoint for the e-Energy router. Reads replica topology and policy from
environment variables so it composes cleanly with orchestrate/eenergy/*.sh, matching this
repo's existing env-var-driven convention (see scripts/mlsys/hotpatch_*.py).

Env:
  ROUTER_POLICY        round_robin | lmetric | drf   (required)
  ROUTER_REPLICAS       comma-separated host:port:gpu_index:token_budget:max_num_seqs:ramp_ceiling_w_per_s
                         e.g. "127.0.0.1:8001:0:16384:64:100.0,127.0.0.1:8002:1:16384:64:100.0"
  ROUTER_MODEL_NAME      HF model name/path for the tokenizer (required)
  ROUTER_HOST             default 0.0.0.0
  ROUTER_PORT              default 9000
  ROUTER_POWER_INTERVAL_S   default 0.5
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "router"))
from proxy_server import run  # noqa: E402


def _parse_replicas(spec: str) -> list:
    specs = []
    for i, part in enumerate(spec.split(",")):
        host, port, gpu_index, token_budget, max_num_seqs, ramp_ceiling = part.split(":")
        specs.append(dict(
            replica_id=f"r{i}", host=host, port=int(port), gpu_index=int(gpu_index),
            token_budget=int(token_budget), max_num_seqs=int(max_num_seqs),
            ramp_ceiling_w_per_s=float(ramp_ceiling),
        ))
    return specs


def main() -> int:
    policy = os.environ["ROUTER_POLICY"]
    replica_specs = _parse_replicas(os.environ["ROUTER_REPLICAS"])
    model_name = os.environ["ROUTER_MODEL_NAME"]
    host = os.environ.get("ROUTER_HOST", "0.0.0.0")
    port = int(os.environ.get("ROUTER_PORT", "9000"))
    power_interval_s = float(os.environ.get("ROUTER_POWER_INTERVAL_S", "0.5"))
    run(replica_specs, policy, model_name, host, port, power_interval_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_eenergy_proxy_server.py -v`
Expected: PASS (1 test) — note this specific test only imports `build_replica_states`,
which does not require `aiohttp`/`transformers` to be importable at collection time as long
as Python doesn't eagerly fail on the `from aiohttp import ...` at module top. If the local
environment doesn't have `aiohttp`/`transformers` installed (they are not installed on this
machine as of writing — only on the GPU server's venv, per `requirements.txt`), install
them first: `pip install aiohttp transformers`.

- [ ] **Step 5: Commit**

```bash
git add scripts/eenergy/router/proxy_server.py scripts/eenergy/run_router.py tests/test_eenergy_proxy_server.py
git commit -m "eenergy router: add aiohttp reverse-proxy server and CLI entrypoint"
```

- [ ] **Step 6: Manual smoke test (network-facing code, no pytest coverage — see Note above)**

On the GPU server, with at least one vLLM replica already running on port 8001:

```bash
ROUTER_POLICY=lmetric \
ROUTER_REPLICAS=127.0.0.1:8001:0:16384:64:100.0 \
ROUTER_MODEL_NAME=/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct \
ROUTER_PORT=9000 \
python3 scripts/eenergy/run_router.py &

curl -N -X POST http://127.0.0.1:9000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-0.5B-Instruct","prompt":"Hello","max_tokens":8,"stream":true,"stream_options":{"include_usage":true}}'
```

Expected: an SSE stream of completion chunks identical in shape to what `curl`-ing the
replica on port 8001 directly would produce.

---

### Task 10: Orchestration script

**Files:**
- Create: `orchestrate/eenergy/launch_router_experiment.sh`

No pytest coverage — matches this repo's existing convention that `orchestrate/*.sh`
scripts are integration/experiment scripts, not unit tested (see `orchestrate/mlsys/`,
`orchestrate/pesim/`, none of which have corresponding test files).

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# Launches N vLLM replicas (one GPU each), the power_logger.py sidecar (full trace for
# analysis), and the router (routing decisions for whichever POLICY is selected), then
# waits for the caller to run the benchmark harness against the router's port.
#
# Usage:
#   POLICY=drf N_REPLICAS=4 MODEL=/model/... orchestrate/eenergy/launch_router_experiment.sh
set -euo pipefail

POLICY=${POLICY:?set POLICY=round_robin|lmetric|drf}
N_REPLICAS=${N_REPLICAS:-4}
MODEL=${MODEL:?set MODEL=/path/to/model}
BASE_PORT=${BASE_PORT:-8001}
ROUTER_PORT=${ROUTER_PORT:-9000}
TOKEN_BUDGET=${TOKEN_BUDGET:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
RAMP_CEILING_W_PER_S=${RAMP_CEILING_W_PER_S:?set RAMP_CEILING_W_PER_S (calibrated per spec open item)}
POWER_TRACE=${POWER_TRACE:-logs/eenergy_power_trace_${POLICY}.csv}

REPLICA_SPECS=""
PIDS=()
for i in $(seq 0 $((N_REPLICAS - 1))); do
  port=$((BASE_PORT + i))
  echo "Launching replica $i on GPU $i, port $port"
  CUDA_VISIBLE_DEVICES=$i python3 -m vllm.entrypoints.api_server \
    --model "$MODEL" --port "$port" --dtype auto \
    --max-num-batched-tokens "$TOKEN_BUDGET" --max-num-seqs "$MAX_NUM_SEQS" &
  PIDS+=($!)
  if [ -z "$REPLICA_SPECS" ]; then
    REPLICA_SPECS="127.0.0.1:${port}:${i}:${TOKEN_BUDGET}:${MAX_NUM_SEQS}:${RAMP_CEILING_W_PER_S}"
  else
    REPLICA_SPECS="${REPLICA_SPECS},127.0.0.1:${port}:${i}:${TOKEN_BUDGET}:${MAX_NUM_SEQS}:${RAMP_CEILING_W_PER_S}"
  fi
done

echo "Waiting for replicas to come up..."
for i in $(seq 0 $((N_REPLICAS - 1))); do
  port=$((BASE_PORT + i))
  until curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do sleep 2; done
done

echo "Starting power_logger.py sidecar -> ${POWER_TRACE}"
GPU_LIST=$(seq -s, 0 $((N_REPLICAS - 1)))
python3 scripts/pesim/power_logger.py --gpus "$GPU_LIST" --interval-ms 50 --output "$POWER_TRACE" &
POWER_PID=$!

echo "Starting router (policy=$POLICY) on port $ROUTER_PORT"
ROUTER_POLICY="$POLICY" ROUTER_REPLICAS="$REPLICA_SPECS" ROUTER_MODEL_NAME="$MODEL" \
  ROUTER_PORT="$ROUTER_PORT" python3 scripts/eenergy/run_router.py &
ROUTER_PID=$!

echo "Router ready on port ${ROUTER_PORT}. Point the benchmark harness at it, e.g.:"
echo "  python3 src/replay_sharegpt.py --host 127.0.0.1 --port ${ROUTER_PORT} ..."
echo "Press Ctrl+C to tear down replicas, power logger, and router."

trap 'kill "${PIDS[@]}" "$POWER_PID" "$ROUTER_PID" 2>/dev/null' EXIT
wait "$ROUTER_PID"
```

- [ ] **Step 2: Make it executable and syntax-check**

Run: `chmod +x orchestrate/eenergy/launch_router_experiment.sh && bash -n orchestrate/eenergy/launch_router_experiment.sh`
Expected: no output (syntax OK)

- [ ] **Step 3: Commit**

```bash
git add orchestrate/eenergy/launch_router_experiment.sh
git commit -m "eenergy: add orchestration script to launch replicas + power logger + router"
```

---

## Self-Review

**Spec coverage:**
- §3.1 shared machinery (P-token cache mirror, live BS tracking, NVML power sampling) →
  Tasks 2, 3, 5.
- §3.1 three-share DRF mechanism → Task 6 (`dominant_share`, `pick_drf`).
- §3.1 "no vLLM hotpatches required, router-only" → satisfied by construction (no file
  under this plan touches `scheduler.py`).
- §3.3 three evaluation conditions → Task 6 (`pick_round_robin`, `pick_lmetric`,
  `pick_drf`) + Task 7 (`Router` selecting among them via `policy`).
- §3.1 "transparent reverse proxy, harness needs zero changes" → Task 9
  (`proxy_server.py` speaks the same `/v1/completions` SSE contract
  `src/replay_sharegpt.py` already POSTs to).
- §3.2 "report latency conditioned on power-pressure vs. normal windows" → Task 8
  (`classify_power_windows.py`).
- Execution steps 4–8 (launch replicas, build shared machinery, implement conditions 1–3,
  run workload) → Tasks 1–7, 10.
- Execution steps 2, 3 (ramp-ceiling calibration, cross-request prefix-sharing check) are
  explicitly *not* code tasks in the spec's own "Open items" section — reflected here as
  a required env var (`RAMP_CEILING_W_PER_S`) with no default in Task 10, forcing the
  calibration decision to be made before a real run rather than silently defaulted.
- Execution steps 9–11 (aggregate power analysis, Monte Carlo feed-in, replication) are
  empirical/analysis work using this infrastructure, not additional router code — out of
  this plan's scope by the spec's own structure (analysis scripts already exist in
  `scripts/pesim/` and are reused unmodified per the Global Constraints).

**Placeholder scan:** no TBD/TODO/"add error handling" left in any step; every code block
is complete and runnable as shown.

**Type consistency check:** `ReplicaConfig`/`ReplicaState` (Task 1) fields match their use
in `ramp.py` (Task 4: `last_power_w`, `last_power_ts`, `ramp_rate_w_per_s`),
`router_core.py` (Task 7: `config.replica_id`, `config.token_budget`, `config.max_num_seqs`,
`config.ramp_ceiling_w_per_s`, `cached_block_hashes`), and `proxy_server.py` (Task 9:
`ReplicaConfig(**spec)`). `Candidate`'s fields (Task 6) match exactly what
`router_core.Router._build_candidates` constructs (Task 7). `Router.route()`/`.complete()`
signatures (Task 7) match how `proxy_server.handle_completions` calls them (Task 9).

---

**Plan complete and saved to `docs/superpowers/plans/2026-08-30-eenergy-routing-implementation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
