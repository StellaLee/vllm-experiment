# Patching vLLM: one command + the config matrix

We modify vLLM 0.23.0's scheduler (`vllm/v1/core/sched/scheduler.py`) to add a
prefill-chunk controller + warmth reordering, all **env-gated** (no behaviour change unless
you set the env vars). This doc is the single source of truth for how to apply the patches
and which env vars select which configuration.

## TL;DR

```bash
bash scripts/apply_patches.sh          # idempotent; applies everything in the right order
# then launch vLLM with the env vars for the config you want (see matrix below), e.g.
DYNAMIC_CHUNK=1 CHUNK_MODE=slotail DYNAMIC_CHUNK_MIN=512 DYNAMIC_CHUNK_PCTL=99 \
  python -m vllm.entrypoints.openai.api_server --model ... --max-num-batched-tokens 8192
```

Restart vLLM after patching. Re-running the script is safe (each patcher is idempotent).

## Which patch path to use (there are two — use scripts/)

- **`scripts/apply_patches.sh` (imperative Python patchers) — CANONICAL, current.** Applies
  the full controller (`depth | slo | slotail`) + reorder/aging that all recent experiments
  use. Use this.
- **`patches/apply_patches.sh` (unified `.patch` diffs) — STALE, do not use.** It predates
  the SLO/`slotail` controller (the committed `patches/**/*.patch` contain only the base
  depth-mode chunk controller + reorder + eviction). Kept for provenance only.

Regenerating a single up-to-date unified diff from the fully-patched `scheduler.py` (the
"cleaner artifact to ship") is a deferred TODO; until then, `scripts/apply_patches.sh` is the
source of truth.

## What gets patched (order matters)

`apply_patches.sh` runs two patchers in sequence:

1. **`patch_scheduler.py`** — installs the base `ChunkSizeController` class (depth mode),
   wires it into the scheduler, and adds warmth-reorder + soft-aging blocks.
2. **`hotpatch_slo_tail.py`** — **replaces** the controller class with the full version
   supporting three modes: `depth | slo | slotail`. **This supersedes the older
   `hotpatch_slo_chunk.py`** (which only added `slo`); do not run that one — `slo_tail`
   includes it. `hotpatch_slo_chunk.py` is kept only for provenance of earlier findings.

So the canonical stack is just: `patch_scheduler.py` → `hotpatch_slo_tail.py`.

## Env-var reference

**Base (from `patch_scheduler.py`):**
| var | default | meaning |
|---|---|---|
| `DYNAMIC_CHUNK` | `0` | master enable for the chunk controller (0 = static budget) |
| `PREFIX_REORDER` | `0` | enable warmth reordering (cached-prefix requests first) |
| `AGING_ALPHA` | `0.3` | soft-aging weight (anti-starvation for reordering) |
| `AGING_THRESHOLD_MS` | `inf` | age at which a cold request is force-promoted |
| `DYNAMIC_CHUNK_TARGET` | `8` | depth-mode target decode depth |
| `DYNAMIC_CHUNK_HOLD` | `3` | depth-mode hysteresis (steps before acting) |
| `DYNAMIC_CHUNK_MIN` | `256` | floor budget (use `512` = stall-free floor) |

**Controller (from `hotpatch_slo_tail.py`):**
| var | default | mode | meaning |
|---|---|---|---|
| `CHUNK_MODE` | `depth` | — | `depth` \| `slo` (EMA/mean) \| `slotail` (windowed p99) |
| `DYNAMIC_CHUNK_SLO_MS` | `50` | slo, slotail | target per-step (≈per-token) latency |
| `DYNAMIC_CHUNK_EMA` | `0.3` | slo | EMA weight for the mean signal |
| `DYNAMIC_CHUNK_STEP` | `=MIN` | slo, slotail | AIMD additive-increase step (tokens) |
| `DYNAMIC_CHUNK_PCTL` | `99` | slotail | percentile of the step-latency window |
| `DYNAMIC_CHUNK_WINDOW` | `128` | slotail | window size (steps) |
| `DYNAMIC_CHUNK_WINMIN` | `20` | slotail | min samples before acting |

**vLLM native (CLI flags, not env):** `--max-num-batched-tokens N` sets the **ceiling** for
the controller, or the **static budget** when `DYNAMIC_CHUNK=0`. Also `--max-model-len`,
`--max-num-seqs`, `--tensor-parallel-size`.

## Configuration matrix (the experiment arms)

| Config | Env vars | `--max-num-batched-tokens` |
|---|---|---|
| **static / mono** (run-to-completion) | `DYNAMIC_CHUNK=0 PREFIX_REORDER=0` | large (e.g. 8192/16384) |
| **static / chunk** (PS frontier) | `DYNAMIC_CHUNK=0 PREFIX_REORDER=0` | small (512) |
| **baseline** | `DYNAMIC_CHUNK=0 PREFIX_REORDER=0` | 2048 |
| **reorder-only** | `PREFIX_REORDER=1 AGING_ALPHA=0.3 DYNAMIC_CHUNK=0` | 2048 |
| **chunk-only (depth)** | `DYNAMIC_CHUNK=1 CHUNK_MODE=depth DYNAMIC_CHUNK_HOLD=3 PREFIX_REORDER=0` | 2048 |
| **chunk SLO (mean)** — blind | `DYNAMIC_CHUNK=1 CHUNK_MODE=slo DYNAMIC_CHUNK_MIN=512 DYNAMIC_CHUNK_SLO_MS=50` | 8192 (ceiling) |
| **chunk SLO (p99)** — the fix | `DYNAMIC_CHUNK=1 CHUNK_MODE=slotail DYNAMIC_CHUNK_MIN=512 DYNAMIC_CHUNK_SLO_MS=50 DYNAMIC_CHUNK_PCTL=99` | 8192 (ceiling) |
| **combined** (reorder + adaptive chunk) | `PREFIX_REORDER=1 AGING_ALPHA=0.3 DYNAMIC_CHUNK=1 CHUNK_MODE=slo DYNAMIC_CHUNK_MIN=512` | 8192 (ceiling) |

Notes:
- **Reorder needs cacheable prefixes to do anything** — with unique/uncacheable padding
  (Cs² sweeps, sensitive regime) there is no warmth, so keep `PREFIX_REORDER=0` there.
- Controller budget is logged at INFO (`ChunkCtrl[slo]` / `ChunkCtrl[slotail]` lines with
  `chunk=`), so you can read the budget trajectory from the server log.
