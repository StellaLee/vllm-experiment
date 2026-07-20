#!/usr/bin/env bash
# scripts/env.sh -- activate the vLLM 0.23.0 experiment environment.
#
# Usage:
#   source scripts/env.sh            # activate + brief summary
#   source scripts/env.sh --check    # also verify torch/GPUs and that patches are applied
#
# Override the venv location if it lives elsewhere:
#   VLLM_VENV=/path/to/venv source scripts/env.sh

# ── must be sourced, not executed ────────────────────────────────────────────
# (executing it would activate the venv in a subshell that exits immediately)
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "ERROR: env.sh must be SOURCED, not executed." >&2
    echo "       Run:  source ${0}" >&2
    exit 1
fi

# ── locate repo + venv ───────────────────────────────────────────────────────
ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$ENV_SH_DIR/.." && pwd)"
VLLM_VENV="${VLLM_VENV:-/root/pli/venv-vllm023}"

if [ ! -f "$VLLM_VENV/bin/activate" ]; then
    echo "ERROR: no venv at $VLLM_VENV" >&2
    echo "       Set VLLM_VENV=/path/to/venv, or create it:" >&2
    echo "       uv venv --python 3.10 $VLLM_VENV && \\" >&2
    echo "         source $VLLM_VENV/bin/activate && \\" >&2
    echo "         uv pip install vllm==0.23.0 -r $REPO_ROOT/requirements.txt" >&2
    return 1
fi

source "$VLLM_VENV/bin/activate"

# ── summary ──────────────────────────────────────────────────────────────────
echo "venv   : $VIRTUAL_ENV"
echo "repo   : $REPO_ROOT"
echo "vllm   : $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'NOT IMPORTABLE')"

# ── optional deeper check (slow: imports torch) ──────────────────────────────
if [ "${1:-}" = "--check" ]; then
    python - <<'PY'
import inspect
import torch
import vllm.v1.core.sched.scheduler as s

print(f"torch  : {torch.__version__} | cuda={torch.cuda.is_available()} | gpus={torch.cuda.device_count()}")
src = inspect.getsource(s)
missing = [m for m in ("ChunkSizeController", "_step_slo", "_step_slotail",
                       "PREFIX_REORDER", "AGING_ALPHA") if m not in src]
if missing:
    print(f"patches: MISSING {', '.join(missing)} -- run: bash scripts/apply_patches.sh")
else:
    print("patches: applied (modes: depth | slo | slotail)")
PY
fi

# Reminder of the TP pairing that matters on this box (no NVLink on 4090):
# GPUs (0,1) and (6,7) share a PCIe switch [PIX]; 0-3 vs 4-7 are only NODE-linked.
# e.g.  CUDA_VISIBLE_DEVICES=0,1  vllm serve ... --tensor-parallel-size 2
