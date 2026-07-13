#!/bin/bash
# apply_patches.sh -- apply ALL vLLM scheduler patches in the correct order (idempotent).
# Canonical stack: patch_scheduler.py (base controller + reorder/aging) then
# hotpatch_slo_tail.py (full controller: depth | slo | slotail). The latter SUPERSEDES
# hotpatch_slo_chunk.py -- do not run that one. Restart vLLM after applying.
# See docs/patching.md for the env-var matrix.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/root/miniconda3/bin/python3}

echo "[1/2] base scheduler patch (ChunkSizeController depth + reorder/aging wiring)"
$PYTHON scripts/patch_scheduler.py
echo "[2/2] controller upgrade (adds CHUNK_MODE=slo mean + slotail p99)"
$PYTHON scripts/hotpatch_slo_tail.py

# Verify the full controller is present.
SCHED=$($PYTHON - <<'PY'
import vllm, os
print(os.path.join(os.path.dirname(vllm.__file__), "v1/core/sched/scheduler.py"))
PY
)
if grep -q "_step_slotail" "$SCHED" && grep -q "ChunkSizeController" "$SCHED"; then
    echo "OK: vLLM patched ($SCHED) -- modes: depth | slo | slotail"
else
    echo "ERROR: patch verification failed in $SCHED" >&2; exit 1
fi

cat <<'EOF'

=== Quick configs (env at server launch; see docs/patching.md for the full matrix) ===
  static/mono   DYNAMIC_CHUNK=0 PREFIX_REORDER=0                    --max-num-batched-tokens 16384
  static/chunk  DYNAMIC_CHUNK=0 PREFIX_REORDER=0                    --max-num-batched-tokens 512
  reorder-only  PREFIX_REORDER=1 AGING_ALPHA=0.3 DYNAMIC_CHUNK=0
  chunk depth   DYNAMIC_CHUNK=1 CHUNK_MODE=depth DYNAMIC_CHUNK_HOLD=3
  chunk slo     DYNAMIC_CHUNK=1 CHUNK_MODE=slo     DYNAMIC_CHUNK_MIN=512 DYNAMIC_CHUNK_SLO_MS=50
  chunk p99     DYNAMIC_CHUNK=1 CHUNK_MODE=slotail DYNAMIC_CHUNK_MIN=512 DYNAMIC_CHUNK_SLO_MS=50 DYNAMIC_CHUNK_PCTL=99
Restart vLLM to take effect.
EOF
