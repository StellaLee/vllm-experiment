#!/bin/bash
# patches/apply_patches.sh
# Apply vLLM 0.23.0 patches: eviction policy and/or dynamic chunk size controller.
#
# Usage:
#   bash patches/apply_patches.sh              # apply all patches
#   bash patches/apply_patches.sh --eviction-only
#   bash patches/apply_patches.sh --chunk-only
#   bash patches/apply_patches.sh [VLLM_SITE_DIR]
set -e

PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

APPLY_EVICTION=1
APPLY_CHUNK=1

for arg in "$@"; do
  case "$arg" in
    --eviction-only) APPLY_CHUNK=0 ;;
    --chunk-only)    APPLY_EVICTION=0 ;;
    /*)              VLLM_SITE="$arg" ;;
  esac
done

if [ -z "${VLLM_SITE:-}" ]; then
  VLLM_SITE=$($PYTHON -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
fi

VLLM_VERSION=$($PYTHON -c "import vllm; print(vllm.__version__)")
echo "vLLM install : $VLLM_SITE"
echo "vLLM version : $VLLM_VERSION"

if [ "$VLLM_VERSION" != "0.23.0" ]; then
  echo "WARNING: patches were written against vLLM 0.23.0; got $VLLM_VERSION."
fi

PATCH_ROOT="$VLLM_SITE/.."

# ── Eviction policy (kv_cache_utils + block_pool) ────────────────────────────
if [ "$APPLY_EVICTION" = "1" ]; then
  if grep -q 'hit_count' "$VLLM_SITE/v1/core/kv_cache_utils.py" 2>/dev/null; then
    echo "Eviction patches already applied — skipping."
  else
    echo "Applying kv_cache_utils.patch ..."
    patch -p1 -d "$PATCH_ROOT" < "$SCRIPT_DIR/kv_cache_utils.patch"
    echo "Applying block_pool.patch ..."
    patch -p1 -d "$PATCH_ROOT" < "$SCRIPT_DIR/block_pool.patch"
    echo "  -> Eviction policy active. Set EVICTION_POLICY=cf|tdf|lru"
  fi
fi

# ── Dynamic chunk size controller (scheduler) ─────────────────────────────────
if [ "$APPLY_CHUNK" = "1" ]; then
  if grep -q 'ChunkSizeController' "$VLLM_SITE/v1/core/sched/scheduler.py" 2>/dev/null; then
    echo "Chunk size patch already applied — skipping."
  else
    echo "Applying scheduler.patch ..."
    patch -p1 -d "$PATCH_ROOT" < "$SCRIPT_DIR/scheduler.patch"
    echo "  -> Dynamic chunk size active. Set DYNAMIC_CHUNK=1 to enable."
  fi
fi

echo ""
echo "Done. Quick-start examples:"
echo ""
echo "  # CF eviction (recommended for multi-turn)"
echo "  EVICTION_POLICY=cf python -m vllm.entrypoints.openai.api_server \\"
echo "    --model <model> --enable-prefix-caching --port 8000"
echo ""
echo "  # Dynamic chunk size (recommended for mixed prefill/decode load)"
echo "  DYNAMIC_CHUNK=1 DYNAMIC_CHUNK_TARGET=8 \\"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model <model> --port 8000"
echo ""
echo "  # Both combined"
echo "  EVICTION_POLICY=cf DYNAMIC_CHUNK=1 \\"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model <model> --enable-prefix-caching --port 8000"
