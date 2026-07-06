#!/bin/bash
# patches/apply_patches.sh
# Apply vLLM 0.23.0 patches to the active vLLM installation.
#
# Usage:
#   bash patches/apply_patches.sh                # apply all
#   bash patches/apply_patches.sh --eviction     # KV-cache eviction policy only
#   bash patches/apply_patches.sh --chunk-size   # dynamic chunk size only
#   bash patches/apply_patches.sh [VLLM_SITE]    # explicit install path
set -e

PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

APPLY_EVICTION=1
APPLY_CHUNK=1

for arg in "$@"; do
  case "$arg" in
    --eviction)    APPLY_CHUNK=0 ;;
    --chunk-size)  APPLY_EVICTION=0 ;;
    /*)            VLLM_SITE="$arg" ;;
  esac
done

if [ -z "${VLLM_SITE:-}" ]; then
  VLLM_SITE=$($PYTHON -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
fi

VLLM_VERSION=$($PYTHON -c "import vllm; print(vllm.__version__)")
PATCH_ROOT="$VLLM_SITE/.."

echo "vLLM install : $VLLM_SITE"
echo "vLLM version : $VLLM_VERSION"
if [ "$VLLM_VERSION" != "0.23.0" ]; then
  echo "WARNING: patches were written against vLLM 0.23.0; got $VLLM_VERSION."
fi

# ── eviction/ ─────────────────────────────────────────────────────────────────
if [ "$APPLY_EVICTION" = "1" ]; then
  if grep -q 'hit_count' "$VLLM_SITE/v1/core/kv_cache_utils.py" 2>/dev/null; then
    echo "[eviction] Already applied — skipping."
  else
    echo "[eviction] Applying kv_cache_utils.patch ..."
    patch -p1 -d "$PATCH_ROOT" < "$SCRIPT_DIR/eviction/kv_cache_utils.patch"
    echo "[eviction] Applying block_pool.patch ..."
    patch -p1 -d "$PATCH_ROOT" < "$SCRIPT_DIR/eviction/block_pool.patch"
    echo "[eviction] Done. Use EVICTION_POLICY=cf|tdf|lru"
  fi
fi

# ── chunk_size/ ───────────────────────────────────────────────────────────────
if [ "$APPLY_CHUNK" = "1" ]; then
  if grep -q 'ChunkSizeController' "$VLLM_SITE/v1/core/sched/scheduler.py" 2>/dev/null; then
    echo "[chunk_size] Already applied — skipping."
  else
    echo "[chunk_size] Applying scheduler.patch ..."
    patch -p1 -d "$PATCH_ROOT" < "$SCRIPT_DIR/chunk_size/scheduler.patch"
    echo "[chunk_size] Done. Use DYNAMIC_CHUNK=1 to enable."
  fi
fi

echo ""
echo "Quick-start:"
echo "  EVICTION_POLICY=cf python -m vllm.entrypoints.openai.api_server \\"
echo "    --model <model> --enable-prefix-caching --port 8000"
echo ""
echo "  DYNAMIC_CHUNK=1 DYNAMIC_CHUNK_TARGET=8 \\"
echo "  python -m vllm.entrypoints.openai.api_server --model <model> --port 8000"
echo ""
echo "  EVICTION_POLICY=cf DYNAMIC_CHUNK=1 \\"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model <model> --enable-prefix-caching --port 8000"
