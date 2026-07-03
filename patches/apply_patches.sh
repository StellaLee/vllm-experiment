#!/bin/bash
# patches/apply_patches.sh
# Apply pluggable KV-cache eviction policy patches to vLLM 0.23.0.
#
# Usage:
#   bash patches/apply_patches.sh [VLLM_SITE_DIR]
#
# VLLM_SITE_DIR is optional — defaults to the location Python reports for vllm.
# Example:
#   bash patches/apply_patches.sh
#   bash patches/apply_patches.sh /opt/conda/lib/python3.10/site-packages/vllm
set -e

PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -n "$1" ]; then
  VLLM_SITE="$1"
else
  VLLM_SITE=$($PYTHON -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
fi

VLLM_VERSION=$($PYTHON -c "import vllm; print(vllm.__version__)")
echo "vLLM install : $VLLM_SITE"
echo "vLLM version : $VLLM_VERSION"

if [ "$VLLM_VERSION" != "0.23.0" ]; then
  echo "WARNING: patches were written against vLLM 0.23.0; got $VLLM_VERSION."
  echo "         Apply may still work but is not guaranteed."
fi

TARGET_DIR="$VLLM_SITE/v1/core"
if [ ! -d "$TARGET_DIR" ]; then
  echo "ERROR: $TARGET_DIR not found. Is vLLM installed correctly?"
  exit 1
fi

if grep -q 'hit_count' "$TARGET_DIR/kv_cache_utils.py" 2>/dev/null; then
  echo "Patches appear already applied (hit_count found in kv_cache_utils.py). Skipping."
  exit 0
fi

# patch -p1 strips the leading a/ or b/ from paths, so run from site-packages parent
PATCH_ROOT="$VLLM_SITE/.."

echo "Applying kv_cache_utils.patch ..."
patch -p1 -d "$PATCH_ROOT" < "$SCRIPT_DIR/kv_cache_utils.patch"

echo "Applying block_pool.patch ..."
patch -p1 -d "$PATCH_ROOT" < "$SCRIPT_DIR/block_pool.patch"

echo ""
echo "Patches applied. Three eviction policies are now available via EVICTION_POLICY:"
echo ""
echo "  EVICTION_POLICY=lru   (default, vLLM built-in LRU — no change)"
echo "  EVICTION_POLICY=cf    (Cascaded Frequency — recommended for multi-turn)"
echo "  EVICTION_POLICY=tdf   (Time-Decayed Frequency; also set TDF_LAMBDA, default 0.1)"
echo ""
echo "Example:"
echo "  EVICTION_POLICY=cf python -m vllm.entrypoints.api_server \\"
echo "    --model <model> --enable-prefix-caching --gpu-memory-utilization 0.7"
