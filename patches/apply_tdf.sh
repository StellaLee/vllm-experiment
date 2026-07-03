#!/bin/bash
# apply_tdf.sh — Apply TDF eviction policy patches to an installed vLLM 0.23.0.
#
# Usage:
#   bash patches/apply_tdf.sh [VLLM_SITE_DIR]
#
# VLLM_SITE_DIR defaults to the vllm package location reported by Python.
# Example:
#   bash patches/apply_tdf.sh
#   bash patches/apply_tdf.sh /opt/conda/lib/python3.10/site-packages/vllm
set -e

PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find vllm installation
if [ -n "$1" ]; then
  VLLM_SITE="$1"
else
  VLLM_SITE=$($PYTHON -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
fi

echo "vLLM site dir: $VLLM_SITE"
echo "vLLM version:  $($PYTHON -c 'import vllm; print(vllm.__version__)')"

TARGET_DIR="$VLLM_SITE/v1/core"

if [ ! -d "$TARGET_DIR" ]; then
  echo "ERROR: $TARGET_DIR does not exist. Is vLLM 0.23.0 installed?"
  exit 1
fi

# Check not already applied
if grep -q 'hit_count' "$TARGET_DIR/kv_cache_utils.py" 2>/dev/null; then
  echo "WARNING: TDF patch appears already applied (hit_count found). Skipping."
  exit 0
fi

echo "Applying kv_cache_utils.patch ..."
patch -p1 -d "$VLLM_SITE/.." < "$SCRIPT_DIR/kv_cache_utils.patch"

echo "Applying block_pool.patch ..."
patch -p1 -d "$VLLM_SITE/.." < "$SCRIPT_DIR/block_pool.patch"

echo ""
echo "TDF patches applied successfully."
echo ""
echo "Usage:"
echo "  TDF_EVICTION=1 TDF_LAMBDA=0.1 python -m vllm.entrypoints.api_server ..."
echo ""
echo "Environment variables:"
echo "  TDF_EVICTION  — set to 1 to enable TDF (default: 0 = LRU)"
echo "  TDF_LAMBDA    — decay rate lambda (default: 0.1)"
