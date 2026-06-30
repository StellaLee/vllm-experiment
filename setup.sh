#!/bin/bash
set -e
EXPERIMENT_DIR=/root/experiment
BURSTGPT_DIR=$EXPERIMENT_DIR/BurstGPT
PYTHON=/root/miniconda3/bin/python3
PIP=/root/miniconda3/bin/pip

echo "=== [1/4] Cloning BurstGPT ==="
if [ ! -d "$BURSTGPT_DIR" ]; then
  git clone https://github.com/HPMLL/BurstGPT "$BURSTGPT_DIR"
else
  echo "Already cloned, skipping."
fi

echo "=== [2/4] Installing Python deps ==="
$PIP install --quiet "aiohttp>=3.8.6" "numpy>=1.25.1" "pandas>=2.2.2" "scipy>=1.14.0" "transformers>=4.41.1"

echo "=== [3/4] Checking shareGPT.json ==="
SHAREGPT=$BURSTGPT_DIR/example/preprocess_data/shareGPT.json
if [ ! -f "$SHAREGPT" ]; then
  echo "ERROR: shareGPT.json not found at $SHAREGPT"
  exit 1
fi
echo "Found shareGPT.json ($(wc -l < $SHAREGPT) lines)"

echo "=== [4/4] Fetching BurstGPT_1.csv ==="
DATA_DIR=$BURSTGPT_DIR/data
mkdir -p "$DATA_DIR"
CSV=$DATA_DIR/BurstGPT_1.csv
if [ ! -f "$CSV" ]; then
  echo "Attempting git-lfs pull..."
  cd "$BURSTGPT_DIR" && git lfs pull 2>/dev/null && cd - || true
  if [ ! -f "$CSV" ]; then
    echo "LFS pull failed. Downloading from GitHub releases..."
    wget -q --show-progress \
      "https://github.com/HPMLL/BurstGPT/releases/download/v1.0/BurstGPT_1.csv" \
      -O "$CSV" 2>/dev/null || \
    wget -q --show-progress \
      "https://github.com/HPMLL/BurstGPT/raw/main/data/BurstGPT_1.csv" \
      -O "$CSV" 2>/dev/null || true
  fi
fi
if [ -f "$CSV" ]; then
  echo "BurstGPT_1.csv ready ($(wc -l < $CSV) rows)"
else
  echo "WARNING: BurstGPT_1.csv not found. Will use Poisson distribution instead."
fi

echo "=== Setup complete ==="
