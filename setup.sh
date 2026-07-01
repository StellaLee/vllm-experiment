#!/bin/bash
set -e
EXPERIMENT_DIR=/root/vllm-experiment
BURSTGPT_DIR=$EXPERIMENT_DIR/BurstGPT
PYTHON=/root/miniconda3/bin/python3
PIP=/root/miniconda3/bin/pip

echo "=== [1/5] Cloning BurstGPT ==="
if [ ! -d "$BURSTGPT_DIR" ]; then
  git clone https://github.com/HPMLL/BurstGPT "$BURSTGPT_DIR"
else
  echo "Already cloned, skipping."
fi

echo "=== [2/5] Installing Python deps ==="
$PIP install --quiet "aiohttp>=3.8.6" "numpy>=1.25.1" "pandas>=2.2.2" "scipy>=1.14.0" "transformers>=4.41.1"

echo "=== [3/5] Installing BurstGPT package ==="
cd $BURSTGPT_DIR/example && $PIP install --quiet -e . && cd $EXPERIMENT_DIR

echo "=== [4/5] Fetching BurstGPT_1.csv ==="
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
  echo "WARNING: BurstGPT_1.csv not found. run_baseline.sh will use Poisson distribution."
fi

echo "=== [5/5] Fetching ShareGPT V3 dataset ==="
SHAREGPT_DIR=$EXPERIMENT_DIR/data
mkdir -p "$SHAREGPT_DIR"
SHAREGPT=$SHAREGPT_DIR/sharegpt_v3.json
if [ ! -f "$SHAREGPT" ]; then
  echo "Downloading ShareGPT V3 (~90 MB)..."
  wget -q --show-progress \
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" \
    -O "$SHAREGPT" || echo "WARNING: ShareGPT download failed. run_sharegpt.sh will not work."
else
  echo "ShareGPT V3 ready ($(wc -c < $SHAREGPT | awk '{printf \"%.0f MB\", $1/1048576}'))"
fi

echo "=== Setup complete ==="
