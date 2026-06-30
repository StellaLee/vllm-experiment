#!/bin/bash
PYTHON=/root/miniconda3/bin/python3
MODEL=/model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct
PORT=8000

echo "Starting vLLM server: model=$MODEL port=$PORT"
echo "Ready when you see 'Application startup complete'."

exec $PYTHON -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --dtype auto \
  --max-model-len 4096
