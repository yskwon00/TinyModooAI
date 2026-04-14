#!/bin/bash

# 1. Model & Port Selection
echo "=== TinyModooAI Serving Menu ==="
echo "1: [Model] Base (vllm_model_base)"
echo "2: [Model] Chat (vllm_model_chat)"
read -p "Select model (1/2): " m_choice

if [ "$m_choice" == "2" ]; then
    REL_PATH="../outputs/vllm_chat"
    PORT=8001
else
    REL_PATH="../outputs/vllm_base"
    PORT=8000
fi
MODEL_DIR="$(pwd)/$REL_PATH"

echo -e "\n=== Execution Mode ==="
echo "1: Local Run (Native Mac - Direct Python)"
echo "2: Docker Run (Containerized)"
read -p "Select mode (1/2): " e_choice

# 2. Local Run Implementation
if [ "$e_choice" == "1" ]; then
    echo "Starting Local Server (Native Mac)..."
    echo "Make sure vllm is installed: pip install vllm"
    # For Mac, we explicitly set device to cpu for stability
    V_DEVICE="cpu"
    export VLLM_TARGET_DEVICE=$V_DEVICE
    
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_DIR \
        --port $PORT \
        --trust-remote-code \
        --max-model-len 512 \
        --device $V_DEVICE \
        --disable-async-output-proc \
        --dtype float32  # Recommended for CPU stability
    exit 0
fi

# 3. Docker Run Implementation
IMAGE_NAME="tinymodoo-vllm"
TAG="latest"

echo "Building/Checking Docker image..."
docker build -t $IMAGE_NAME:$TAG .

# Detection for Docker
GPU_OPTS=""
V_DEVICE="cpu"
if command -v nvidia-smi &> /dev/null; then
    echo "Detected NVIDIA GPU. Using GPU mode."
    GPU_OPTS="--runtime nvidia --gpus all"
    V_DEVICE="cuda"
else
    echo "Using CPU mode for Container."
    V_DEVICE="cpu"
fi

echo "Starting Docker vLLM server..."
docker run $GPU_OPTS \
    -e VLLM_TARGET_DEVICE=$V_DEVICE \
    -v $MODEL_DIR:/app/model \
    -p $PORT:$PORT \
    --rm \
    $IMAGE_NAME:$TAG \
    --model /app/model \
    --port $PORT \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 512 \
    --device $V_DEVICE \
    --disable-async-output-proc \
    --dtype float32
