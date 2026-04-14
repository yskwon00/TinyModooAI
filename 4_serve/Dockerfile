# Use the official vLLM image (Optimized for NVIDIA GPUs)
FROM vllm/vllm-openai:latest

# Set working directory
WORKDIR /app

# The model files will be mounted to /app/model at runtime
ENV MODEL_PATH=/app/model

# Default entrypoint for vLLM OpenAI API
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
