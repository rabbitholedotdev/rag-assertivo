FROM python:3.12-slim
WORKDIR /app
# Sharded ONNX (e5-large, Jina reranker) breaks when huggingface_hub stores
# .onnx and .onnx_data as symlinks under different blob prefixes:
# "External data path escapes model directory". Disabling the shared blob
# store keeps both files as real files in the same snapshot directory.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    FASTEMBED_CACHE_DIR=/models \
    HF_HUB_DISABLE_SHARED_BLOBS=1 \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    COLLECTION_NAME=publico_br
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=10 CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1
CMD ["python", "-m", "rag.mcp_server"]
