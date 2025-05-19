FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    libopencv-dev \
    libgl1 \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for storage and output
RUN mkdir -p storage output

# Download and cache model weights (optional, comment if you want to download on first run)
RUN python3 -c "from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration; \
    AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct-AWQ'); \
    Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct-AWQ')"

# Set up non-root user for better security
RUN groupadd -r ocr && useradd -r -g ocr ocr
RUN chown -R ocr:ocr /app
USER ocr

# Expose API port
EXPOSE 8000

# Start command - can be overridden
CMD ["uvicorn", "api.routes:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative start command for CLI usage:
# ENTRYPOINT ["python3", "main.py"]
