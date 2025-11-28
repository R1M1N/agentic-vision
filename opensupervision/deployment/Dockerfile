# OpenSupervision Dockerfile
# Multi-stage build for inference and training services

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.9 -m pip install --upgrade pip

# Create Python virtual environment
RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install ONNX Runtime with CUDA
RUN pip install onnxruntime-gpu

# Install other Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code
WORKDIR /app
COPY . /app/

# Install OpenSupervision in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Set proper permissions
RUN chmod +x /app/deployment/entrypoint.sh

# Copy entrypoint script
COPY deployment/entrypoint.sh /app/
ENTRYPOINT ["/app/deployment/entrypoint.sh"]
CMD ["inference"]

# =============================================================================
# Inference Service Stage
# =============================================================================

FROM base as inference

# Inference-specific configurations
ENV SERVICE_TYPE=inference
ENV INFERENCE_WORKERS=4
ENV INFERENCE_BATCH_SIZE=8

# Install additional inference dependencies
RUN pip install uvicorn[standard] gunicorn

# Expose inference port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "opensupervision.inference.server", "--port", "8000"]

# =============================================================================
# Training Service Stage  
# =============================================================================

FROM base as training

# Training-specific configurations
ENV SERVICE_TYPE=training
ENV CUDA_VISIBLE_DEVICES=0

# Install additional training dependencies
RUN pip install wandb neptune-client tensorboard

# Create training workspace
RUN mkdir -p /app/workspace /app/experiments

# Set working directory for training
WORKDIR /app/workspace

# Expose training port (for monitoring)
EXPOSE 8080

CMD ["python", "-m", "opensupervision.training.yolo_trainer", "--monitor"]

# =============================================================================
# Full Development Stage
# =============================================================================

FROM base as development

# Development dependencies
RUN pip install \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter \
    jupyterlab \
    nbstripout \
    pre-commit

# Install development tools
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set up development environment
RUN python3.9 -m pip install --upgrade pip setuptools wheel

# Install pre-commit hooks
RUN pre-commit install

# Set development mode
ENV DEVELOPMENT_MODE=true

# Default to development shell
CMD ["/bin/bash"]