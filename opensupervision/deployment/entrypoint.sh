#!/bin/bash

# OpenSupervision Entry Point Script
# Handles service initialization and startup

set -e

# Configuration
SERVICE_TYPE=${SERVICE_TYPE:-"inference"}
INFERENCE_PORT=${INFERENCE_PORT:-8000}
INFERENCE_WORKERS=${INFERENCE_WORKERS:-4}
INFERENCE_BATCH_SIZE=${INFERENCE_BATCH_SIZE:-8}
MODEL_PATH=${MODEL_PATH:-"/app/models/best.onnx"}
DATA_PATH=${DATA_PATH:-"/app/data"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$SERVICE_TYPE] $1"
}

# Health check function
health_check() {
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:${INFERENCE_PORT}/health > /dev/null 2>&1; then
            log "Service is healthy"
            return 0
        fi
        
        log "Health check attempt $attempt/$max_attempts failed, waiting..."
        sleep 2
        ((attempt++))
    done
    
    log "Health check failed after $max_attempts attempts"
    return 1
}

# Initialize storage directories
init_storage() {
    log "Initializing storage directories..."
    
    # Create necessary directories
    mkdir -p "${DATA_PATH}"/{raw,processed,versions,annotations,models,logs}
    
    # Set permissions
    chmod -R 755 "${DATA_PATH}"
    
    log "Storage directories initialized"
}

# Initialize database connections
init_databases() {
    log "Initializing database connections..."
    
    # Wait for databases to be ready
    local max_attempts=30
    local attempt=1
    
    # Wait for MongoDB
    while [ $attempt -le $max_attempts ]; do
        if nc -z mongodb 27017 > /dev/null 2>&1; then
            log "MongoDB is ready"
            break
        fi
        
        log "Waiting for MongoDB... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    # Wait for PostgreSQL
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if nc -z postgres 5432 > /dev/null 2>&1; then
            log "PostgreSQL is ready"
            break
        fi
        
        log "Waiting for PostgreSQL... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    # Wait for MinIO
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if nc -z minio 9000 > /dev/null 2>&1; then
            log "MinIO is ready"
            break
        fi
        
        log "Waiting for MinIO... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    log "Database connections initialized"
}

# Load model if exists
load_model() {
    log "Checking for model at: $MODEL_PATH"
    
    if [ -f "$MODEL_PATH" ]; then
        log "Model found: $MODEL_PATH"
        
        # Get model info
        local model_size=$(du -h "$MODEL_PATH" | cut -f1)
        log "Model size: $model_size"
    else
        log "Warning: Model not found at $MODEL_PATH"
        log "Inference will work once a model is provided"
    fi
}

# Start inference server
start_inference() {
    log "Starting inference server..."
    
    # Set environment variables
    export INFERENCE_PORT
    export INFERENCE_WORKERS
    export INFERENCE_BATCH_SIZE
    export MODEL_PATH
    
    # Create model directory if it doesn't exist
    mkdir -p "$(dirname "$MODEL_PATH")"
    
    # Start server with Gunicorn for production
    exec gunicorn \
        --bind 0.0.0.0:${INFERENCE_PORT} \
        --workers ${INFERENCE_WORKERS} \
        --worker-class uvicorn.workers.UvicornWorker \
        --timeout 300 \
        --keep-alive 5 \
        --max-requests 1000 \
        --max-requests-jitter 50 \
        --log-level ${LOG_LEVEL,,} \
        --access-logfile /app/logs/access.log \
        --error-logfile /app/logs/error.log \
        --log-config /app/deployment/logging.conf \
        opensupervision.inference.server:create_app \
        --port ${INFERENCE_PORT}
}

# Start training service
start_training() {
    log "Starting training service..."
    
    # Set training environment
    export CUDA_VISIBLE_DEVICES
    
    # Start training monitor
    exec python -m opensupervision.training.yolo_trainer \
        --host 0.0.0.0 \
        --port 8080 \
        --log-level ${LOG_LEVEL,,}
}

# Development mode
start_development() {
    log "Starting in development mode..."
    
    # Install any development dependencies
    pip install -e .
    
    # Start development server
    if [ "$SERVICE_TYPE" = "inference" ]; then
        uvicorn opensupervision.inference.server:create_app \
            --host 0.0.0.0 \
            --port ${INFERENCE_PORT} \
            --reload \
            --log-level ${LOG_LEVEL,,}
    else
        exec /bin/bash
    fi
}

# Signal handlers for graceful shutdown
trap 'log "Received SIGTERM, shutting down gracefully..."; exit 0' TERM
trap 'log "Received SIGINT, shutting down gracefully..."; exit 0' INT

# Main execution
main() {
    log "Starting OpenSupervision ${SERVICE_TYPE} service..."
    log "Configuration:"
    log "  Service Type: $SERVICE_TYPE"
    log "  Port: $INFERENCE_PORT"
    log "  Workers: $INFERENCE_WORKERS"
    log "  Batch Size: $INFERENCE_BATCH_SIZE"
    log "  Model Path: $MODEL_PATH"
    log "  Data Path: $DATA_PATH"
    log "  Log Level: $LOG_LEVEL"
    
    # Initialize components
    init_storage
    init_databases
    load_model
    
    # Start appropriate service
    case "$SERVICE_TYPE" in
        "inference")
            start_inference
            ;;
        "training")
            start_training
            ;;
        "development")
            start_development
            ;;
        *)
            log "Unknown service type: $SERVICE_TYPE"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"