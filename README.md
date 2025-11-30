# Viseon - Computer Vision Platform

A comprehensive computer vision platform that provides end-to-end functionality for data management, model training, inference, and tracking. Built as a self-hosted alternative to commercial computer vision services.

## Features

### Core Components
- **Data Management**: Project-based dataset organization with versioning
- **Detections System**: Model-agnostic detection handling and manipulation  
- **Annotation Integration**: CVAT integration for labeling workflows
- **Model Training**: YOLO-based training with experiment tracking
- **Inference Server**: High-performance model serving
- **Object Tracking**: Multi-object tracking algorithms
- **REST API**: Complete API for external integrations

### Supported Workflows
- Upload and organize computer vision datasets
- Create and manage annotation projects
- Train custom object detection models
- Deploy models for real-time inference
- Track objects in video streams
- Integrate with external systems via API

## Quick Start

### Installation
```bash
git clone <repository-url>
cd viseon
pip install -r requirements.txt
```

### Basic Usage
```python
import viseon as vs

# Initialize platform
platform = vs.Viseon()

# Create project
project = platform.create_project(
    name="my_project",
    description="Computer vision project"
)

# Upload data
platform.upload_data("./images", "my_project")

# Get detections
detections = vs.Detections(
    xyxy=boxes,
    confidence=scores,
    class_id=classes
)

# Start training
trainer = vs.YOLOTrainer()
trainer.train("yolov8n", dataset_version="v1.0")

# Deploy model
server = vs.InferenceServer()
server.deploy("./models/best.pt")

# Track objects
tracker = vs.ObjectTracker()
tracks = tracker.track_video("video.mp4", model_path="./models/best.pt")
```

### API Examples

#### Inference API
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Single image prediction
with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict/file", files=files)
    result = response.json()
    print(f"Detections: {len(result['detections'])}")
```

#### Object Tracking
```python
from viseon import Detections, ObjectTracker
import cv2

tracker = ObjectTracker()
video = cv2.VideoCapture("input.mp4")

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    # Get detections (your model inference here)
    detections = your_model_predict(frame)
    
    # Track objects
    tracked = tracker.track_frame(detections)
    
    # Visualize results
    annotated = tracker.visualize(frame, tracked)
    
    cv2.imshow("Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## Architecture

### Project Structure
```
viseon/
├── core/                    # Core components
│   ├── detections.py       # Detection primitives
│   └── project.py          # Project management
├── annotation/             # Annotation tools
│   └── cvat_integration.py # CVAT integration
├── training/               # Training components
│   └── yolo_trainer.py     # YOLO training
├── inference/              # Inference components
│   └── server.py           # Inference server
├── tracking/               # Tracking algorithms
│   └── object_tracker.py   # Object tracking
├── deployment/             # Deployment configs
├── examples/               # Usage examples
└── tests/                  # Test suite
```

### Technology Stack
- **Training**: Ultralytics YOLO, MLflow
- **Inference**: FastAPI, ONNX Runtime
- **Tracking**: ByteTrack, DeepSORT, BoT-SORT
- **Annotation**: CVAT
- **Data Management**: DVC, local filesystem
- **Deployment**: Docker, Kubernetes

## Configuration

### Basic Configuration
```python
config = {
    'storage': {
        'type': 'local',
        'base_path': './data'
    },
    'inference': {
        'port': 8000,
        'model_path': './models',
        'batch_size': 8
    },
    'tracking': {
        'algorithm': 'bytetrack',
        'max_age': 30,
        'min_hits': 3
    }
}
```

### Advanced Configuration
```yaml
# config.yaml
storage:
  type: minio
  base_path: /data
  minio:
    endpoint: minio:9000
    access_key: ${MINIO_ACCESS_KEY}
    secret_key: ${MINIO_SECRET_KEY}

inference:
  port: 8000
  batch_size: 8
  workers: 4
  auto_reload: false

mlflow:
  tracking_uri: postgresql://user:pass@postgres:5432/mlflow

tracking:
  algorithm: bytetrack
  max_age: 60
```

## Deployment

### Docker Deployment
```bash
# Build image
docker build -t viseon .

# Run with docker-compose
docker-compose up -d

# Run inference server
docker run -p 8000:8000 viseon inference
```

### Kubernetes Deployment
```bash
# Apply configuration
kubectl apply -f deployment/k8s/

# Check status
kubectl get pods -n viseon
```

### Production Considerations
- Use MinIO for scalable object storage
- Set up PostgreSQL for MLflow metadata
- Configure GPU resources for training/inference
- Implement proper logging and monitoring
- Set up backup and disaster recovery

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=viseon

# Run specific test
pytest tests/test_detections.py
```

### Code Style
```bash
# Format code
black viseon/

# Lint code
flake8 viseon/

# Type checking
mypy viseon/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## API Reference

### Core Classes

#### Viseon
Main platform class that orchestrates all components.

```python
class Viseon:
    def create_project(name: str, description: str = "")
    def upload_data(data_path: str, project_name: str = None)
    def annotate(subset=None)
    def train(model_type: str = "yolov8n", dataset_version: str = None)
    def deploy(model_path: str = None)
    def track(video_path: str, model_path: str = None)
```

#### Detections
Handles detection data manipulation and format conversion.

```python
class Detections:
    def filter(confidence_threshold=None, class_ids=None)
    def merge(other_detections)
    def iou_with(other_detections)
    def to_yolo(image_shape)
    def to_coco()
    def to_ultralytics()
```

#### Project
Manages datasets and versions.

```python
class Project:
    def upload_data(data_path)
    def create_version(version_name, description="", include_raw=True)
    def get_project_stats()
    def get_versions()
```

### REST API Endpoints

#### Inference Server
- `GET /` - Server information
- `GET /health` - Health check
- `POST /predict` - Single image prediction
- `POST /predict/file` - File upload prediction
- `POST /predict/batch` - Batch prediction
- `GET /model/info` - Model information
- `GET /metrics` - Server metrics

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

**Model Loading Issues**
```python
# Verify model file exists
import os
print(os.path.exists("./models/best.pt"))

# Check model format
from ultralytics import YOLO
model = YOLO("./models/best.pt")
```

**CVAT Integration**
```bash
# Start CVAT server
docker-compose up cvat

# Check CVAT status
curl http://localhost:8080/api/server/info
```

**Performance Issues**
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

- **Documentation**: Check the docs/ directory for detailed guides
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join community discussions for help and ideas
- **Examples**: See examples/ directory for usage patterns

## Acknowledgments

Built using several excellent open-source projects:
- Ultralytics YOLO for object detection
- CVAT for annotation workflows  
- MLflow for experiment tracking
- FastAPI for API development
- viseon library for tracking algorithms

---

**Viseon** - *Empowering computer vision with open-source technology*