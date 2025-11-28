# OpenSupervision: Sovereign Computer Vision Platform

## 🎯 Overview

OpenSupervision is a **complete open-source computer vision platform** that replicates the entire Roboflow ecosystem using only open-source components. Built with data sovereignty and independence in mind, it provides everything needed for computer vision workflows without relying on proprietary cloud services.

## 🚀 Key Features

### Complete Ecosystem Clone
- **Data Management**: Visual dataset curation and management
- **Annotation**: Integrated annotation workflows  
- **Training**: State-of-the-art model training with experiment tracking
- **Inference**: High-performance model serving
- **Tracking**: Advanced object tracking algorithms
- **Analytics**: Zone-based counting and spatial analysis

### Open-Source Architecture
- ✅ **Zero proprietary dependencies**
- ✅ **Complete data sovereignty**
- ✅ **Docker-based deployment**
- ✅ **Microservices architecture**
- ✅ **GPU acceleration support**
- ✅ **Scalable infrastructure**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenSupervision Platform                 │
├─────────────────────────────────────────────────────────────┤
│  SDK Layer (Python API)                                     │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │   Project   │   CVAT      │   YOLO      │ Inference   │  │
│  │ Management  │ Annotation  │  Training   │   Server    │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Core Components                                            │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ Detections  │   Object    │   Video     │  Geometry   │  │
│  │   System    │  Tracking   │ Processing  │   Utils     │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Storage & Versioning                                      │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │   MinIO     │     DVC     │  FiftyOne   │  MongoDB    │  │
│  │ (S3-Compatible)│(Data Version)│(Visual DB)│ (Backend)  │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Services                                                   │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │     CVAT    │   MLflow    │  Streamlit  │   FastAPI   │  │
│  │Annotation   │   Tracking  │   Training  │  Inference  │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Components

### 1. Core SDK (`opensupervision`)

The main Python package that provides a unified interface:

```python
import opensupervision as osv

# Initialize platform
platform = osv.OpenSupervision()

# Create project
project = platform.create_project("my_project", "Computer Vision Project")

# Upload data
platform.upload_data("./images", "my_project")

# Annotate
annotator = platform.annotate()

# Train model
platform.train("yolov8n", dataset_version="v1")

# Deploy inference
platform.deploy("./models/best.pt")

# Track objects
platform.track("video.mp4", "./models/best.pt")
```

### 2. Detections System

Model-agnostic detection handling:

```python
from opensupervision.core import Detections

# Create detections
detections = Detections.from_yolo(yolo_results, (640, 640))
detections = Detections.from_coco(coco_annotations)
detections = Detections.from_torchvision(faster_rcnn_results, (800, 600))

# Filter detections
filtered = detections.filter(confidence_threshold=0.5, class_ids=[0, 1])

# Merge detections
merged = detections1 | detections2
```

### 3. Project Management

Dataset and version management:

```python
from opensupervision.core import Project

# Create project
project = Project("my_dataset", "Custom dataset")

# Upload data
summary = project.upload_data("./images", include_labels=True)

# Create version
version = project.create_version("v1", "Initial dataset")

# Export annotations
export = project.export_annotations("./exports", format="yolo")
```

### 4. Annotation Integration

CVAT integration for annotation workflows:

```python
from opensupervision.annotation import CVATAnnotate

# Initialize CVAT annotator
annotator = CVATAnnotate({
    'url': 'http://localhost:8080',
    'username': 'admin',
    'password': 'password'
})

# Send samples to annotation
task = annotator.send_samples_to_annotation(
    samples=["./images/img1.jpg", "./images/img2.jpg"],
    labels=["person", "car", "bicycle"]
)

# Download annotations
annotations = annotator.download_annotations(task['task_id'])
```

### 5. Training System

YOLO training with experiment tracking:

```python
from opensupervision.training import YOLOTrainer

# Initialize trainer
trainer = YOLOTrainer(config)

# Start training
result = trainer.start_training(
    model_type="yolov8n",
    dataset_path="./dataset",
    epochs=100,
    batch_size=16
)

# Export model
exports = trainer.export_model(
    model_path="./models/best.pt",
    formats=["onnx", "tensorrt"]
)
```

### 6. Inference Server

High-performance model serving:

```python
from opensupervision.inference import InferenceServer

# Initialize server
server = InferenceServer(config)

# Load model
model_info = server.load_model("./models/best.onnx")

# Deploy server
deployment = server.deploy(
    model_path="./models/best.onnx",
    host="0.0.0.0",
    port=8000
)

# Run server
server.run_server(host="0.0.0.0", port=8000)
```

### 7. Object Tracking

Advanced tracking algorithms:

```python
from opensupervision.tracking import ObjectTracker

# Initialize tracker
tracker = ObjectTracker({
    'algorithm': 'bytetrack',
    'max_age': 30,
    'min_hits': 3
})

# Track video
results = tracker.track_video(
    video_path="input.mp4",
    model_path="./models/best.pt",
    output_path="output.mp4"
)
```

## 🛠️ Installation

### Prerequisites

1. **Python 3.8+**
2. **CUDA** (for GPU acceleration, optional)
3. **Docker** and **Docker Compose**
4. **FFmpeg** (for video processing)

### Quick Install

```bash
# Clone repository
git clone https://github.com/your-org/opensupervision.git
cd opensupervision

# Install dependencies
pip install -r requirements.txt

# Optional: Install with GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu

# Optional: Install with all extras
pip install -r requirements.txt[full]
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access services
# - OpenSupervision UI: http://localhost:8501
# - Inference API: http://localhost:8000
# - MLflow Tracking: http://localhost:5000
# - CVAT Annotation: http://localhost:8080
# - FiftyOne App: http://localhost:5151
```

## 🚀 Quick Start

### 1. Basic Workflow

```python
import opensupervision as osv

# Initialize platform
platform = osv.OpenSupervision()

# 1. Create and upload project
project = platform.create_project("my_project", "My CV project")
platform.upload_data("./images", "my_project")

# 2. Annotate data
annotator = platform.annotate()
task = annotator.send_samples_to_annotation(["./images/img1.jpg"], labels=["person", "car"])

# 3. Train model
result = platform.train("yolov8n", dataset_version="v1")

# 4. Deploy for inference
platform.deploy("./models/best.pt")

# 5. Run inference server
platform.server.run_server(port=8000)
```

### 2. Python API Usage

```python
from opensupervision.core import Detections
from opensupervision.inference import ModelLoader

# Load model and run inference
model_loader = ModelLoader()
model = model_loader.load_model("./models/best.pt")

# Your detection code here...
# See examples/ directory for detailed examples
```

### 3. REST API Usage

```python
import requests

# Single image inference
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "image_url": "base64_encoded_image",
        "confidence_threshold": 0.5
    }
)
detections = response.json()

# Batch inference
files = [("files", open("image1.jpg", "rb")), ("files", open("image2.jpg", "rb"))]
response = requests.post("http://localhost:8000/predict/batch", files=files)
results = response.json()
```

## 📁 Project Structure

```
opensupervision/
├── __init__.py                 # Main package entry point
├── requirements.txt           # Dependencies
├── README.md                  # This file
│
├── core/                      # Core components
│   ├── __init__.py
│   ├── detections.py          # Detections system
│   └── project.py             # Project management
│
├── annotation/                # Annotation tools
│   ├── __init__.py
│   └── cvat_integration.py    # CVAT integration
│
├── training/                  # Training system
│   ├── __init__.py
│   └── yolo_trainer.py        # YOLO training
│
├── inference/                 # Inference serving
│   ├── __init__.py
│   └── server.py              # FastAPI server
│
├── tracking/                  # Object tracking
│   ├── __init__.py
│   └── object_tracker.py      # Tracking algorithms
│
├── examples/                  # Usage examples
│   ├── basic_usage.py         # Basic examples
│   ├── api_usage.py           # API examples
│   ├── tracking_example.py    # Tracking example
│   └── deployment_example.py  # Deployment example
│
├── deployment/                # Deployment configs
│   ├── docker-compose.yml     # Docker Compose
│   ├── Dockerfile             # Container config
│   └── nginx.conf             # Nginx config
│
└── tests/                     # Test suite
    ├── test_core.py
    ├── test_inference.py
    └── test_tracking.py
```

## 🔧 Configuration

### Basic Configuration

```python
config = {
    'storage': {
        'type': 'minio',  # or 'local'
        'base_path': './data',
        'minio': {
            'endpoint': 'localhost:9000',
            'access_key': 'minioadmin',
            'secret_key': 'minioadmin'
        }
    },
    'fiftyone': {
        'port': 5151,
        'mongodb': 'mongodb://localhost:27017'
    },
    'cvat': {
        'url': 'http://localhost:8080',
        'username': 'admin',
        'password': 'password'
    },
    'mlflow': {
        'port': 5000,
        'tracking_uri': 'http://localhost:5000'
    },
    'inference': {
        'port': 8000,
        'batch_size': 8
    },
    'tracking': {
        'algorithm': 'bytetrack',
        'max_age': 30
    }
}
```

### Environment Variables

```bash
# Storage
STORAGE_TYPE=minio
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Database
MONGODB_URI=mongodb://localhost:27017

# CVAT
CVAT_URL=http://localhost:8080
CVAT_USERNAME=admin
CVAT_PASSWORD=password

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Inference
INFERENCE_PORT=8000
INFERENCE_BATCH_SIZE=8
```

## 🐳 Deployment

### Docker Compose

The easiest way to deploy the entire platform:

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Kubernetes

For production deployment:

```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/k8s/

# Check deployment status
kubectl get pods -n opensupervision

# Access services
kubectl port-forward svc/inference-service 8000:8000
kubectl port-forward svc/streamlit-ui 8501:8501
```

### Individual Services

Deploy individual components:

```python
# Inference server only
from opensupervision.inference import InferenceServer

server = InferenceServer(config)
server.deploy("./models/best.onnx", port=8000)
server.run_server()
```

## 📊 Monitoring

### Metrics Endpoints

- **Health Check**: `GET /health`
- **Model Info**: `GET /model/info`
- **Metrics**: `GET /metrics`
- **Inference**: `POST /predict`

### Performance Monitoring

```python
# Get server metrics
response = requests.get("http://localhost:8000/metrics")
metrics = response.json()

print(f"Request count: {metrics['request_count']}")
print(f"Average inference time: {metrics['avg_inference_time']:.3f}s")
print(f"Requests per second: {metrics['requests_per_second']:.1f}")
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=opensupervision

# Run specific test
pytest tests/test_core.py::test_detections

# Run performance tests
pytest tests/test_performance.py
```

### Example Tests

```python
# Test detections system
from opensupervision.core import Detections
import numpy as np

# Create test detections
detections = Detections(
    xyxy=np.array([[10, 20, 50, 60], [100, 110, 150, 160]]),
    confidence=np.array([0.8, 0.9]),
    class_id=np.array([0, 1]),
    class_name=["person", "car"]
)

# Test filtering
filtered = detections.filter(confidence_threshold=0.85)
assert len(filtered) == 1

# Test merging
detections2 = Detections(
    xyxy=np.array([[200, 210, 250, 260]]),
    confidence=np.array([0.7]),
    class_id=np.array([2]),
    class_name=["bicycle"]
)

merged = detections | detections2
assert len(merged) == 3
```

## 🔧 Advanced Usage

### Custom Model Integration

```python
from opensupervision.core import Detections

class CustomModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
    
    def predict(self, image):
        # Your custom inference logic
        results = self.model(image)
        return Detections.from_yolo(results, image.shape[:2])

# Use with OpenSupervision
model = CustomModel("./models/custom.pt")
detections = model.predict(image)
```

### Custom Tracking Algorithm

```python
from opensupervision.tracking import ObjectTracker

class CustomTracker:
    def __init__(self):
        self.tracks = {}
    
    def update(self, detections):
        # Your custom tracking logic
        return detections

# Use with OpenSupervision
custom_tracker = CustomTracker()
tracker = ObjectTracker({'algorithm': 'custom'})
```

### Batch Processing

```python
from opensupervision.core import Detections
from concurrent.futures import ThreadPoolExecutor
import cv2

def process_image(image_path):
    # Load and process image
    image = cv2.imread(image_path)
    detections = model.predict(image)
    return detections

# Batch process images
image_paths = [f"./images/img{i}.jpg" for i in range(100)]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, image_paths))
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/opensupervision.git
cd opensupervision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **pytest** for testing

```bash
# Format code
black opensupervision/

# Run linting
flake8 opensupervision/

# Type checking
mypy opensupervision/
```

## 📚 Examples

### Basic Usage
See [examples/basic_usage.py](examples/basic_usage.py) for comprehensive examples.

### API Usage
See [examples/api_usage.py](examples/api_usage.py) for REST API examples.

### Tracking Example
See [examples/tracking_example.py](examples/tracking_example.py) for tracking workflows.

### Deployment Example
See [examples/deployment_example.py](examples/deployment_example.py) for deployment patterns.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Roboflow** - For inspiring the architecture and feature set
- **Ultralytics** - For the excellent YOLO implementation
- **CVAT** - For the comprehensive annotation platform
- **FiftyOne** - For the visual dataset management
- **OpenCV** - For the computer vision foundations
- **FastAPI** - For the high-performance web framework

## 🆘 Support

- **Documentation**: [https://opensupervision.readthedocs.io](https://opensupervision.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-org/opensupervision/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/opensupervision/discussions)
- **Community**: [Discord Server](https://discord.gg/opensupervision)

## 🗺️ Roadmap

- [ ] **v1.1**: Advanced model export formats (TensorRT, OpenVINO)
- [ ] **v1.2**: Real-time streaming inference
- [ ] **v1.3**: Advanced annotation tools integration
- [ ] **v1.4**: Multi-model ensemble support
- [ ] **v1.5**: Edge deployment optimizations
- [ ] **v2.0**: Distributed training support

---

**OpenSupervision** - *Building the future of sovereign computer vision infrastructure* 🚀