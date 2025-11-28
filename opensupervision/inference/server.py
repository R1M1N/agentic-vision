"""
Inference Server - replaces roboflow-python deploy functionality
High-performance serving using FastAPI and ONNX Runtime
"""

import os
import json
import time
import base64
import asyncio
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ..core.detections import Detections


class InferenceRequest(BaseModel):
    """Inference request model"""
    image_url: Optional[str] = None
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    max_detections: int = 300
    augment: bool = False


class InferenceResponse(BaseModel):
    """Inference response model"""
    detections: List[Dict]
    inference_time: float
    image_shape: Tuple[int, int]
    model_info: Dict
    batch_id: Optional[str] = None


class InferenceServer:
    """
    High-Performance Inference Server
    Serves models using FastAPI and ONNX Runtime
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Inference Server
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.inference_config = config.get('inference', {})
        
        # Server state
        self.app = None
        self.onnx_session = None
        self.model_info = {}
        self.model_path = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        self.output_shape = None
        
        # Performance tracking
        self.request_count = 0
        self.total_inference_time = 0
        self.batch_queue = asyncio.Queue()
        self.batch_processor_running = False
        
        # Model metadata
        self.class_names = []
        self.num_classes = 0
        
    def load_model(self, model_path: str) -> Dict:
        """
        Load ONNX model for inference
        
        Args:
            model_path: Path to ONNX model file
            
        Returns:
            Model loading information
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load ONNX session with optimizations
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if not self._cuda_available():
                providers.remove('CUDAExecutionProvider')
            
            self.onnx_session = ort.InferenceSession(
                model_path,
                providers=providers
            )
            
            # Get model info
            self.model_path = model_path
            self.input_name = self.onnx_session.get_inputs()[0].name
            self.output_names = [output.name for output in self.onnx_session.get_outputs()]
            self.input_shape = self.onnx_session.get_inputs()[0].shape
            self.output_shape = self.onnx_session.get_outputs()[0].shape
            
            # Detect model type and extract metadata
            self._extract_model_metadata()
            
            return {
                'status': 'loaded',
                'model_path': model_path,
                'input_shape': self.input_shape,
                'output_shape': self.output_shape,
                'input_name': self.input_name,
                'output_names': self.output_names,
                'providers': self.onnx_session.get_providers(),
                'class_names': self.class_names,
                'num_classes': self.num_classes
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _extract_model_metadata(self):
        """Extract metadata from model"""
        # This is a simplified implementation
        # Real implementation would parse ONNX model to extract class names
        
        # For YOLO models, try to extract from output shape
        if len(self.output_shape) == 3:  # YOLO format: [batch, num_detections, attributes]
            # YOLOv8 format typically has 84 attributes (4 bbox + 80 classes for COCO)
            # YOLOv5 format has 85 attributes (4 bbox + 80 classes + 1 objectness)
            
            if self.output_shape[2] == 84:
                # YOLOv8 COCO
                self.num_classes = 80
                self.class_names = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                    'toothbrush'
                ]
            else:
                # Generic detection model
                num_attrs = self.output_shape[2]
                # Assume 4 bbox coords + remaining as classes + confidence
                self.num_classes = num_attrs - 5  # 4 bbox + 1 confidence
                self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        else:
            self.num_classes = 1
            self.class_names = ["object"]
    
    def preprocess_image(self, image: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for model input
        
        Args:
            image: Input image array
            target_size: Target image size
            
        Returns:
            Preprocessed image and original shape
        """
        original_shape = image.shape[:2]
        
        # Letterbox resize to maintain aspect ratio
        h, w = original_shape
        scale = target_size / max(h, w)
        
        if scale < 1.0:  # Only resize if image is larger than target
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w
        
        # Calculate padding
        pad_h = (target_size - h) // 2
        pad_w = (target_size - w) // 2
        
        # Create letterboxed image
        letterboxed = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        letterboxed[pad_h:pad_h + h, pad_w:pad_w + w] = image
        
        # Normalize and convert to model input format
        input_tensor = letterboxed.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        return input_tensor, original_shape
    
    def postprocess_predictions(self,
                               predictions: np.ndarray,
                               original_shape: Tuple[int, int],
                               confidence_threshold: float = 0.5,
                               iou_threshold: float = 0.5,
                               max_detections: int = 300) -> Detections:
        """
        Post-process model predictions
        
        Args:
            predictions: Raw model predictions
            original_shape: Original image shape
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections
            
        Returns:
            Processed Detections object
        """
        # YOLOv8 postprocessing
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension
        
        # Transpose if needed (for YOLO format)
        if predictions.shape[1] == 85 or predictions.shape[1] == 84:
            predictions = predictions.T
            predictions = predictions.T
        
        # Process predictions
        num_detections = 0
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # Process each row of predictions
        for pred in predictions:
            if len(pred) < 5:  # Skip invalid predictions
                continue
            
            # Extract prediction components
            if len(pred) == 85:  # YOLOv5 format
                x, y, w, h, obj_score = pred[:5]
                class_scores = pred[5:]
            elif len(pred) == 84:  # YOLOv8 format
                x, y, w, h = pred[:4]
                class_scores = pred[4:]
                obj_score = 1.0  # YOLOv8 doesn't have objectness separately
            else:
                continue
            
            # Skip if no valid class predictions
            if len(class_scores) == 0:
                continue
            
            # Find best class and its score
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            confidence = obj_score * class_score
            
            # Apply confidence threshold
            if confidence < confidence_threshold:
                continue
            
            # Convert center format to corner format
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(confidence)
            all_classes.append(class_id)
            
            num_detections += 1
            if num_detections >= max_detections:
                break
        
        if not all_boxes:
            return Detections.empty()
        
        # Convert to numpy arrays
        boxes = np.array(all_boxes)
        scores = np.array(all_scores)
        classes = np.array(all_classes)
        
        # Apply Non-Maximum Suppression
        if len(boxes) > 1:
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                confidence_threshold,
                iou_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = boxes[indices]
                scores = scores[indices]
                classes = classes[indices]
        
        # Scale boxes back to original image size
        original_h, original_w = original_shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * original_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * original_h
        
        # Clip to image boundaries
        boxes[:, 0] = np.clip(boxes[:, 0], 0, original_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, original_h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, original_w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, original_h - 1)
        
        # Get class names
        class_names = [self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}" 
                      for cls_id in classes]
        
        return Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=classes,
            class_name=class_names
        )
    
    def predict_single(self,
                      image: Union[str, np.ndarray, bytes],
                      confidence_threshold: float = 0.5,
                      iou_threshold: float = 0.5,
                      max_detections: int = 300) -> Dict:
        """
        Run inference on single image
        
        Args:
            image: Image as path, array, or base64 bytes
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold
            max_detections: Maximum detections
            
        Returns:
            Prediction results
        """
        start_time = time.time()
        
        # Load and preprocess image
        if isinstance(image, str):
            # Image path
            image_array = cv2.imread(image)
            if image_array is None:
                raise ValueError(f"Could not load image: {image}")
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        elif isinstance(image, bytes):
            # Base64 encoded image
            image_data = base64.b64decode(image)
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            # Direct array
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_array = image
            else:
                raise ValueError("Invalid image array format")
        else:
            raise ValueError("Invalid image format")
        
        # Preprocess
        input_tensor, original_shape = self.preprocess_image(image_array)
        
        # Run inference
        try:
            outputs = self.onnx_session.run(
                self.output_names,
                {self.input_name: input_tensor}
            )
            
            # Postprocess
            predictions = self.onnx_session.run(None, {self.input_name: input_tensor})[0]
            detections = self.postprocess_predictions(
                predictions,
                original_shape,
                confidence_threshold,
                iou_threshold,
                max_detections
            )
            
            inference_time = time.time() - start_time
            
            # Update performance metrics
            self.request_count += 1
            self.total_inference_time += inference_time
            
            # Format response
            response = {
                'detections': [
                    {
                        'xyxy': det.xyxy.tolist(),
                        'confidence': float(det.confidence),
                        'class_id': int(det.class_id),
                        'class_name': det.class_name
                    }
                    for det in detections
                ],
                'inference_time': inference_time,
                'image_shape': original_shape,
                'model_info': {
                    'model_path': self.model_path,
                    'num_classes': self.num_classes,
                    'class_names': self.class_names,
                    'input_shape': self.input_shape,
                    'output_shape': self.output_shape
                }
            }
            
            return response
            
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
    
    def create_api(self) -> FastAPI:
        """
        Create FastAPI application
        
        Returns:
            Configured FastAPI app
        """
        app = FastAPI(
            title="OpenSupervision Inference Server",
            description="High-performance computer vision inference server",
            version="1.0.0"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/")
        async def root():
            return {"message": "OpenSupervision Inference Server", "status": "running"}
        
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "model_loaded": self.onnx_session is not None,
                "request_count": self.request_count,
                "avg_inference_time": self.total_inference_time / max(1, self.request_count)
            }
        
        @app.post("/predict", response_model=InferenceResponse)
        async def predict(request: InferenceRequest):
            """Single image prediction endpoint"""
            if not self.onnx_session:
                raise HTTPException(status_code=400, detail="Model not loaded")
            
            try:
                # Handle different input types
                if request.image_url:
                    if request.image_url.startswith('data:image'):
                        # Base64 encoded image
                        image_data = request.image_url.split(',')[1]
                        image_bytes = base64.b64decode(image_data)
                    else:
                        # URL or file path - simplified implementation
                        raise HTTPException(status_code=400, detail="URL/image path not supported in this endpoint")
                else:
                    raise HTTPException(status_code=400, detail="No image provided")
                
                # Run prediction
                result = self.predict_single(
                    image_bytes,
                    request.confidence_threshold,
                    request.iou_threshold,
                    request.max_detections
                )
                
                return InferenceResponse(**result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/predict/file")
        async def predict_file(file: UploadFile = File(...)):
            """File upload prediction endpoint"""
            if not self.onnx_session:
                raise HTTPException(status_code=400, detail="Model not loaded")
            
            try:
                # Read file content
                contents = await file.read()
                
                # Validate file type
                if not file.content_type or not file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail="File must be an image")
                
                # Run prediction
                result = self.predict_single(contents)
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/predict/batch")
        async def predict_batch(files: List[UploadFile] = File(...)):
            """Batch prediction endpoint"""
            if not self.onnx_session:
                raise HTTPException(status_code=400, detail="Model not loaded")
            
            if len(files) == 0:
                raise HTTPException(status_code=400, detail="No files provided")
            
            batch_id = f"batch_{int(time.time())}"
            results = []
            
            for file in files:
                try:
                    contents = await file.read()
                    if not file.content_type or not file.content_type.startswith('image/'):
                        results.append({
                            'file': file.filename,
                            'status': 'failed',
                            'error': 'Not an image file'
                        })
                        continue
                    
                    result = self.predict_single(contents)
                    results.append({
                        'file': file.filename,
                        'status': 'success',
                        'result': result
                    })
                    
                except Exception as e:
                    results.append({
                        'file': file.filename,
                        'status': 'failed',
                        'error': str(e)
                    })
            
            return {
                'batch_id': batch_id,
                'total_files': len(files),
                'results': results
            }
        
        @app.get("/model/info")
        async def model_info():
            """Get model information"""
            if not self.onnx_session:
                raise HTTPException(status_code=400, detail="Model not loaded")
            
            return {
                'model_path': self.model_path,
                'input_shape': self.input_shape,
                'output_shape': self.output_shape,
                'input_name': self.input_name,
                'output_names': self.output_names,
                'providers': self.onnx_session.get_providers(),
                'class_names': self.class_names,
                'num_classes': self.num_classes
            }
        
        @app.get("/metrics")
        async def get_metrics():
            """Get server metrics"""
            return {
                'request_count': self.request_count,
                'total_inference_time': self.total_inference_time,
                'avg_inference_time': self.total_inference_time / max(1, self.request_count),
                'requests_per_second': self.request_count / max(1, time.time() - self._start_time) if hasattr(self, '_start_time') else 0
            }
        
        return app
    
    def deploy(self,
              model_path: str,
              host: str = "0.0.0.0",
              port: int = 8000,
              workers: int = 1,
              auto_reload: bool = False) -> Dict:
        """
        Deploy inference server
        
        Args:
            model_path: Path to model file
            host: Server host address
            port: Server port
            workers: Number of workers
            auto_reload: Enable auto-reload for development
            
        Returns:
            Deployment information
        """
        # Load model
        model_info = self.load_model(model_path)
        
        # Create API
        self.app = self.create_api()
        
        # Set start time for metrics
        self._start_time = time.time()
        
        return {
            'status': 'deploying',
            'model_info': model_info,
            'server_config': {
                'host': host,
                'port': port,
                'workers': workers,
                'auto_reload': auto_reload
            },
            'endpoints': {
                'health': f'http://{host}:{port}/health',
                'predict': f'http://{host}:{port}/predict',
                'predict_file': f'http://{host}:{port}/predict/file',
                'predict_batch': f'http://{host}:{port}/predict/batch',
                'model_info': f'http://{host}:{port}/model/info',
                'metrics': f'http://{host}:{port}/metrics'
            }
        }
    
    def run_server(self,
                  host: str = "0.0.0.0",
                  port: int = 8000,
                  workers: int = 1,
                  auto_reload: bool = False):
        """
        Run the inference server
        
        Args:
            host: Server host address
            port: Server port
            workers: Number of workers
            auto_reload: Enable auto-reload for development
        """
        if not self.app:
            raise RuntimeError("Server not deployed. Call deploy() first.")
        
        print(f"Starting OpenSupervision Inference Server on {host}:{port}")
        print(f"Model: {self.model_path}")
        print(f"Classes: {self.num_classes} ({', '.join(self.class_names[:10])}{'...' if len(self.class_names) > 10 else ''})")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            reload=auto_reload
        )


class ModelLoader:
    """
    Model loading utility
    Handles different model formats and provides unified interface
    """
    
    def __init__(self):
        """Initialize Model Loader"""
        self.loaders = {
            'onnx': self._load_onnx,
            'pt': self._load_pytorch,
            'pth': self._load_pytorch,
            'engine': self._load_tensorrt,
            'model': self._load_ultralytics
        }
    
    def load_model(self, model_path: str, model_type: str = None) -> Any:
        """
        Load model from file
        
        Args:
            model_path: Path to model file
            model_type: Model type (auto-detected if None)
            
        Returns:
            Loaded model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Auto-detect model type
        if model_type is None:
            model_type = model_path.suffix.lower().lstrip('.')
        
        if model_type not in self.loaders:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        loader = self.loaders[model_type]
        return loader(model_path)
    
    def _load_onnx(self, model_path: Path):
        """Load ONNX model"""
        return InferenceServer({'inference': {}}).load_model(str(model_path))
    
    def _load_pytorch(self, model_path: Path):
        """Load PyTorch model"""
        try:
            import torch
            return torch.load(model_path)
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
    
    def _load_tensorrt(self, model_path: Path):
        """Load TensorRT model"""
        try:
            import tensorrt as trt
            # TensorRT engine loading would go here
            # This is a placeholder implementation
            raise NotImplementedError("TensorRT loading not yet implemented")
        except ImportError:
            raise ImportError("TensorRT not installed")
    
    def _load_ultralytics(self, model_path: Path):
        """Load Ultralytics model"""
        try:
            from ultralytics import YOLO
            return YOLO(str(model_path))
        except ImportError:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported model formats"""
        return list(self.loaders.keys())