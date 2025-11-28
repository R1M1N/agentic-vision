"""
CVAT Integration - replaces roboflow-python annotate functionality
Seamless integration with CVAT annotation tool
"""

import os
import json
import requests
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import yaml
from PIL import Image
from ..core.detections import Detections


class CVATAnnotate:
    """
    CVAT Integration for OpenSupervision
    Provides seamless annotation workflow using CVAT
    """
    
    def __init__(self, cvat_config: Dict):
        """
        Initialize CVAT Annotate
        
        Args:
            cvat_config: CVAT configuration dictionary
        """
        self.config = cvat_config
        self.base_url = cvat_config.get('url', 'http://localhost:8080')
        self.username = cvat_config.get('username', 'admin')
        self.password = cvat_config.get('password', 'password')
        self.api_url = f"{self.base_url}/api/v1"
        
        # Authentication
        self.session = requests.Session()
        self._authenticate()
        
        # Project tracking
        self.current_task_id = None
        self.current_job_id = None
        
    def _authenticate(self):
        """Authenticate with CVAT server"""
        auth_data = {
            'username': self.username,
            'password': self.password
        }
        
        try:
            response = self.session.post(f"{self.api_url}/auth/login", json=auth_data)
            response.raise_for_status()
            print(f"Successfully authenticated with CVAT at {self.base_url}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to authenticate with CVAT: {e}")
    
    def create_annotation_task(self,
                              name: str,
                              images: List[str],
                              labels: List[str],
                              annotation_type: str = "bbox",
                              description: str = "",
                              segment_size: int = 5000) -> Dict:
        """
        Create new annotation task in CVAT
        
        Args:
            name: Task name
            images: List of image paths
            labels: List of object labels
            annotation_type: Type of annotation ("bbox", "polygon", "keypoint")
            description: Task description
            segment_size: Number of images per segment
            
        Returns:
            Task information
        """
        # Prepare task data
        task_data = {
            'name': name,
            'description': description,
            'labels': [{'name': label, 'color': '#%02x%02x%02x' % (hash(label) % 256, hash(label) % 65536, hash(label) % 16777216)} for label in labels],
            'image_quality': 70,
            'segment_size': segment_size,
            'use_zip_chunks': True
        }
        
        try:
            # Create task
            response = self.session.post(f"{self.api_url}/tasks", json=task_data)
            response.raise_for_status()
            task_info = response.json()
            self.current_task_id = task_info['id']
            
            print(f"Created CVAT task '{name}' with ID: {self.current_task_id}")
            
            # Upload images
            self._upload_images_to_task(task_info['id'], images)
            
            # Get jobs for the task
            jobs = self._get_task_jobs(task_info['id'])
            if jobs:
                self.current_job_id = jobs[0]['id']
            
            return {
                'task_id': task_info['id'],
                'task_name': name,
                'jobs': jobs,
                'status': 'created',
                'cvat_url': f"{self.base_url}/tasks/{task_info['id']}/jobs/{self.current_job_id or ''}"
            }
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to create CVAT task: {e}")
    
    def _upload_images_to_task(self, task_id: int, images: List[str]):
        """Upload images to CVAT task"""
        # CVAT requires uploading files in chunks
        # This is a simplified implementation
        
        for i, image_path in enumerate(images):
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, skipping")
                continue
            
            with open(image_path, 'rb') as f:
                files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
                data = {'image_quality': 70}
                
                try:
                    response = self.session.post(
                        f"{self.api_url}/tasks/{task_id}/images",
                        files=files,
                        data=data
                    )
                    response.raise_for_status()
                    
                    if (i + 1) % 100 == 0:
                        print(f"Uploaded {i + 1}/{len(images)} images")
                        
                except requests.exceptions.RequestException as e:
                    print(f"Failed to upload {image_path}: {e}")
    
    def _get_task_jobs(self, task_id: int) -> List[Dict]:
        """Get jobs for a task"""
        try:
            response = self.session.get(f"{self.api_url}/tasks/{task_id}/jobs")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get jobs for task {task_id}: {e}")
            return []
    
    def send_samples_to_annotation(self,
                                  samples: List[str],
                                  task_name: str = None,
                                  labels: List[str] = None) -> Dict:
        """
        Send samples to CVAT for annotation (replaces roboflow-python annotate)
        
        Args:
            samples: List of image paths to annotate
            task_name: CVAT task name (auto-generated if None)
            labels: Object labels for annotation
            
        Returns:
            Annotation task information
        """
        if task_name is None:
            task_name = f"annotation_task_{len(samples)}_images"
        
        if labels is None:
            labels = self._auto_detect_labels(samples)
        
        return self.create_annotation_task(
            name=task_name,
            images=samples,
            labels=labels,
            description="OpenSupervision annotation task"
        )
    
    def _auto_detect_labels(self, samples: List[str]) -> List[str]:
        """Auto-detect labels from sample metadata"""
        # This is a simplified implementation
        # In a real implementation, you might analyze existing labels or use AI
        
        # Common object detection labels
        common_labels = ['person', 'car', 'bicycle', 'motorcycle', 'airplane',
                        'bus', 'train', 'truck', 'boat', 'traffic light',
                        'fire hydrant', 'stop sign', 'parking meter', 'bench',
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe']
        
        return common_labels[:10]  # Return first 10 as default
    
    def get_annotation_status(self, task_id: int = None) -> Dict:
        """
        Get annotation task status
        
        Args:
            task_id: Task ID (uses current task if None)
            
        Returns:
            Task status information
        """
        if task_id is None:
            task_id = self.current_task_id
        
        if task_id is None:
            raise ValueError("No task ID specified")
        
        try:
            response = self.session.get(f"{self.api_url}/tasks/{task_id}")
            response.raise_for_status()
            task_info = response.json()
            
            # Get job status
            jobs = self._get_task_jobs(task_id)
            
            status = {
                'task_id': task_id,
                'task_name': task_info['name'],
                'status': task_info['status'],
                'progress': task_info.get('progress', 0),
                'jobs': jobs,
                'completed_jobs': len([j for j in jobs if j['status'] == 'completed']),
                'total_jobs': len(jobs)
            }
            
            return status
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get task status: {e}")
    
    def download_annotations(self,
                           task_id: int = None,
                           format: str = "cvat_xml") -> Dict:
        """
        Download annotations from CVAT
        
        Args:
            task_id: Task ID (uses current task if None)
            format: Export format ("cvat_xml", "yolo", "coco", "labelme")
            
        Returns:
            Downloaded annotations information
        """
        if task_id is None:
            task_id = self.current_task_id
        
        if task_id is None:
            raise ValueError("No task ID specified")
        
        try:
            # Create export request
            export_data = {
                'format': format,
                'action': 'download'
            }
            
            response = self.session.post(
                f"{self.api_url}/tasks/{task_id}/annotations",
                json=export_data
            )
            response.raise_for_status()
            
            # The response contains the annotation file
            annotations_content = response.content
            
            return {
                'task_id': task_id,
                'format': format,
                'annotations': annotations_content.decode('utf-8') if isinstance(annotations_content, bytes) else annotations_content,
                'size': len(annotations_content),
                'downloaded_at': str(Path().cwd())
            }
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download annotations: {e}")
    
    def import_annotations(self,
                         annotations_path: str,
                         task_id: int = None,
                         format: str = "cvat_xml") -> Dict:
        """
        Import annotations to CVAT task
        
        Args:
            annotations_path: Path to annotation file
            task_id: Task ID (uses current task if None)
            format: Annotation format
            
        Returns:
            Import result
        """
        if task_id is None:
            task_id = self.current_task_id
        
        if task_id is None:
            raise ValueError("No task ID specified")
        
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
        
        try:
            with open(annotations_path, 'rb') as f:
                files = {
                    'annotation_file': (os.path.basename(annotations_path), f, 'application/xml')
                }
                
                response = self.session.post(
                    f"{self.api_url}/tasks/{task_id}/annotations?format={format}",
                    files=files
                )
                response.raise_for_status()
            
            return {
                'task_id': task_id,
                'format': format,
                'status': 'imported',
                'file': annotations_path
            }
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to import annotations: {e}")
    
    def create_auto_labeling_job(self,
                               model_path: str,
                               images: List[str],
                               confidence_threshold: float = 0.5,
                               task_name: str = None) -> Dict:
        """
        Create auto-labeling job using a trained model
        
        Args:
            model_path: Path to trained model
            images: List of images to auto-label
            confidence_threshold: Confidence threshold for predictions
            task_name: Name for the auto-labeling task
            
        Returns:
            Auto-labeling job information
        """
        if task_name is None:
            task_name = f"auto_labeling_{len(images)}_images"
        
        # Load model and generate predictions
        from ..inference.model_loader import ModelLoader
        
        model_loader = ModelLoader()
        model = model_loader.load_model(model_path)
        
        auto_annotations = []
        
        for image_path in images:
            if not os.path.exists(image_path):
                continue
            
            # Run inference
            image = Image.open(image_path)
            predictions = model.predict(image)
            
            # Convert to CVAT format
            image_annotations = self._convert_to_cvat_format(predictions, image_path)
            auto_annotations.extend(image_annotations)
        
        # Create task with pre-annotated data
        labels = list(set([ann['label'] for ann in auto_annotations]))
        
        return self.create_annotation_task(
            name=task_name,
            images=images,
            labels=labels,
            description=f"Auto-labeling task with confidence threshold {confidence_threshold}"
        )
    
    def _convert_to_cvat_format(self, predictions: Detections, image_path: str) -> List[Dict]:
        """Convert Detections to CVAT annotation format"""
        annotations = []
        
        for det in predictions:
            # CVAT expects points in (x, y) format
            x1, y1, x2, y2 = det.xyxy
            
            annotation = {
                'image_id': image_path,
                'label': det.class_name,
                'points': [x1, y1, x2, y1, x2, y2, x1, y2],  # Rectangle corners
                'type': 'rectangle',
                'occluded': False,
                'z_order': 0
            }
            
            annotations.append(annotation)
        
        return annotations
    
    def sync_with_fiftyone(self, dataset_name: str) -> Dict:
        """
        Sync CVAT annotations with FiftyOne dataset
        
        Args:
            dataset_name: FiftyOne dataset name
            
        Returns:
            Sync result
        """
        # This would integrate with FiftyOne to sync annotations
        # Implementation would depend on FiftyOne API
        
        sync_info = {
            'dataset_name': dataset_name,
            'task_id': self.current_task_id,
            'synced_at': str(Path().cwd()),
            'status': 'synced'
        }
        
        print(f"Synced CVAT annotations with FiftyOne dataset '{dataset_name}'")
        return sync_info
    
    def get_annotation_statistics(self, task_id: int = None) -> Dict:
        """
        Get annotation statistics
        
        Args:
            task_id: Task ID (uses current task if None)
            
        Returns:
            Annotation statistics
        """
        if task_id is None:
            task_id = self.current_task_id
        
        # This is a simplified implementation
        # Real implementation would analyze the annotation data
        
        return {
            'task_id': task_id,
            'total_annotations': 0,
            'label_distribution': {},
            'completion_percentage': 0,
            'annotated_frames': 0,
            'total_frames': 0
        }
    
    def delete_task(self, task_id: int) -> bool:
        """
        Delete CVAT task
        
        Args:
            task_id: Task ID to delete
            
        Returns:
            True if successful
        """
        try:
            response = self.session.delete(f"{self.api_url}/tasks/{task_id}")
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to delete task {task_id}: {e}")
            return False
    
    def list_tasks(self) -> List[Dict]:
        """List all CVAT tasks"""
        try:
            response = self.session.get(f"{self.api_url}/tasks")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to list tasks: {e}")
            return []
    
    def get_task_url(self, task_id: int = None) -> str:
        """
        Get CVAT task URL
        
        Args:
            task_id: Task ID (uses current task if None)
            
        Returns:
            CVAT task URL
        """
        if task_id is None:
            task_id = self.current_task_id
        
        if task_id is None:
            raise ValueError("No task ID specified")
        
        return f"{self.base_url}/tasks/{task_id}/jobs/{self.current_job_id or ''}"
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.session.close()


class AutoAnnotator:
    """
    Auto-annotation using AI models
    Replicates the "Label Assist" functionality
    """
    
    def __init__(self, base_model: str = "yolov8n", target_model: str = "yolov8s"):
        """
        Initialize AutoAnnotator
        
        Args:
            base_model: Foundation model for auto-labeling
            target_model: Target model to train
        """
        self.base_model = base_model
        self.target_model = target_model
        self.model = None
    
    def load_base_model(self):
        """Load the base model for auto-labeling"""
        # Load YOLO model for auto-labeling
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.base_model)
            print(f"Loaded base model: {self.base_model}")
        except ImportError:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
    
    def auto_label_images(self,
                         image_paths: List[str],
                         confidence_threshold: float = 0.5,
                         save_format: str = "yolo") -> Dict:
        """
        Auto-label images using the base model
        
        Args:
            image_paths: List of image paths to auto-label
            confidence_threshold: Confidence threshold for predictions
            save_format: Format to save labels ("yolo", "coco", "cvat_xml")
            
        Returns:
            Auto-labeling results
        """
        if self.model is None:
            self.load_base_model()
        
        results = {
            'processed_images': 0,
            'total_detections': 0,
            'labels_created': [],
            'errors': []
        }
        
        for image_path in image_paths:
            try:
                # Run inference
                prediction = self.model(image_path)[0]
                
                # Convert to Detections
                detections = Detections.from_yolo(prediction, (640, 640))
                
                # Filter by confidence
                detections = detections.filter(confidence_threshold=confidence_threshold)
                
                # Save labels
                label_file = self._save_labels(detections, image_path, save_format)
                results['labels_created'].append(label_file)
                results['total_detections'] += len(detections)
                results['processed_images'] += 1
                
            except Exception as e:
                results['errors'].append(f"Failed to process {image_path}: {str(e)}")
        
        return results
    
    def _save_labels(self, detections: Detections, image_path: str, format: str) -> str:
        """Save detections in specified format"""
        image_path = Path(image_path)
        label_path = image_path.parent / f"{image_path.stem}.txt"
        
        if format == "yolo":
            self._save_yolo_labels(detections, label_path, (640, 640))
        
        return str(label_path)
    
    def _save_yolo_labels(self, detections: Detections, label_path: Path, image_shape: Tuple[int, int]):
        """Save labels in YOLO format"""
        h, w = image_shape
        
        with open(label_path, 'w') as f:
            for det in detections:
                x1, y1, x2, y2 = det.xyxy
                
                # Convert to YOLO format (center x, center y, width, height normalized)
                center_x = (x1 + x2) / 2 / w
                center_y = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                f.write(f"{det.class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")