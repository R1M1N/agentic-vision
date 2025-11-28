"""
Core Detections primitive - foundation of OpenSupervision
Model-agnostic detection handling that works with any computer vision model
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass


@dataclass
class Detection:
    """
    Single detection result
    """
    xyxy: np.ndarray      # Bounding box [x1, y1, x2, y2]
    confidence: float     # Detection confidence score
    class_id: int         # Class ID
    class_name: str       # Class name
    mask: Optional[np.ndarray] = None  # Segmentation mask (if available)
    keypoints: Optional[np.ndarray] = None  # Keypoints (if available)
    tracker_id: Optional[int] = None   # Tracking ID (if available)


class Detections:
    """
    Core Detections class - model-agnostic detection handling
    Replicates supervision.Detections functionality
    """
    
    def __init__(
        self,
        xyxy: Union[np.ndarray, List],
        confidence: Union[np.ndarray, List, float],
        class_id: Union[np.ndarray, List, int],
        class_name: Union[List, str, None] = None,
        mask: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        tracker_id: Optional[np.ndarray] = None,
        data: Optional[Dict] = None
    ):
        """
        Initialize Detections
        
        Args:
            xyxy: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
            confidence: Confidence scores [N] or single float
            class_id: Class IDs [N] or single int
            class_name: Class names [N] or single string
            mask: Optional segmentation masks [N, H, W]
            keypoints: Optional keypoints [N, K, 3] (x, y, visibility)
            tracker_id: Optional tracking IDs [N]
            data: Additional metadata
        """
        # Convert to numpy arrays
        self.xyxy = np.array(xyxy) if not isinstance(xyxy, np.ndarray) else xyxy
        self.confidence = np.array(confidence) if not isinstance(confidence, (int, float)) else np.array([confidence])
        self.class_id = np.array(class_id) if not isinstance(class_id, (int, np.ndarray)) else class_id
        
        # Handle class names
        if class_name is None:
            self.class_name = [f"class_{i}" for i in self.class_id]
        elif isinstance(class_name, str):
            self.class_name = [class_name] * len(self.class_id)
        else:
            self.class_name = list(class_name)
        
        # Optional fields
        self.mask = mask
        self.keypoints = keypoints
        self.tracker_id = tracker_id
        self.data = data or {}
        
        # Validate arrays have same length
        self._validate_arrays()
    
    def _validate_arrays(self):
        """Validate that all arrays have consistent lengths"""
        n = len(self.xyxy)
        
        if len(self.confidence) != n and len(self.confidence) != 1:
            raise ValueError(f"Confidence array length {len(self.confidence)} doesn't match bbox count {n}")
        
        if len(self.class_id) != n and len(self.class_id) != 1:
            raise ValueError(f"Class ID array length {len(self.class_id)} doesn't match bbox count {n}")
        
        if len(self.class_name) != n and len(self.class_name) != 1:
            raise ValueError(f"Class name length {len(self.class_name)} doesn't match bbox count {n}")
        
        if self.mask is not None:
            if len(self.mask) != n:
                raise ValueError(f"Mask array length {len(self.mask)} doesn't match bbox count {n}")
        
        if self.keypoints is not None:
            if len(self.keypoints) != n:
                raise ValueError(f"Keypoints array length {len(self.keypoints)} doesn't match bbox count {n}")
        
        if self.tracker_id is not None:
            if len(self.tracker_id) != n:
                raise ValueError(f"Tracker ID array length {len(self.tracker_id)} doesn't match bbox count {n}")
    
    @classmethod
    def from_yolo(cls, results, image_shape: Tuple[int, int]):
        """
        Create Detections from YOLO results
        
        Args:
            results: YOLO results object
            image_shape: Original image shape (height, width)
        """
        if hasattr(results, 'boxes'):
            boxes = results.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confidence = boxes.conf.cpu().numpy()
            class_id = boxes.cls.cpu().numpy().astype(int)
            
            # Get class names from YOLO model
            if hasattr(results.names):
                class_name = [results.names[int(cls_id)] for cls_id in class_id]
            else:
                class_name = [f"class_{i}" for i in class_id]
            
            return cls(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                mask=None
            )
        else:
            return cls.empty()
    
    @classmethod
    def from_coco(cls, detections, image_shape: Tuple[int, int]):
        """
        Create Detections from COCO format
        
        Args:
            detections: List of COCO detection dicts
            image_shape: Original image shape (height, width)
        """
        if not detections:
            return cls.empty()
        
        xyxy = []
        confidence = []
        class_id = []
        class_name = []
        
        for det in detections:
            bbox = det['bbox']  # [x, y, w, h] in COCO format
            x, y, w, h = bbox
            xyxy.append([x, y, x + w, y + h])
            
            confidence.append(det.get('score', 1.0))
            class_id.append(det.get('category_id', 0))
            
            # COCO category mapping would be needed here
            class_name.append(f"coco_class_{det.get('category_id', 0)}")
        
        return cls(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id),
            class_name=class_name
        )
    
    @classmethod
    def from_ultralytics(cls, results):
        """
        Create Detections from Ultralytics results
        
        Args:
            results: Ultralytics results object
        """
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confidence = boxes.conf.cpu().numpy()
            class_id = boxes.cls.cpu().numpy().astype(int)
            
            # Get class names
            class_name = []
            if hasattr(results, 'names'):
                names = results.names
                class_name = [names.get(int(cls_id), f"class_{cls_id}") for cls_id in class_id]
            else:
                class_name = [f"class_{i}" for i in class_id]
            
            return cls(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name
            )
        else:
            return cls.empty()
    
    @classmethod
    def from_torchvision(cls, outputs, image_shape: Tuple[int, int], class_names: List[str] = None):
        """
        Create Detections from TorchVision Faster R-CNN outputs
        
        Args:
            outputs: TorchVision model outputs
            image_shape: Original image shape (height, width)
            class_names: Class names mapping
        """
        if len(outputs) == 0:
            return cls.empty()
        
        output = outputs[0]
        
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        keep = scores > 0.5  # Default threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # Get class names
        if class_names:
            class_name = [class_names[label] if label < len(class_names) else f"class_{label}" for label in labels]
        else:
            class_name = [f"class_{label}" for label in labels]
        
        return cls(
            xyxy=boxes,
            confidence=scores,
            class_id=labels,
            class_name=class_name
        )
    
    @classmethod
    def from_mediapipe(cls, results, image_shape: Tuple[int, int]):
        """
        Create Detections from MediaPipe results
        
        Args:
            results: MediaPipe results object
            image_shape: Original image shape (height, width)
        """
        detections = []
        
        if hasattr(results, 'detections') and results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative to absolute coordinates
                h, w, _ = image_shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                xyxy = [x, y, x + width, y + height]
                confidence = detection.score[0] if detection.score else 0.0
                class_id = 0  # MediaPipe typically detects a specific class
                class_name = "person" if class_id == 0 else f"class_{class_id}"
                
                detections.append({
                    'xyxy': xyxy,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        if not detections:
            return cls.empty()
        
        xyxy = [d['xyxy'] for d in detections]
        confidence = [d['confidence'] for d in detections]
        class_id = [d['class_id'] for d in detections]
        class_name = [d['class_name'] for d in detections]
        
        return cls(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id),
            class_name=class_name
        )
    
    @classmethod
    def empty(cls):
        """Create empty Detections object"""
        return cls(
            xyxy=np.array([]).reshape(0, 4),
            confidence=np.array([]),
            class_id=np.array([], dtype=int),
            class_name=[]
        )
    
    def is_empty(self) -> bool:
        """Check if detections is empty"""
        return len(self.xyxy) == 0
    
    def __len__(self) -> int:
        """Number of detections"""
        return len(self.xyxy)
    
    def __getitem__(self, index) -> Union['Detection', 'Detections']:
        """
        Get detection(s) by index or boolean mask
        
        Args:
            index: Integer index, slice, or boolean array
            
        Returns:
            Single Detection (if integer index) or new Detections (if boolean/index array)
        """
        if isinstance(index, (bool, np.bool_)):
            # Handle single boolean (rare case)
            if index:
                return self[0]
            else:
                return self[[]]
        
        if isinstance(index, np.ndarray) and index.dtype == bool:
            # Boolean indexing - return new Detections object
            return Detections(
                xyxy=self.xyxy[index],
                confidence=self.confidence[index],
                class_id=self.class_id[index],
                class_name=[self.class_name[i] for i in range(len(self.class_name)) if index[i]],
                mask=self.mask[index] if self.mask is not None else None,
                keypoints=self.keypoints[index] if self.keypoints is not None else None,
                tracker_id=self.tracker_id[index] if self.tracker_id is not None else None
            )
        
        if isinstance(index, (list, tuple, np.ndarray)):
            # Array indexing - return new Detections object
            return Detections(
                xyxy=self.xyxy[index],
                confidence=self.confidence[index],
                class_id=self.class_id[index],
                class_name=[self.class_name[i] for i in index],
                mask=self.mask[index] if self.mask is not None else None,
                keypoints=self.keypoints[index] if self.keypoints is not None else None,
                tracker_id=self.tracker_id[index] if self.tracker_id is not None else None
            )
        
        # Single integer indexing - return single Detection object
        return Detection(
            xyxy=self.xyxy[index],
            confidence=self.confidence[index],
            class_id=self.class_id[index],
            class_name=self.class_name[index],
            mask=self.mask[index] if self.mask is not None else None,
            keypoints=self.keypoints[index] if self.keypoints is not None else None,
            tracker_id=self.tracker_id[index] if self.tracker_id is not None else None
        )
    
    def filter(self, 
               confidence_threshold: Optional[float] = None,
               class_ids: Optional[List[int]] = None,
               tracker_ids: Optional[List[int]] = None) -> 'Detections':
        """
        Filter detections based on criteria
        
        Args:
            confidence_threshold: Minimum confidence threshold
            class_ids: Allowed class IDs
            tracker_ids: Allowed tracker IDs
            
        Returns:
            Filtered Detections
        """
        keep = np.ones(len(self), dtype=bool)
        
        if confidence_threshold is not None:
            keep &= self.confidence >= confidence_threshold
        
        if class_ids is not None:
            keep &= np.isin(self.class_id, class_ids)
        
        if tracker_ids is not None and self.tracker_id is not None:
            keep &= np.isin(self.tracker_id, tracker_ids)
        
        return self[keep]
    
    def merge(self, other: 'Detections') -> 'Detections':
        """
        Merge two Detections objects
        
        Args:
            other: Other Detections to merge
            
        Returns:
            Merged Detections
        """
        if self.is_empty():
            return other
        if other.is_empty():
            return self
        
        xyxy = np.vstack([self.xyxy, other.xyxy])
        confidence = np.concatenate([self.confidence, other.confidence])
        class_id = np.concatenate([self.class_id, other.class_id])
        class_name = self.class_name + other.class_name
        
        mask = None
        if self.mask is not None or other.mask is not None:
            if self.mask is None:
                mask = np.vstack([np.zeros((len(self), *other.mask.shape[1:])), other.mask])
            elif other.mask is None:
                mask = np.vstack([self.mask, np.zeros((len(other), *self.mask.shape[1:]))])
            else:
                mask = np.vstack([self.mask, other.mask])
        
        keypoints = None
        if self.keypoints is not None or other.keypoints is not None:
            if self.keypoints is None:
                keypoints = np.vstack([np.zeros((len(self), *other.keypoints.shape[1:])), other.keypoints])
            elif other.keypoints is None:
                keypoints = np.vstack([self.keypoints, np.zeros((len(other), *self.keypoints.shape[1:]))])
            else:
                keypoints = np.vstack([self.keypoints, other.keypoints])
        
        tracker_id = None
        if self.tracker_id is not None or other.tracker_id is not None:
            if self.tracker_id is None:
                tracker_id = np.concatenate([np.full(len(self), -1), other.tracker_id])
            elif other.tracker_id is None:
                tracker_id = np.concatenate([self.tracker_id, np.full(len(other), -1)])
            else:
                tracker_id = np.concatenate([self.tracker_id, other.tracker_id])
        
        return Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            mask=mask,
            keypoints=keypoints,
            tracker_id=tracker_id,
            data={**self.data, **other.data}
        )
    
    def __and__(self, condition: np.ndarray) -> 'Detections':
        """Boolean indexing support"""
        return self[condition]
    
    def __or__(self, other: 'Detections') -> 'Detections':
        """Merge support with | operator"""
        return self.merge(other)
    
    def get_area(self) -> np.ndarray:
        """Get area of bounding boxes"""
        if self.is_empty():
            return np.array([])
        
        widths = self.xyxy[:, 2] - self.xyxy[:, 0]
        heights = self.xyxy[:, 3] - self.xyxy[:, 1]
        return widths * heights
    
    def get_centers(self) -> np.ndarray:
        """Get centers of bounding boxes"""
        if self.is_empty():
            return np.array([]).reshape(0, 2)
        
        centers_x = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2
        centers_y = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2
        return np.column_stack([centers_x, centers_y])
    
    def iou_with(self, other: 'Detections') -> np.ndarray:
        """
        Calculate IoU matrix with other detections
        
        Args:
            other: Other Detections object
            
        Returns:
            IoU matrix [len(self), len(other)]
        """
        if self.is_empty() or other.is_empty():
            return np.array([]).reshape(len(self), len(other))
        
        iou_matrix = np.zeros((len(self), len(other)))
        
        for i, box1 in enumerate(self.xyxy):
            for j, box2 in enumerate(other.xyxy):
                iou_matrix[i, j] = self._calculate_iou(box1, box2)
        
        return iou_matrix
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def to_coco(self) -> List[Dict]:
        """
        Convert to COCO format
        
        Returns:
            List of COCO detection dictionaries
        """
        coco_detections = []
        
        for det in self:
            x1, y1, x2, y2 = det.xyxy
            width = x2 - x1
            height = y2 - y1
            
            coco_detections.append({
                'image_id': 0,  # Would need image ID from dataset
                'category_id': det.class_id,
                'bbox': [float(x1), float(y1), float(width), float(height)],
                'score': float(det.confidence)
            })
        
        return coco_detections
    
    def to_yolo(self, image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Convert to YOLO format
        
        Args:
            image_shape: Original image shape (height, width)
            
        Returns:
            List of YOLO detection dictionaries
        """
        yolo_detections = []
        h, w = image_shape
        
        for det in self:
            x1, y1, x2, y2 = det.xyxy
            
            # Convert to YOLO format (center x, center y, width, height normalized)
            center_x = (x1 + x2) / 2 / w
            center_y = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            yolo_detections.append({
                'class': det.class_id,
                'x_center': float(center_x),
                'y_center': float(center_y),
                'width': float(width),
                'height': float(height),
                'confidence': float(det.confidence)
            })
        
        return yolo_detections
    
    def annotate_image(self, image: np.ndarray, annotator) -> np.ndarray:
        """
        Annotate image with detections
        
        Args:
            image: Input image
            annotator: Annotation function
            
        Returns:
            Annotated image
        """
        return annotator.annotate(image, self)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Detections(n={len(self)}, classes={list(set(self.class_name))})"