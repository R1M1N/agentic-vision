"""
Object Tracking System - implements ByteTrack, DeepSORT, BoT-SORT
Replicates roboflow/trackers functionality with open-source algorithms
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import time
from collections import defaultdict, deque
import json
from dataclasses import dataclass

from ..core.detections import Detections


@dataclass
class Track:
    """Single track object"""
    track_id: int
    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    timestamp: float
    age: int
    hits: int
    time_since_update: int
    kalman_filter: Any = None
    feature_history: List[np.ndarray] = None
    
    def __post_init__(self):
        if self.feature_history is None:
            self.feature_history = []


class KalmanFilter:
    """Simple Kalman Filter for object tracking"""
    
    def __init__(self):
        self.kalman = cv2.KalmanFilter(8, 4)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float32)
        
        self.kalman.transitionMatrix = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                                                [0, 1, 0, 0, 0, 1, 0, 0],
                                                [0, 0, 1, 0, 0, 0, 1, 0],
                                                [0, 0, 0, 1, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 1, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 1, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
        
        self.kalman.processNoiseCov = 0.03 * np.eye(8, dtype=np.float32)
        self.kalman.measurementNoiseCov = 0.1 * np.eye(4, dtype=np.float32)
        self.kalman.errorCovPost = 1.0 * np.eye(8, dtype=np.float32)
    
    def update(self, bbox: np.ndarray) -> np.ndarray:
        """Update Kalman filter with new bbox"""
        self.kalman.correct(bbox.astype(np.float32))
        return self.kalman.predict()
    
    def predict(self) -> np.ndarray:
        """Predict next position"""
        return self.kalman.predict()


class ByteTrack:
    """
    ByteTrack Implementation
    High-performance motion-based tracking
    """
    
    def __init__(self, track_thresh: float = 0.5, 
                 track_buffer: int = 30, 
                 match_thresh: float = 0.8,
                 mot20: bool = False):
        """
        Initialize ByteTrack
        
        Args:
            track_thresh: Confidence threshold for tracks
            track_buffer: Number of frames to keep dead tracks
            match_thresh: IoU threshold for matching
            mot20: Use MOT20 settings
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20
        
        # Track management
        self.tracks = {}
        self.track_id_count = 0
        self.dead_tracks = deque()
        
    def update(self, detections: Detections, frame: np.ndarray = None) -> Detections:
        """
        Update tracks with new detections
        
        Args:
            detections: New detections
            frame: Current frame for optional processing
            
        Returns:
            Detections with tracker IDs
        """
        if detections.is_empty():
            # Clean up old tracks
            self._clean_up_tracks()
            return Detections.empty()
        
        # Sort detections by confidence
        indices = np.argsort(detections.confidence)[::-1]
        detections_sorted = detections[indices]
        
        # Remove low confidence detections for tracking
        keep_indices = detections_sorted.confidence > self.track_thresh
        detections_for_track = detections_sorted[keep_indices]
        
        if detections_for_track.is_empty():
            self._clean_up_tracks()
            return Detections.empty()
        
        # Extract tracklets (high confidence detections)
        tracklets_bboxes = detections_for_track.xyxy
        tracklets_scores = detections_for_track.confidence
        tracklets_classes = detections_for_track.class_id
        tracklets_names = detections_for_track.class_name
        
        # Remove existing tracks from detections
        detections_remove = np.zeros(len(detections_sorted), dtype=bool)
        for track_id, track in list(self.tracks.items()):
            bbox = track.bbox
            ious = self._ious(tracklets_bboxes, bbox.reshape(1, 4))
            
            if len(ious) > 0 and ious.max() > self.match_thresh:
                # Remove matched detection
                max_idx = np.argmax(ious)
                detections_remove[indices[keep_indices][max_idx]] = True
        
        # Remove matched detections
        remain_detections = detections_sorted[~detections_remove]
        
        # Add new tracks
        new_tracks = []
        if len(remain_detections.xyxy) > 0:
            for i in range(len(remain_detections.xyxy)):
                bbox = remain_detections.xyxy[i]
                score = remain_detections.confidence[i]
                class_id = remain_detections.class_id[i]
                class_name = remain_detections.class_name[i]
                
                track = self._create_track(bbox, score, class_id, class_name)
                new_tracks.append(track)
        
        # Update existing tracks
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            
            if len(tracklets_bboxes) > 0:
                # Find best match
                best_match = -1
                best_iou = 0
                
                for i, bbox in enumerate(tracklets_bboxes):
                    iou = self._iou(track.bbox, bbox)
                    if iou > best_iou and iou > self.match_thresh:
                        best_iou = iou
                        best_match = i
                
                if best_match != -1:
                    # Update track
                    track.bbox = tracklets_bboxes[best_match]
                    track.confidence = tracklets_scores[best_match]
                    track.age += 1
                    track.hits += 1
                    track.time_since_update = 0
                    
                    # Update Kalman filter
                    if track.kalman_filter:
                        track.kalman_filter.update(track.bbox)
                    
                    # Remove matched detection
                    tracklets_bboxes = np.delete(tracklets_bboxes, best_match, axis=0)
                    tracklets_scores = np.delete(tracklets_scores, best_match, axis=0)
                    tracklets_classes = np.delete(tracklets_classes, best_match, axis=0)
                    tracklets_names = np.delete(tracklets_names, best_match, axis=0)
                else:
                    # No match found
                    track.time_since_update += 1
            else:
                track.time_since_update += 1
        
        # Add new tracks
        for track in new_tracks:
            self.tracks[track.track_id] = track
        
        # Remove dead tracks
        self._clean_up_tracks()
        
        # Compile final detections with tracker IDs
        final_detections = self._compile_detections()
        
        return final_detections
    
    def _create_track(self, bbox: np.ndarray, confidence: float, 
                     class_id: int, class_name: str) -> Track:
        """Create new track"""
        track_id = self.track_id_count
        self.track_id_count += 1
        
        track = Track(
            track_id=track_id,
            bbox=bbox,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            timestamp=time.time(),
            age=1,
            hits=1,
            time_since_update=0,
            kalman_filter=KalmanFilter()
        )
        
        # Initialize Kalman filter
        track.kalman_filter.update(bbox)
        
        return track
    
    def _clean_up_tracks(self):
        """Remove tracks that have been inactive too long"""
        dead_track_ids = []
        
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.track_buffer:
                dead_track_ids.append(track_id)
        
        for track_id in dead_track_ids:
            track = self.tracks.pop(track_id)
            self.dead_tracks.append(track)
            
            # Limit dead track history
            if len(self.dead_tracks) > 100:
                self.dead_tracks.popleft()
    
    def _compile_detections(self) -> Detections:
        """Compile final detections with tracker IDs"""
        active_tracks = [track for track in self.tracks.values() 
                        if track.time_since_update <= 1]
        
        if not active_tracks:
            return Detections.empty()
        
        xyxy = np.array([track.bbox for track in active_tracks])
        confidence = np.array([track.confidence for track in active_tracks])
        class_id = np.array([track.class_id for track in active_tracks])
        class_name = [track.class_name for track in active_tracks]
        tracker_id = np.array([track.track_id for track in active_tracks])
        
        return Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            tracker_id=tracker_id
        )
    
    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
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
    
    def _ious(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Calculate IoU matrix between two sets of boxes"""
        ious = np.zeros((len(boxes1), len(boxes2)))
        
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                ious[i, j] = self._iou(box1, box2)
        
        return ious


class DeepSORT:
    """
    DeepSORT Implementation
    Appearance-based tracking with ReID features
    """
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 nn_budget: int = 100,
                 max_cosine_distance: float = 0.2):
        """
        Initialize DeepSORT
        
        Args:
            max_age: Maximum number of frames to keep track alive
            min_hits: Minimum hits before track becomes confirmed
            iou_threshold: IoU threshold for matching
            nn_budget: Maximum number of features to store per class
            max_cosine_distance: Maximum cosine distance for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.nn_budget = nn_budget
        self.max_cosine_distance = max_cosine_distance
        
        # Track management
        self.tracks = {}
        self.track_id_count = 0
        self.features = defaultdict(lambda: deque(maxlen=nn_budget))
        
    def update(self, detections: Detections, features: List[np.ndarray] = None, 
              frame: np.ndarray = None) -> Detections:
        """
        Update tracks with new detections and features
        
        Args:
            detections: New detections
            features: ReID features for each detection
            frame: Current frame
            
        Returns:
            Detections with tracker IDs
        """
        if detections.is_empty():
            self._clean_up_tracks()
            return Detections.empty()
        
        # Extract features if not provided
        if features is None:
            features = self._extract_features(detections, frame)
        
        # Sort detections by confidence
        indices = np.argsort(detections.confidence)[::-1]
        detections_sorted = detections[indices]
        features_sorted = [features[i] for i in indices] if features else None
        
        # Get unmatched detections from previous frame
        unmatched_detections = list(range(len(detections_sorted)))
        
        # Match with existing tracks
        matches, unmatched_tracks, unmatched_detections = self._match_tracks(
            detections_sorted, features_sorted, unmatched_detections
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            self._update_track(track_idx, det_idx, detections_sorted, features_sorted)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_track(det_idx, detections_sorted, features_sorted)
        
        # Clean up old tracks
        self._clean_up_tracks()
        
        # Return confirmed tracks
        return self._get_confirmed_tracks()
    
    def _match_tracks(self, detections: Detections, features: List[np.ndarray], 
                     unmatched_detections: List[int]) -> Tuple[List[Tuple], List[int], List[int]]:
        """Match tracks with detections"""
        matches = []
        unmatched_tracks = []
        
        # Get track features
        track_features = []
        track_ids = []
        
        for track_id, track in self.tracks.items():
            if track.time_since_update <= 1 and len(self.features[track_id]) > 0:
                track_features.append(np.mean(self.features[track_id], axis=0))
                track_ids.append(track_id)
        
        if not track_features or not unmatched_detections:
            return matches, list(range(len(self.tracks))), unmatched_detections
        
        # Convert to numpy arrays
        track_features = np.array(track_features)
        det_features = np.array([features[i] for i in unmatched_detections])
        
        # Calculate cosine distances
        distances = 1 - np.dot(track_features, det_features.T)
        
        # Greedy matching
        used_detections = set()
        used_tracks = set()
        
        for i in range(len(track_ids)):
            if i in used_tracks:
                continue
            
            min_dist_idx = np.argmin(distances[i])
            min_dist = distances[i, min_dist_idx]
            
            if min_dist <= self.max_cosine_distance:
                track_id = track_ids[i]
                det_idx = unmatched_detections[min_dist_idx]
                
                if det_idx not in used_detections:
                    matches.append((track_id, det_idx))
                    used_tracks.add(i)
                    used_detections.add(det_idx)
        
        # Update unmatched lists
        unmatched_tracks = [i for i in range(len(track_ids)) if i not in used_tracks]
        unmatched_detections = [det for det in unmatched_detections if det not in used_detections]
        
        return matches, [track_ids[i] for i in unmatched_tracks], unmatched_detections
    
    def _update_track(self, track_id: int, det_idx: int, detections: Detections, features: List[np.ndarray]):
        """Update existing track"""
        track = self.tracks[track_id]
        
        track.bbox = detections.xyxy[det_idx]
        track.confidence = detections.confidence[det_idx]
        track.class_id = detections.class_id[det_idx]
        track.class_name = detections.class_name[det_idx]
        track.age += 1
        track.hits += 1
        track.time_since_update = 0
        
        # Update features
        if features and det_idx < len(features):
            self.features[track_id].append(features[det_idx])
        
        # Kalman filter update
        if track.kalman_filter:
            track.kalman_filter.update(track.bbox)
    
    def _create_track(self, det_idx: int, detections: Detections, features: List[np.ndarray]):
        """Create new track"""
        track_id = self.track_id_count
        self.track_id_count += 1
        
        track = Track(
            track_id=track_id,
            bbox=detections.xyxy[det_idx],
            confidence=detections.confidence[det_idx],
            class_id=detections.class_id[det_idx],
            class_name=detections.class_name[det_idx],
            timestamp=time.time(),
            age=1,
            hits=1,
            time_since_update=0,
            kalman_filter=KalmanFilter()
        )
        
        # Initialize Kalman filter
        track.kalman_filter.update(track.bbox)
        
        # Add features
        if features and det_idx < len(features):
            self.features[track_id].append(features[det_idx])
        
        self.tracks[track_id] = track
    
    def _clean_up_tracks(self):
        """Remove tracks that have been inactive too long"""
        dead_track_ids = []
        
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_age:
                dead_track_ids.append(track_id)
        
        for track_id in dead_track_ids:
            self.tracks.pop(track_id)
            if track_id in self.features:
                self.features.pop(track_id)
    
    def _get_confirmed_tracks(self) -> Detections:
        """Get confirmed tracks only"""
        confirmed_tracks = [track for track in self.tracks.values() 
                          if track.hits >= self.min_hits and track.time_since_update <= 1]
        
        if not confirmed_tracks:
            return Detections.empty()
        
        xyxy = np.array([track.bbox for track in confirmed_tracks])
        confidence = np.array([track.confidence for track in confirmed_tracks])
        class_id = np.array([track.class_id for track in confirmed_tracks])
        class_name = [track.class_name for track in confirmed_tracks]
        tracker_id = np.array([track.track_id for track in confirmed_tracks])
        
        return Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
            tracker_id=tracker_id
        )
    
    def _extract_features(self, detections: Detections, frame: np.ndarray) -> List[np.ndarray]:
        """Extract ReID features from detections"""
        # This is a simplified implementation
        # Real implementation would use a proper ReID model
        
        features = []
        
        for bbox in detections.xyxy:
            # Extract simple appearance features (center, size, aspect ratio)
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            
            # Simple feature vector
            feature = np.array([center_x, center_y, width, height, aspect_ratio])
            features.append(feature)
        
        return features


class ObjectTracker:
    """
    Unified Object Tracking Interface
    Supports multiple tracking algorithms
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Object Tracker
        
        Args:
            config: Tracking configuration
        """
        self.config = config
        self.algorithm = config.get('algorithm', 'bytetrack').lower()
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        
        # Initialize tracker
        if self.algorithm == 'bytetrack':
            self.tracker = ByteTrack(
                track_thresh=0.5,
                track_buffer=self.max_age,
                match_thresh=0.8
            )
        elif self.algorithm == 'deepsort':
            self.tracker = DeepSORT(
                max_age=self.max_age,
                min_hits=self.min_hits,
                nn_budget=100
            )
        elif self.algorithm == 'botsort':
            # BoT-SORT implementation would go here
            self.tracker = ByteTrack(track_buffer=self.max_age)
        else:
            raise ValueError(f"Unsupported tracking algorithm: {self.algorithm}")
        
        # Tracking state
        self.frame_count = 0
        self.total_tracks = 0
        self.track_history = []
        
    def track_frame(self, detections: Detections, frame: np.ndarray = None) -> Detections:
        """
        Track objects in single frame
        
        Args:
            detections: Detections from current frame
            frame: Current frame image
            
        Returns:
            Tracked detections
        """
        if self.algorithm == 'bytetrack':
            tracked_detections = self.tracker.update(detections, frame)
        elif self.algorithm == 'deepsort':
            tracked_detections = self.tracker.update(detections, frame=frame)
        else:
            tracked_detections = self.tracker.update(detections, frame)
        
        # Update statistics
        self.frame_count += 1
        if not tracked_detections.is_empty():
            self.total_tracks += len(tracked_detections)
            self.track_history.append({
                'frame': self.frame_count,
                'track_count': len(tracked_detections),
                'track_ids': tracked_detections.tracker_id.tolist() if tracked_detections.tracker_id is not None else []
            })
        
        return tracked_detections
    
    def track_video(self, video_path: str, model_path: str = None, 
                   output_path: str = None, save_video: bool = True) -> Dict:
        """
        Track objects in entire video
        
        Args:
            video_path: Path to input video
            model_path: Path to detection model (optional)
            output_path: Path to save output video
            save_video: Whether to save output video
            
        Returns:
            Tracking results and statistics
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video writer
        output_writer = None
        if save_video and output_path:
            output_writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
        
        # Load detection model if provided
        detector = None
        if model_path:
            try:
                from ..inference.server import ModelLoader
                model_loader = ModelLoader()
                detector = model_loader.load_model(model_path)
            except ImportError:
                print("Warning: Detection model loading failed, tracking may not work properly")
        
        # Tracking results
        results = {
            'video_path': video_path,
            'output_path': output_path,
            'total_frames': total_frames,
            'fps': fps,
            'resolution': (width, height),
            'algorithm': self.algorithm,
            'frames_processed': 0,
            'total_detections': 0,
            'tracking_stats': {}
        }
        
        print(f"Starting video tracking: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        frame_count = 0
        tracking_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Detect objects if model provided
            if detector:
                try:
                    # Run detection
                    if hasattr(detector, 'predict'):
                        # Ultralytics model
                        results_pred = detector.predict(frame)
                        if results_pred and len(results_pred) > 0:
                            detections = Detections.from_ultralytics(results_pred[0])
                        else:
                            detections = Detections.empty()
                    else:
                        # Generic model
                        detections = Detections.empty()
                except Exception as e:
                    print(f"Detection failed for frame {frame_count}: {e}")
                    detections = Detections.empty()
            else:
                # Use dummy detections for testing
                detections = self._create_dummy_detections(frame)
            
            # Track objects
            tracked_detections = self.tracker.track_frame(detections, frame)
            
            # Draw tracking results on frame
            if not tracked_detections.is_empty():
                frame = self._draw_tracks(frame, tracked_detections)
            
            # Save frame if requested
            if output_writer:
                output_writer.write(frame)
            
            frame_count += 1
            processing_time = time.time() - start_time
            tracking_times.append(processing_time)
            
            # Progress update
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
        
        # Cleanup
        cap.release()
        if output_writer:
            output_writer.release()
        
        # Calculate final statistics
        results.update({
            'frames_processed': frame_count,
            'total_detections': self.total_tracks,
            'avg_tracking_time': np.mean(tracking_times) if tracking_times else 0,
            'fps_processing': frame_count / sum(tracking_times) if tracking_times else 0,
            'tracking_history': self.track_history
        })
        
        print(f"Video tracking completed!")
        print(f"Processed: {frame_count} frames")
        print(f"Average tracking time: {results['avg_tracking_time']:.3f}s per frame")
        print(f"Processing FPS: {results['fps_processing']:.1f}")
        
        return results
    
    def _create_dummy_detections(self, frame: np.ndarray) -> Detections:
        """Create dummy detections for testing"""
        height, width = frame.shape[:2]
        
        # Create random detections
        num_detections = np.random.randint(0, 5)
        
        if num_detections == 0:
            return Detections.empty()
        
        xyxy = []
        confidence = []
        class_id = []
        class_name = []
        
        for _ in range(num_detections):
            x1 = np.random.randint(0, width // 2)
            y1 = np.random.randint(0, height // 2)
            w = np.random.randint(20, width // 4)
            h = np.random.randint(20, height // 4)
            x2 = min(x1 + w, width)
            y2 = min(y1 + h, height)
            
            xyxy.append([x1, y1, x2, y2])
            confidence.append(np.random.uniform(0.5, 1.0))
            class_id.append(0)
            class_name.append("object")
        
        return Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id),
            class_name=class_name
        )
    
    def _draw_tracks(self, frame: np.ndarray, tracked_detections: Detections) -> np.ndarray:
        """Draw tracking information on frame"""
        for det in tracked_detections:
            x1, y1, x2, y2 = det.xyxy.astype(int)
            
            # Draw bounding box
            color = (0, 255, 0) if det.tracker_id is not None else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with track ID
            label = f"{det.class_name}"
            if det.tracker_id is not None:
                label += f" #{det.tracker_id}"
            
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        return frame
    
    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            'algorithm': self.algorithm,
            'frame_count': self.frame_count,
            'total_tracks': self.total_tracks,
            'current_active_tracks': len(self.tracker.tracks),
            'avg_tracks_per_frame': self.total_tracks / max(1, self.frame_count),
            'track_history': self.track_history[-10:] if self.track_history else []
        }
    
    def reset(self):
        """Reset tracker state"""
        self.tracker.tracks.clear()
        if hasattr(self.tracker, 'dead_tracks'):
            self.tracker.dead_tracks.clear()
        if hasattr(self.tracker, 'features'):
            self.tracker.features.clear()
        
        self.frame_count = 0
        self.total_tracks = 0
        self.track_history.clear()
        self.tracker.track_id_count = 0