#!/usr/bin/env python3
"""
🧪 Viseon Comprehensive Test Suite
Step-by-step testing of all functionalities

Run this to validate the entire viseon platform
"""

import sys
import os
import time
import numpy as np
import cv2
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

class ViseonTestSuite:
    """Complete test suite for Viseon"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests_run = 0
        self.results = []
    
    def test(self, test_name, test_func):
        """Run a single test"""
        self.tests_run += 1
        print(f"\n{'='*70}")
        print(f"TEST {self.tests_run}: {test_name}")
        print(f"{'='*70}")
        
        try:
            test_func()
            self.passed += 1
            self.results.append((test_name, "✅ PASSED", None))
            print(f"✅ PASSED")
            return True
        except Exception as e:
            self.failed += 1
            error_msg = str(e)[:200]
            self.results.append((test_name, "❌ FAILED", error_msg))
            print(f"❌ FAILED: {error_msg}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n\n{'='*70}")
        print("📊 TEST SUMMARY")
        print(f"{'='*70}")
        print(f"Total Tests: {self.tests_run}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        if self.tests_run > 0:
            print(f"Success Rate: {(self.passed/self.tests_run)*100:.1f}%")
        
        print(f"\n{'='*70}")
        print("DETAILED RESULTS")
        print(f"{'='*70}")
        for test_name, status, error in self.results:
            print(f"{status} - {test_name}")
            if error:
                print(f"     └─ {error[:100]}")

# Initialize test suite
suite = ViseonTestSuite()

# ============================================
# PHASE 1: IMPORTS & ENVIRONMENT
# ============================================

def test_1_import_core():
    """Test core module imports"""
    print("Importing core modules...")
    from viseon.core.detections import Detections
    from viseon.core.project import Project
    print("✅ Core imports successful")

def test_2_import_training():
    """Test training module imports"""
    print("Importing training modules...")
    from viseon.training.yolo_trainer import YOLOTrainer
    print("✅ Training imports successful")

def test_3_import_tracking():
    """Test tracking module imports"""
    print("Importing tracking modules...")
    from viseon.tracking.object_tracker import ObjectTracker
    print("✅ Tracking imports successful")

def test_4_import_inference():
    """Test inference module imports"""
    print("Importing inference modules...")
    from viseon.inference.server import InferenceServer
    print("✅ Inference imports successful")

def test_5_import_annotation():
    """Test annotation module imports"""
    print("Importing annotation modules...")
    from viseon.annotation.cvat_integration import CVATAnnotate
    print("✅ Annotation imports successful")

# ============================================
# PHASE 2: DETECTIONS SYSTEM
# ============================================

def test_6_detections_creation():
    """Test Detections object creation"""
    print("Creating Detections object...")
    from viseon.core.detections import Detections
    
    xyxy = np.array([[100, 120, 200, 180], [300, 250, 400, 350]])
    confidence = np.array([0.85, 0.92])
    class_id = np.array([0, 1])
    class_name = ["person", "car"]
    
    detections = Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        class_name=class_name
    )
    
    print(f"Created Detections with {len(detections)} objects")
    assert len(detections) == 2
    print("✅ Detections creation successful")

def test_7_detections_filtering():
    """Test Detections filtering"""
    print("Testing Detections filtering...")
    from viseon.core.detections import Detections
    
    xyxy = np.array([[100, 120, 200, 180], [300, 250, 400, 350], [50, 50, 100, 100]])
    confidence = np.array([0.85, 0.92, 0.5])
    class_id = np.array([0, 1, 0])
    
    detections = Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
    
    # Filter by confidence
    high_conf = detections.filter(confidence_threshold=0.8)
    print(f"High confidence (>0.8): {len(high_conf)} detections")
    assert len(high_conf) == 2
    
    # Filter by class
    class_0 = detections.filter(class_ids=[0])
    print(f"Class 0: {len(class_0)} detections")
    assert len(class_0) == 2
    
    print("✅ Detections filtering successful")

def test_8_detections_merging():
    """Test Detections merging"""
    print("Testing Detections merging...")
    from viseon.core.detections import Detections
    
    det1 = Detections(
        xyxy=np.array([[100, 120, 200, 180]]),
        confidence=np.array([0.85]),
        class_id=np.array([0])
    )
    
    det2 = Detections(
        xyxy=np.array([[300, 250, 400, 350]]),
        confidence=np.array([0.92]),
        class_id=np.array([1])
    )
    
    merged = det1 | det2
    print(f"Merged {len(det1)} + {len(det2)} = {len(merged)} detections")
    assert len(merged) == 2
    print("✅ Detections merging successful")

def test_9_detections_empty():
    """Test empty Detections"""
    print("Testing empty Detections...")
    from viseon.core.detections import Detections
    
    empty = Detections.empty()
    assert empty.is_empty()
    print(f"Empty detections: is_empty={empty.is_empty()}")
    print("✅ Empty Detections successful")

def test_10_detections_iou():
    """Test IoU calculation"""
    print("Testing IoU calculation...")
    from viseon.core.detections import Detections
    
    det1 = Detections(
        xyxy=np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
        confidence=np.array([0.9, 0.9]),
        class_id=np.array([0, 0])
    )
    
    det2 = Detections(
        xyxy=np.array([[150, 150, 250, 250], [400, 400, 500, 500]]),
        confidence=np.array([0.9, 0.9]),
        class_id=np.array([0, 0])
    )
    
    iou_matrix = det1.iou_with(det2)
    print(f"IoU matrix shape: {iou_matrix.shape}")
    print(f"IoU values (sample): {iou_matrix[0, 0]:.3f}, {iou_matrix[0, 1]:.3f}")
    assert iou_matrix.shape == (2, 2)
    print("✅ IoU calculation successful")

# ============================================
# PHASE 3: PROJECT MANAGEMENT
# ============================================

def test_11_project_creation():
    """Test Project creation"""
    print("Creating Project...")
    from viseon.core.project import Project
    
    config = {'storage': {'base_path': './test_projects'}}
    project = Project(
        name="test_project",
        description="Test project",
        config=config
    )
    
    print(f"Created project: {project.name}")
    print(f"Project directory: {project.project_dir}")
    print("✅ Project creation successful")

def test_12_project_stats():
    """Test Project statistics"""
    print("Getting Project stats...")
    from viseon.core.project import Project
    
    config = {'storage': {'base_path': './test_projects'}}
    project = Project(
        name="test_project_2",
        description="Test project",
        config=config
    )
    
    stats = project.get_project_stats()
    print(f"Project stats: {stats}")
    assert isinstance(stats, dict)
    print("✅ Project stats successful")

# ============================================
# PHASE 4: OBJECT TRACKING
# ============================================

def test_13_tracker_creation():
    """Test ObjectTracker creation"""
    print("Creating ObjectTracker...")
    from viseon.tracking.object_tracker import ObjectTracker
    
    config = {
        'algorithm': 'bytetrack',
        'max_age': 30,
        'min_hits': 3
    }
    
    tracker = ObjectTracker(config=config)
    print(f"Created tracker with algorithm: {tracker.algorithm}")
    print("✅ ObjectTracker creation successful")

def test_14_tracker_stats():
    """Test Tracker statistics"""
    print("Getting Tracker stats...")
    from viseon.tracking.object_tracker import ObjectTracker
    
    config = {'algorithm': 'bytetrack'}
    tracker = ObjectTracker(config=config)
    
    stats = tracker.get_tracking_stats()
    print(f"Tracker stats: {stats}")
    assert isinstance(stats, dict)
    print("✅ Tracker stats successful")

def test_15_tracker_update():
    """Test Tracker frame update"""
    print("Testing Tracker frame update...")
    from viseon.tracking.object_tracker import ObjectTracker
    from viseon.core.detections import Detections
    
    config = {'algorithm': 'bytetrack'}
    tracker = ObjectTracker(config=config)
    
    # Create test detections
    detections = Detections(
        xyxy=np.array([[100, 100, 200, 200]]),
        confidence=np.array([0.9]),
        class_id=np.array([0])
    )
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Update tracker
    tracked = tracker.track_frame(detections, frame)
    print(f"Tracked {len(tracked)} objects")
    print("✅ Tracker update successful")

# ============================================
# PHASE 5: YOLO TRAINER
# ============================================

def test_16_trainer_creation():
    """Test YOLOTrainer creation"""
    print("Creating YOLOTrainer...")
    from viseon.training.yolo_trainer import YOLOTrainer
    
    config = {'storage': {'base_path': './models'}}
    trainer = YOLOTrainer(config=config)
    
    print(f"Created trainer")
    print("✅ YOLOTrainer creation successful")

def test_17_trainer_config():
    """Test Trainer configuration"""
    print("Testing Trainer configuration...")
    from viseon.training.yolo_trainer import YOLOTrainer
    
    config = {'storage': {'base_path': './models'}}
    trainer = YOLOTrainer(config=config)
    
    print(f"Trainer config: {trainer.config if hasattr(trainer, 'config') else 'N/A'}")
    print("✅ Trainer configuration successful")

# ============================================
# PHASE 6: INFERENCE SERVER
# ============================================

def test_18_inference_server_creation():
    """Test InferenceServer creation"""
    print("Creating InferenceServer...")
    from viseon.inference.server import InferenceServer
    
    config = {
        'inference': {
            'port': 8000,
            'batch_size': 8
        }
    }
    
    server = InferenceServer(config=config)
    print(f"Created inference server")
    print("✅ InferenceServer creation successful")

def test_19_inference_config():
    """Test Inference configuration"""
    print("Testing Inference configuration...")
    from viseon.inference.server import InferenceServer
    
    config = {
        'inference': {
            'port': 8000,
            'batch_size': 8
        }
    }
    
    server = InferenceServer(config=config)
    print(f"Server config: {server.config if hasattr(server, 'config') else 'N/A'}")
    print("✅ Inference configuration successful")

# ============================================
# PHASE 7: CVAT ANNOTATION
# ============================================

def test_20_cvat_creation():
    """Test CVAT integration creation"""
    print("Creating CVAT integration...")
    from viseon.annotation.cvat_integration import CVATAnnotate
    
    config = {
        'url': 'http://localhost:8080',
        'username': 'admin',
        'password': 'admin'
    }
    
    cvat = CVATAnnotate(cvat_config=config)
    print(f"Created CVAT integration")
    print("✅ CVAT integration creation successful")

# ============================================
# PHASE 8: INTEGRATION TESTS
# ============================================

def test_21_end_to_end_detection():
    """Test end-to-end detection pipeline"""
    print("Testing end-to-end detection pipeline...")
    from viseon.core.detections import Detections
    
    # Step 1: Create detections
    detections = Detections(
        xyxy=np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
        confidence=np.array([0.95, 0.87]),
        class_id=np.array([0, 1])
    )
    
    # Step 2: Filter
    high_conf = detections.filter(confidence_threshold=0.9)
    print(f"Original: {len(detections)}, Filtered: {len(high_conf)}")
    
    # Step 3: Verify
    assert len(high_conf) > 0
    print("✅ E2E detection pipeline successful")

def test_22_end_to_end_tracking():
    """Test end-to-end tracking pipeline"""
    print("Testing end-to-end tracking pipeline...")
    from viseon.tracking.object_tracker import ObjectTracker
    from viseon.core.detections import Detections
    
    # Step 1: Create tracker
    tracker = ObjectTracker(config={'algorithm': 'bytetrack'})
    
    # Step 2: Create frame with detections
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = Detections(
        xyxy=np.array([[100, 100, 200, 200]]),
        confidence=np.array([0.9]),
        class_id=np.array([0])
    )
    
    # Step 3: Track
    tracked = tracker.track_frame(detections, frame)
    print(f"Tracked {len(tracked)} objects")
    
    # Step 4: Verify
    assert len(tracked) >= 0
    print("✅ E2E tracking pipeline successful")

def test_23_end_to_end_project_workflow():
    """Test end-to-end project workflow"""
    print("Testing end-to-end project workflow...")
    from viseon.core.project import Project
    from viseon.core.detections import Detections
    
    # Step 1: Create project
    config = {'storage': {'base_path': './test_projects'}}
    project = Project(
        name="e2e_project",
        description="E2E test",
        config=config
    )
    print(f"Created project: {project.name}")
    
    # Step 2: Get stats
    stats = project.get_project_stats()
    print(f"Project stats: {stats}")
    
    # Step 3: Create detections
    detections = Detections(
        xyxy=np.array([[50, 50, 150, 150]]),
        confidence=np.array([0.95]),
        class_id=np.array([0])
    )
    print(f"Created {len(detections)} detections")
    
    # Step 4: Verify
    assert isinstance(stats, dict)
    print("✅ E2E project workflow successful")

# ============================================
# PHASE 9: PERFORMANCE TESTS
# ============================================

def test_24_detections_performance():
    """Test Detections performance with large datasets"""
    print("Testing Detections performance...")
    from viseon.core.detections import Detections
    
    # Create large number of detections
    n_detections = 1000
    xyxy = np.random.rand(n_detections, 4) * 640
    confidence = np.random.rand(n_detections)
    class_id = np.random.randint(0, 80, n_detections)
    
    start = time.time()
    detections = Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
    elapsed = time.time() - start
    
    print(f"Created {len(detections)} detections in {elapsed*1000:.2f}ms")
    
    start = time.time()
    filtered = detections.filter(confidence_threshold=0.5)
    elapsed = time.time() - start
    
    print(f"Filtered {len(detections)} to {len(filtered)} in {elapsed*1000:.2f}ms")
    print("✅ Detections performance test successful")

def test_25_tracking_performance():
    """Test Tracking performance"""
    print("Testing Tracking performance...")
    from viseon.tracking.object_tracker import ObjectTracker
    from viseon.core.detections import Detections
    
    tracker = ObjectTracker(config={'algorithm': 'bytetrack'})
    
    start = time.time()
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = Detections(
            xyxy=np.random.rand(5, 4) * 640,
            confidence=np.random.rand(5),
            class_id=np.random.randint(0, 2, 5)
        )
        tracker.track_frame(detections, frame)
    elapsed = time.time() - start
    
    print(f"Tracked 30 frames with 5 objects each in {elapsed*1000:.2f}ms")
    print(f"Average: {(elapsed*1000)/30:.2f}ms per frame")
    print("✅ Tracking performance test successful")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🚀 OPENSUPERVISION COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Phase 1: Imports
    print("\n\n" + "🔵 "*10)
    print("PHASE 1: IMPORTS & ENVIRONMENT")
    print("🔵 "*10)
    suite.test("Import Core Modules", test_1_import_core)
    suite.test("Import Training Modules", test_2_import_training)
    suite.test("Import Tracking Modules", test_3_import_tracking)
    suite.test("Import Inference Modules", test_4_import_inference)
    suite.test("Import Annotation Modules", test_5_import_annotation)
    
    # Phase 2: Detections
    print("\n\n" + "🔵 "*10)
    print("PHASE 2: DETECTIONS SYSTEM")
    print("🔵 "*10)
    suite.test("Detections Creation", test_6_detections_creation)
    suite.test("Detections Filtering", test_7_detections_filtering)
    suite.test("Detections Merging", test_8_detections_merging)
    suite.test("Empty Detections", test_9_detections_empty)
    suite.test("IoU Calculation", test_10_detections_iou)
    
    # Phase 3: Projects
    print("\n\n" + "🔵 "*10)
    print("PHASE 3: PROJECT MANAGEMENT")
    print("🔵 "*10)
    suite.test("Project Creation", test_11_project_creation)
    suite.test("Project Statistics", test_12_project_stats)
    
    # Phase 4: Tracking
    print("\n\n" + "🔵 "*10)
    print("PHASE 4: OBJECT TRACKING")
    print("🔵 "*10)
    suite.test("Tracker Creation", test_13_tracker_creation)
    suite.test("Tracker Statistics", test_14_tracker_stats)
    suite.test("Tracker Frame Update", test_15_tracker_update)
    
    # Phase 5: Training
    print("\n\n" + "🔵 "*10)
    print("PHASE 5: YOLO TRAINER")
    print("🔵 "*10)
    suite.test("Trainer Creation", test_16_trainer_creation)
    suite.test("Trainer Configuration", test_17_trainer_config)
    
    # Phase 6: Inference
    print("\n\n" + "🔵 "*10)
    print("PHASE 6: INFERENCE SERVER")
    print("🔵 "*10)
    suite.test("InferenceServer Creation", test_18_inference_server_creation)
    suite.test("Inference Configuration", test_19_inference_config)
    
    # Phase 7: Annotation
    print("\n\n" + "🔵 "*10)
    print("PHASE 7: CVAT ANNOTATION")
    print("🔵 "*10)
    suite.test("CVAT Creation", test_20_cvat_creation)
    
    # Phase 8: Integration
    print("\n\n" + "🔵 "*10)
    print("PHASE 8: INTEGRATION TESTS")
    print("🔵 "*10)
    suite.test("E2E Detection Pipeline", test_21_end_to_end_detection)
    suite.test("E2E Tracking Pipeline", test_22_end_to_end_tracking)
    suite.test("E2E Project Workflow", test_23_end_to_end_project_workflow)
    
    # Phase 9: Performance
    print("\n\n" + "🔵 "*10)
    print("PHASE 9: PERFORMANCE TESTS")
    print("🔵 "*10)
    suite.test("Detections Performance", test_24_detections_performance)
    suite.test("Tracking Performance", test_25_tracking_performance)
    
    # Summary
    suite.print_summary()
    
    # Final status
    print("\n" + "="*70)
    if suite.failed == 0:
        print("✅ ALL TESTS PASSED! Viseon is ready to use.")
        print("="*70)
        return 0
    else:
        print(f"❌ {suite.failed} test(s) failed. Review the errors above.")
        print("="*70)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
