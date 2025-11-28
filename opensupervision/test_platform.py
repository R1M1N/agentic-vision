#!/usr/bin/env python3
"""
OpenSupervision Quick Test
Tests the basic functionality of the platform
"""

import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test core imports
        from opensupervision.core.detections import Detections
        from opensupervision.core.project import Project
        print("✓ Core imports successful")
        
        # Test annotation imports
        from opensupervision.annotation.cvat_integration import CVATAnnotate
        print("✓ Annotation imports successful")
        
        # Test training imports
        from opensupervision.training.yolo_trainer import YOLOTrainer
        print("✓ Training imports successful")
        
        # Test inference imports
        from opensupervision.inference.server import InferenceServer
        print("✓ Inference imports successful")
        
        # Test tracking imports
        from opensupervision.tracking.object_tracker import ObjectTracker
        print("✓ Tracking imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_detections():
    """Test the Detections system"""
    print("\nTesting Detections system...")
    
    try:
        from opensupervision.core.detections import Detections
        
        # Create sample detections
        detections = Detections(
            xyxy=np.array([[100, 120, 200, 180], [300, 250, 400, 350]]),
            confidence=np.array([0.85, 0.92]),
            class_id=np.array([0, 1]),
            class_name=["person", "car"]
        )
        
        print(f"✓ Created detections: {detections}")
        
        # Test filtering
        filtered = detections.filter(confidence_threshold=0.8)
        print(f"✓ Filtered detections: {len(filtered)} remaining")
        
        # Test merging
        detections2 = Detections(
            xyxy=np.array([[150, 200, 250, 280]]),
            confidence=np.array([0.75]),
            class_id=np.array([0]),
            class_name=["person"]
        )
        
        merged = detections | detections2
        print(f"✓ Merged detections: {len(merged)} total")
        
        # Test empty detections
        empty = Detections.empty()
        print(f"✓ Empty detections: {empty.is_empty()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Detections test failed: {e}")
        return False


def test_project():
    """Test the Project system"""
    print("\nTesting Project system...")
    
    try:
        from opensupervision.core.project import Project
        
        # Create sample project
        config = {'storage': {'base_path': './test_data'}}
        project = Project("test_project", "Test description", config)
        
        print(f"✓ Created project: {project}")
        
        # Test project info
        info = project.get_project_stats()
        print(f"✓ Project stats: {info}")
        
        return True
        
    except Exception as e:
        print(f"✗ Project test failed: {e}")
        return False


def test_tracking():
    """Test the tracking system"""
    print("\nTesting Tracking system...")
    
    try:
        from opensupervision.tracking.object_tracker import ObjectTracker
        
        # Create tracker
        config = {'algorithm': 'bytetrack', 'max_age': 30, 'min_hits': 3}
        tracker = ObjectTracker(config)
        
        print(f"✓ Created tracker: {tracker.algorithm}")
        
        # Test stats
        stats = tracker.get_tracking_stats()
        print(f"✓ Tracker stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Tracking test failed: {e}")
        return False


def test_main_platform():
    """Test the main OpenSupervision platform"""
    print("\nTesting main platform...")
    
    try:
        # Test main import
        import opensupervision as osv
        print("✓ OpenSupervision imported successfully")
        
        # Create platform instance
        platform = osv.OpenSupervision()
        print(f"✓ Platform created: {platform}")
        
        # Test configuration
        config = platform.config
        print(f"✓ Configuration loaded: {len(config)} sections")
        
        return True
        
    except Exception as e:
        print(f"✗ Main platform test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 OpenSupervision Platform Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_detections,
        test_project,
        test_tracking,
        test_main_platform
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print("❌ Test failed")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! OpenSupervision is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run complete example: python examples/complete_example.py")
        print("3. Start deployment: docker-compose up -d")
        return True
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)