"""
Test script to verify MediaPipe and OpenCV installation
Run this before using the main application
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe version: {mp.__version__}")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    return True

def test_camera():
    """Test if camera is accessible"""
    print("\nTesting camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Camera not accessible")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("✗ Could not read from camera")
            cap.release()
            return False
        
        print(f"✓ Camera working - Frame size: {frame.shape}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_mediapipe_pose():
    """Test MediaPipe pose initialization"""
    print("\nTesting MediaPipe pose...")
    
    try:
        import mediapipe as mp
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✓ MediaPipe pose initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ MediaPipe pose test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Pose Activity Detector - Installation Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test camera
    if not test_camera():
        all_tests_passed = False
    
    # Test MediaPipe
    if not test_mediapipe_pose():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All tests passed! You can run the main application.")
        print("Run: python pose_activity_detector.py")
    else:
        print("✗ Some tests failed. Please check the installation.")
        print("Make sure to install requirements: pip install -r requirements.txt")
    print("=" * 50)

if __name__ == "__main__":
    main()
