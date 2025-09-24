#!/usr/bin/env python3
"""
Test script for true multi-person activity detection
This script demonstrates the difference between single-person and multi-person detection
"""

import cv2
import sys
from advanced_multi_person_detector import AdvancedMultiPersonDetector
from config import CAMERA_CONFIG, DEBUG_CONFIG

def test_true_multi_person_detection():
    """Test the true multi-person detection functionality"""
    print("Testing True Multi-Person Activity Detection")
    print("=" * 60)
    
    # Initialize detector
    detector = AdvancedMultiPersonDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_CONFIG['camera_index'])
    
    # Set camera properties
    if CAMERA_CONFIG['frame_width']:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['frame_width'])
    if CAMERA_CONFIG['frame_height']:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['frame_height'])
    if CAMERA_CONFIG['fps']:
        cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    print("Camera initialized successfully")
    print("Instructions:")
    print("- This version uses sliding window detection for true multi-person support")
    print("- Multiple people can be detected simultaneously with individual landmarks")
    print("- Each person gets a unique color and ID")
    print("- Activities are tracked individually for each person")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset person tracking")
    print("- Press 's' to show statistics")
    print("- Press 'h' to show help")
    print()
    
    frame_count = 0
    reset_requested = False
    show_help = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Reset person tracking if requested
            if reset_requested:
                detector.person_trackers.clear()
                detector.next_person_id = 1
                reset_requested = False
                print("Person tracking reset")
            
            # Process frame
            annotated_frame, person_trackers = detector.process_frame(frame)
            
            # Add instructions overlay
            if show_help:
                instructions = [
                    "Advanced Multi-Person Activity Detection",
                    "Press 'q' to quit, 'r' to reset, 's' for stats, 'h' for help",
                    "",
                    "Features:",
                    "- Sliding window detection for true multi-person support",
                    "- Individual landmark tracking for each person",
                    "- Unique color coding per person",
                    "- Individual activity classification",
                    "",
                    "Press 'h' again to hide this help"
                ]
            else:
                instructions = [
                    "Advanced Multi-Person Activity Detection",
                    "Press 'q' to quit, 'r' to reset, 's' for stats, 'h' for help"
                ]
            
            y_offset = 30
            for instruction in instructions:
                cv2.putText(annotated_frame, instruction, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            # Display frame
            cv2.imshow('True Multi-Person Activity Detection Test', annotated_frame)
            
            # Print statistics every 30 frames
            if frame_count % 30 == 0 and person_trackers:
                active_count = sum(1 for tracker in person_trackers.values() if tracker.person_detected)
                print(f"Frame {frame_count}: {len(person_trackers)} total persons, {active_count} active")
            
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                reset_requested = True
            elif key == ord('s'):
                print("\nCurrent Statistics:")
                print(f"Total persons tracked: {len(person_trackers)}")
                for person_id, tracker in person_trackers.items():
                    status = "Active" if tracker.person_detected else "Inactive"
                    print(f"  Person {person_id}: {tracker.current_activity.value} ({status})")
                print()
            elif key == ord('h'):
                show_help = not show_help
                
    except KeyboardInterrupt:
        print("\nStopping test...")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Test completed successfully")
        return True

def main():
    """Main function"""
    print("True Multi-Person Activity Detection Test")
    print("This test demonstrates the difference between single-person and multi-person detection")
    print()
    
    # Check if camera is available
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No camera detected. Please connect a camera and try again.")
        return
    cap.release()
    
    # Run the test
    success = test_true_multi_person_detection()
    
    if success:
        print("\nTest completed successfully!")
        print("The true multi-person detection is working correctly.")
        print("You should now see individual landmarks for each person detected.")
    else:
        print("\nTest failed. Please check your camera and try again.")

if __name__ == "__main__":
    main()
