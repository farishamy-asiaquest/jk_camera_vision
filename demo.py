"""
Demo script for Pose Activity Detector
This script provides a simple interface to test different configurations
"""

import cv2
import time
from pose_activity_detector import PoseActivityDetector, Activity
from config import DETECTION_CONFIG, DISPLAY_CONFIG

def demo_activity_detection():
    """Run a demo of the activity detection with different configurations"""
    
    print("=" * 60)
    print("Pose Activity Detector - Demo Mode")
    print("=" * 60)
    print("This demo will show you how different parameters affect detection")
    print("Press 'q' to quit, 's' to save current frame, 'r' to reset")
    print("=" * 60)
    
    # Initialize detector
    detector = PoseActivityDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Demo configurations
    demo_configs = [
        {
            'name': 'Sensitive (High Movement Detection)',
            'movement_threshold': 0.01,
            'idle_time_threshold': 2.0,
            'phone_hand_threshold': 0.25
        },
        {
            'name': 'Default (Balanced)',
            'movement_threshold': 0.02,
            'idle_time_threshold': 3.0,
            'phone_hand_threshold': 0.3
        },
        {
            'name': 'Conservative (Low Movement Detection)',
            'movement_threshold': 0.05,
            'idle_time_threshold': 5.0,
            'phone_hand_threshold': 0.4
        }
    ]
    
    current_config = 0
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update detector parameters based on current config
            config = demo_configs[current_config]
            detector.movement_threshold = config['movement_threshold']
            detector.idle_time_threshold = config['idle_time_threshold']
            detector.phone_hand_threshold = config['phone_hand_threshold']
            
            # Process frame
            annotated_frame, activity = detector.process_frame(frame)
            
            # Add configuration info to frame
            config_text = f"Config: {config['name']}"
            cv2.putText(annotated_frame, config_text, (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add parameter info
            param_text = f"Movement: {config['movement_threshold']:.3f} | Idle: {config['idle_time_threshold']:.1f}s | Phone: {config['phone_hand_threshold']:.2f}"
            cv2.putText(annotated_frame, param_text, (20, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Add instructions
            cv2.putText(annotated_frame, "Press '1', '2', '3' to switch configs", (20, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Display frame
            cv2.imshow('Pose Activity Detector - Demo', annotated_frame)
            
            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                print(f"\rFPS: {fps:.1f} | Activity: {activity.value} | Config: {config['name']}", end='', flush=True)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_config = 0
                print(f"\nSwitched to: {demo_configs[current_config]['name']}")
            elif key == ord('2'):
                current_config = 1
                print(f"\nSwitched to: {demo_configs[current_config]['name']}")
            elif key == ord('3'):
                current_config = 2
                print(f"\nSwitched to: {demo_configs[current_config]['name']}")
            elif key == ord('s'):
                # Save current frame
                filename = f"demo_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"\nFrame saved as: {filename}")
            elif key == ord('r'):
                # Reset detector
                detector.last_pose_landmarks = None
                detector.movement_history.clear()
                detector.last_activity_time = time.time()
                detector.current_activity = Activity.AWAY
                print("\nDetector reset")
                
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo completed")

def show_activity_examples():
    """Show examples of different activities"""
    print("\n" + "=" * 60)
    print("Activity Detection Examples")
    print("=" * 60)
    print("WORKING:")
    print("  - Sit at a desk with your hands on keyboard/mouse")
    print("  - Make small typing movements")
    print("  - Keep hands below shoulder level")
    print()
    print("IDLE:")
    print("  - Stay in working position but stop moving")
    print("  - Keep hands still for 3+ seconds")
    print()
    print("ON PHONE:")
    print("  - Hold your hand near your ear or face")
    print("  - Bring phone close to your head")
    print()
    print("AWAY:")
    print("  - Move out of camera view")
    print("  - Or turn away from camera")
    print("=" * 60)

if __name__ == "__main__":
    show_activity_examples()
    input("\nPress Enter to start demo...")
    demo_activity_detection()
