import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from enum import Enum
from config import POSE_CONFIG, DETECTION_CONFIG, DISPLAY_CONFIG, ACTIVITY_COLORS, CAMERA_CONFIG, DEBUG_CONFIG

class Activity(Enum):
    WORKING = "Working"
    IDLE = "Idle"
    ON_PHONE = "On Phone"
    AWAY = "Away"

class PoseActivityDetector:
    def __init__(self):
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(**POSE_CONFIG)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Activity detection parameters from config
        self.movement_threshold = DETECTION_CONFIG['movement_threshold']
        self.idle_time_threshold = DETECTION_CONFIG['idle_time_threshold']
        self.phone_hand_threshold = DETECTION_CONFIG['phone_hand_threshold']
        
        # Tracking variables
        self.last_pose_landmarks = None
        self.movement_history = deque(maxlen=DETECTION_CONFIG['movement_history_size'])
        self.last_activity_time = time.time()
        self.current_activity = Activity.AWAY
        self.person_detected = False
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Define key landmark indices
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.NOSE = 0
        self.LEFT_EAR = 7
        self.RIGHT_EAR = 8

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two 3D points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

    def calculate_movement(self, current_landmarks, previous_landmarks):
        """Calculate average movement of key landmarks"""
        if previous_landmarks is None:
            return 0.0
        
        key_landmarks = [self.LEFT_WRIST, self.RIGHT_WRIST, self.LEFT_ELBOW, self.RIGHT_ELBOW]
        total_movement = 0.0
        
        for landmark_idx in key_landmarks:
            if (landmark_idx < len(current_landmarks.landmark) and 
                landmark_idx < len(previous_landmarks.landmark)):
                current_point = current_landmarks.landmark[landmark_idx]
                previous_point = previous_landmarks.landmark[landmark_idx]
                movement = self.calculate_distance(current_point, previous_point)
                total_movement += movement
        
        return total_movement / len(key_landmarks)

    def detect_phone_usage(self, landmarks):
        """Detect if person is using phone based on hand position relative to head"""
        if not landmarks:
            return False
        
        nose = landmarks.landmark[self.NOSE]
        left_wrist = landmarks.landmark[self.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.RIGHT_WRIST]
        
        # Check if either wrist is close to the head area
        left_distance = self.calculate_distance(nose, left_wrist)
        right_distance = self.calculate_distance(nose, right_wrist)
        
        return (left_distance < self.phone_hand_threshold or 
                right_distance < self.phone_hand_threshold)

    def detect_working_pose(self, landmarks):
        """Detect if person is in a working pose (hands near desk level)"""
        if not landmarks:
            return False
        
        left_wrist = landmarks.landmark[self.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.RIGHT_WRIST]
        left_shoulder = landmarks.landmark[self.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.RIGHT_SHOULDER]
        
        # Check if wrists are below shoulders (desk level)
        left_working = left_wrist.y > left_shoulder.y
        right_working = right_wrist.y > right_shoulder.y
        
        return left_working or right_working

    def classify_activity(self, landmarks):
        """Classify the current activity based on pose and movement"""
        current_time = time.time()
        
        if landmarks is None:
            self.person_detected = False
            self.current_activity = Activity.AWAY
            return self.current_activity
        
        self.person_detected = True
        
        # Calculate movement
        movement = self.calculate_movement(landmarks, self.last_pose_landmarks)
        self.movement_history.append(movement)
        
        # Check for phone usage
        if self.detect_phone_usage(landmarks):
            self.current_activity = Activity.ON_PHONE
            self.last_activity_time = current_time
            return self.current_activity
        
        # Check if in working pose
        if self.detect_working_pose(landmarks):
            # Calculate average movement over recent frames
            avg_movement = np.mean(list(self.movement_history)) if self.movement_history else 0
            
            if avg_movement > self.movement_threshold:
                self.current_activity = Activity.WORKING
                self.last_activity_time = current_time
            else:
                # Check if idle for too long
                if current_time - self.last_activity_time > self.idle_time_threshold:
                    self.current_activity = Activity.IDLE
                else:
                    self.current_activity = Activity.WORKING
        else:
            # Not in working pose, could be standing or moving around
            self.current_activity = Activity.IDLE
        
        self.last_pose_landmarks = landmarks
        return self.current_activity

    def draw_pose_and_activity(self, image, landmarks, activity):
        """Draw pose landmarks and activity status on the image"""
        # Draw pose landmarks
        if landmarks and DISPLAY_CONFIG['show_landmarks']:
            landmark_spec = self.mp_drawing.DrawingSpec(
                color=DISPLAY_CONFIG['landmark_color'], 
                thickness=DISPLAY_CONFIG['landmark_thickness'], 
                circle_radius=DISPLAY_CONFIG['landmark_radius']
            )
            connection_spec = self.mp_drawing.DrawingSpec(
                color=DISPLAY_CONFIG['connection_color'], 
                thickness=DISPLAY_CONFIG['connection_thickness']
            )
            
            self.mp_drawing.draw_landmarks(
                image, landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )
        
        # Draw activity status
        activity_text = f"Activity: {activity.value}"
        activity_color = self.get_activity_color(activity)
        
        # Background rectangle for text
        text_size = cv2.getTextSize(activity_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.rectangle(image, (10, 10), (text_size[0] + 20, text_size[1] + 30), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (text_size[0] + 20, text_size[1] + 30), activity_color, 2)
        
        # Activity text
        cv2.putText(image, activity_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, activity_color, 2)
        
        # Additional info
        info_text = f"Person Detected: {'Yes' if self.person_detected else 'No'}"
        cv2.putText(image, info_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS counter
        if DEBUG_CONFIG['show_fps']:
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.fps_start_time >= 1.0:
                fps = self.frame_count / (current_time - self.fps_start_time)
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(image, fps_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                self.frame_count = 0
                self.fps_start_time = current_time
        
        return image

    def get_activity_color(self, activity):
        """Get color for activity status display"""
        return ACTIVITY_COLORS.get(activity.name, (255, 255, 255))

    def process_frame(self, frame):
        """Process a single frame and return annotated frame with activity"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Classify activity
        activity = self.classify_activity(results.pose_landmarks)
        
        # Draw pose and activity
        annotated_frame = self.draw_pose_and_activity(frame, results.pose_landmarks, activity)
        
        return annotated_frame, activity

def main():
    """Main function to run the pose activity detector"""
    # Initialize detector
    detector = PoseActivityDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_CONFIG['camera_index'])
    
    # Set camera properties if specified
    if CAMERA_CONFIG['frame_width']:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['frame_width'])
    if CAMERA_CONFIG['frame_height']:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['frame_height'])
    if CAMERA_CONFIG['fps']:
        cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Pose Activity Detector Started")
    print("Press 'q' to quit")
    print("Activities detected: Working, Idle, On Phone, Away")
    print(f"Camera: {CAMERA_CONFIG['frame_width']}x{CAMERA_CONFIG['frame_height']} @ {CAMERA_CONFIG['fps']} FPS")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            annotated_frame, activity = detector.process_frame(frame)
            
            # Display frame
            cv2.imshow('Pose Activity Detector', annotated_frame)
            
            # Print activity to console (optional)
            if DEBUG_CONFIG['print_activity']:
                print(f"\rCurrent Activity: {activity.value}", end='', flush=True)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping detector...")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nDetector stopped successfully")

if __name__ == "__main__":
    main()
