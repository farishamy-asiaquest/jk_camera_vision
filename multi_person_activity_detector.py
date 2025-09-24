import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from enum import Enum
from config import POSE_CONFIG, DETECTION_CONFIG, DISPLAY_CONFIG, ACTIVITY_COLORS, CAMERA_CONFIG, DEBUG_CONFIG, MULTI_PERSON_CONFIG

class Activity(Enum):
    WORKING = "Working"
    IDLE = "Idle"
    ON_PHONE = "On Phone"
    AWAY = "Away"

class PersonTracker:
    """Individual person tracker with their own activity detection"""
    def __init__(self, person_id, color):
        self.person_id = person_id
        self.color = color
        
        # Activity detection parameters
        self.movement_threshold = DETECTION_CONFIG['movement_threshold']
        self.idle_time_threshold = DETECTION_CONFIG['idle_time_threshold']
        self.phone_hand_threshold = DETECTION_CONFIG['phone_hand_threshold']
        
        # Tracking variables
        self.last_pose_landmarks = None
        self.movement_history = deque(maxlen=DETECTION_CONFIG['movement_history_size'])
        self.last_activity_time = time.time()
        self.current_activity = Activity.AWAY
        self.person_detected = False
        self.last_seen_time = time.time()
        
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
        self.last_seen_time = current_time
        
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

    def is_stale(self, current_time, max_age=5.0):
        """Check if this person tracker is stale (not seen for too long)"""
        return current_time - self.last_seen_time > max_age

class MultiPersonActivityDetector:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Use holistic for better multi-person detection
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Person tracking
        self.person_trackers = {}
        self.next_person_id = 1
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Colors for different persons
        self.person_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]

    def get_person_color(self, person_id):
        """Get color for a person based on their ID"""
        return self.person_colors[person_id % len(self.person_colors)]

    def find_closest_person(self, new_landmarks, threshold=None):
        """Find the closest existing person to new landmarks"""
        if not new_landmarks or not self.person_trackers:
            return None
        
        if threshold is None:
            threshold = MULTI_PERSON_CONFIG['person_association_threshold']
        
        min_distance = float('inf')
        closest_person = None
        
        for person_id, tracker in self.person_trackers.items():
            if tracker.last_pose_landmarks:
                # Calculate distance between nose positions as a simple metric
                new_nose = new_landmarks.landmark[0]  # Nose landmark
                old_nose = tracker.last_pose_landmarks.landmark[0]
                
                distance = np.sqrt((new_nose.x - old_nose.x)**2 + (new_nose.y - old_nose.y)**2)
                
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    closest_person = person_id
        
        return closest_person

    def create_new_person_tracker(self):
        """Create a new person tracker"""
        # Check if we've reached the maximum number of persons
        if len(self.person_trackers) >= MULTI_PERSON_CONFIG['max_persons']:
            return None
        
        person_id = self.next_person_id
        color = self.get_person_color(person_id)
        tracker = PersonTracker(person_id, color)
        self.person_trackers[person_id] = tracker
        self.next_person_id += 1
        return person_id

    def cleanup_stale_trackers(self):
        """Remove person trackers that haven't been seen recently"""
        current_time = time.time()
        stale_ids = []
        
        for person_id, tracker in self.person_trackers.items():
            if tracker.is_stale(current_time, MULTI_PERSON_CONFIG['person_stale_timeout']):
                stale_ids.append(person_id)
        
        for person_id in stale_ids:
            del self.person_trackers[person_id]

    def draw_pose_and_activity(self, image, person_id, landmarks, activity):
        """Draw pose landmarks and activity status for a specific person"""
        tracker = self.person_trackers[person_id]
        
        # Draw pose landmarks with person-specific color
        if landmarks and DISPLAY_CONFIG['show_landmarks']:
            landmark_spec = self.mp_drawing.DrawingSpec(
                color=tracker.color, 
                thickness=DISPLAY_CONFIG['landmark_thickness'], 
                circle_radius=DISPLAY_CONFIG['landmark_radius']
            )
            connection_spec = self.mp_drawing.DrawingSpec(
                color=tracker.color, 
                thickness=DISPLAY_CONFIG['connection_thickness']
            )
            
            self.mp_drawing.draw_landmarks(
                image, landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )
        
        # Draw person ID and activity status
        person_text = f"Person {person_id}: {activity.value}"
        activity_color = tracker.color
        
        # Position text based on person ID to avoid overlap
        y_offset = 10 + (person_id - 1) * 60
        
        # Background rectangle for text
        text_size = cv2.getTextSize(person_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(image, (10, y_offset), (text_size[0] + 20, text_size[1] + y_offset + 20), (0, 0, 0), -1)
        cv2.rectangle(image, (10, y_offset), (text_size[0] + 20, text_size[1] + y_offset + 20), activity_color, 2)
        
        # Person and activity text
        cv2.putText(image, person_text, (20, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, activity_color, 2)
        
        return image

    def draw_summary_info(self, image):
        """Draw summary information about all detected persons"""
        total_persons = len(self.person_trackers)
        active_persons = sum(1 for tracker in self.person_trackers.values() if tracker.person_detected)
        
        # Summary text
        summary_text = f"Total Persons: {total_persons} | Active: {active_persons}"
        cv2.putText(image, summary_text, (20, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS counter
        if DEBUG_CONFIG['show_fps']:
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.fps_start_time >= 1.0:
                fps = self.frame_count / (current_time - self.fps_start_time)
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(image, fps_text, (20, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                self.frame_count = 0
                self.fps_start_time = current_time
        
        return image

    def process_frame(self, frame):
        """Process a single frame and return annotated frame with activities for all persons"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with holistic
        results = self.holistic.process(rgb_frame)
        
        # Clean up stale trackers
        self.cleanup_stale_trackers()
        
        # Process pose landmarks if detected
        if results.pose_landmarks:
            # Try to match with existing person
            closest_person_id = self.find_closest_person(results.pose_landmarks)
            
            if closest_person_id is not None:
                # Update existing person
                person_id = closest_person_id
            else:
                # Create new person tracker
                person_id = self.create_new_person_tracker()
                if person_id is None:
                    # Max persons reached, skip this detection
                    pass  # Skip processing this detection
                    return frame, self.person_trackers
            
            # Update person's activity
            tracker = self.person_trackers[person_id]
            activity = tracker.classify_activity(results.pose_landmarks)
            
            # Draw pose and activity for this person
            frame = self.draw_pose_and_activity(frame, person_id, results.pose_landmarks, activity)
        else:
            # No pose detected, mark all persons as away
            for person_id, tracker in self.person_trackers.items():
                tracker.classify_activity(None)
        
        # Draw summary information
        frame = self.draw_summary_info(frame)
        
        return frame, self.person_trackers

def main():
    """Main function to run the multi-person pose activity detector"""
    # Initialize detector
    detector = MultiPersonActivityDetector()
    
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
    
    print("Multi-Person Pose Activity Detector Started")
    print("Press 'q' to quit")
    print("Activities detected: Working, Idle, On Phone, Away")
    print(f"Camera: {CAMERA_CONFIG['frame_width']}x{CAMERA_CONFIG['frame_height']} @ {CAMERA_CONFIG['fps']} FPS")
    print("Each person will be assigned a unique color and ID")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            annotated_frame, person_trackers = detector.process_frame(frame)
            
            # Display frame
            cv2.imshow('Multi-Person Pose Activity Detector', annotated_frame)
            
            # Print activities to console (optional)
            if DEBUG_CONFIG['print_activity']:
                activities = []
                for person_id, tracker in person_trackers.items():
                    if tracker.person_detected:
                        activities.append(f"Person {person_id}: {tracker.current_activity.value}")
                if activities:
                    print(f"\rActivities: {' | '.join(activities)}", end='', flush=True)
            
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
