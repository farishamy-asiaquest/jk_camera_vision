"""
Configuration file for Pose Activity Detector
Adjust these parameters to fine-tune detection accuracy
"""

# MediaPipe Pose Configuration
POSE_CONFIG = {
    'static_image_mode': False,
    'model_complexity': 1,  # 0, 1, or 2 (higher = more accurate but slower)
    'enable_segmentation': False,
    'min_detection_confidence': 0.5,  # Minimum confidence for pose detection
    'min_tracking_confidence': 0.5,   # Minimum confidence for pose tracking
}

# Activity Detection Parameters
DETECTION_CONFIG = {
    'movement_threshold': 0.02,        # Threshold for detecting movement (0.01-0.05)
    'idle_time_threshold': 3.0,        # Seconds of no movement to be considered idle
    'phone_hand_threshold': 0.3,       # Distance threshold for phone detection (0.2-0.5)
    'movement_history_size': 30,       # Number of frames to track for movement analysis
}

# Display Configuration
DISPLAY_CONFIG = {
    'show_landmarks': True,            # Show pose landmarks
    'show_connections': True,          # Show pose connections
    'landmark_color': (0, 255, 0),     # BGR color for landmarks (Green)
    'connection_color': (0, 255, 0),   # BGR color for connections (Green)
    'landmark_thickness': 2,           # Thickness of landmark circles
    'connection_thickness': 2,         # Thickness of connection lines
    'landmark_radius': 2,              # Radius of landmark circles
}

# Activity Colors (BGR format)
ACTIVITY_COLORS = {
    'WORKING': (0, 255, 0),      # Green
    'IDLE': (0, 255, 255),       # Yellow
    'ON_PHONE': (255, 0, 255),   # Magenta
    'AWAY': (0, 0, 255),         # Red
}

# Camera Configuration
CAMERA_CONFIG = {
    'camera_index': 0,            # Camera index (0 for default webcam)
    'frame_width': 640,           # Desired frame width
    'frame_height': 480,          # Desired frame height
    'fps': 30,                    # Desired FPS
}

# Multi-Person Detection Configuration
MULTI_PERSON_CONFIG = {
    'max_persons': 8,            # Maximum number of persons to track
    'person_association_threshold': 0.1,  # Threshold for associating new detections with existing persons
    'person_stale_timeout': 5.0,  # Seconds before a person is considered stale and removed
    'enable_person_tracking': True,  # Enable multi-person tracking
}

# Debug Configuration
DEBUG_CONFIG = {
    'print_activity': True,       # Print activity to console
    'show_fps': True,            # Show FPS counter
    'log_movement': False,       # Log movement data (for debugging)
}
