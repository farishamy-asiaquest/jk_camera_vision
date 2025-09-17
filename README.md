# Human Pose Estimation & Activity Recognition App

A real-time human pose estimation and activity recognition application using Python and MediaPipe. This app detects and classifies human activities including Working, Idle, On Phone, and Away states.

## Features

- **Real-time Pose Estimation**: Uses MediaPipe to detect and track human body landmarks
- **Activity Recognition**: Classifies activities based on pose and movement patterns:
  - **Working**: Person is in sitting pose with active hand movements (typing/working)
  - **Idle**: Person is detected but with minimal movement for extended periods
  - **On Phone**: Person's hand is positioned near their head/face
  - **Away**: No person detected in the camera view
- **Visual Feedback**: Displays pose skeleton overlay and current activity status
- **Configurable Parameters**: Adjustable thresholds for movement detection and activity classification

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main application:

```bash
python pose_activity_detector.py
```

### Controls

- **'q' key**: Quit the application
- **Ctrl+C**: Stop the application

## How It Works

### 1. Pose Estimation

- Uses MediaPipe Pose to detect 33 body landmarks in real-time
- Tracks key joints like wrists, elbows, shoulders, and head
- Creates a digital skeleton overlay on the video feed

### 2. Activity Classification

The app uses simple but effective rules based on landmark positions and movement:

- **Working Detection**:
  - Person is in sitting pose (wrists below shoulders)
  - Active hand movements detected over time
- **Idle Detection**:
  - Person detected but minimal movement for 3+ seconds
  - Still in working pose but not actively moving
- **Phone Usage Detection**:
  - Hand landmarks are close to head/face area
  - Uses distance threshold between nose and wrist landmarks
- **Away Detection**:
  - No person detected in the camera view

### 3. Movement Analysis

- Tracks movement of key landmarks (wrists, elbows) over time
- Uses a sliding window to calculate average movement
- Compares movement against configurable thresholds

## Configuration

You can adjust detection parameters in the `PoseActivityDetector` class:

```python
self.movement_threshold = 0.02      # Movement sensitivity
self.idle_time_threshold = 3.0      # Seconds before idle detection
self.phone_hand_threshold = 0.3     # Distance for phone detection
```

## Technical Details

- **Framework**: MediaPipe, OpenCV, NumPy
- **Camera**: Uses default webcam (index 0)
- **Performance**: Optimized for real-time processing
- **Landmarks**: Tracks 33 body keypoints with confidence scores

## Future Enhancements

- Machine learning-based activity classification
- Multiple person detection and tracking
- Custom activity definitions
- Data logging and analytics
- Web interface for remote monitoring

## Troubleshooting

- **Camera not found**: Ensure your webcam is connected and not being used by another application
- **Poor detection**: Ensure good lighting and clear view of the person
- **Performance issues**: Try reducing camera resolution or adjusting detection confidence thresholds
