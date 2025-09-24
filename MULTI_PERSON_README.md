# Multi-Person Activity Detection

This enhanced version of the pose activity detector can now detect and track multiple persons simultaneously, each with their own individual activity classification.

## Features

### Multi-Person Detection

- **Simultaneous Tracking**: Detects and tracks up to 8 persons at once
- **Unique Identification**: Each person gets a unique ID and color
- **Individual Activity Tracking**: Each person's activity is tracked independently
- **Automatic Person Management**: Automatically adds new persons and removes stale ones

### Activity Classification (Per Person)

- **Working**: Person is in sitting pose with active hand movements
- **Idle**: Person is detected but with minimal movement
- **On Phone**: Person's hand is positioned near their head/face
- **Away**: Person is not detected or has been away for too long

### Visual Features

- **Color-Coded Display**: Each person has a unique color for landmarks and text
- **Individual Status**: Shows activity status for each person separately
- **Summary Information**: Displays total and active person counts
- **Real-time Updates**: All information updates in real-time

## Files

### Core Files

- `multi_person_activity_detector.py` - Main multi-person detection implementation
- `config.py` - Configuration file with multi-person settings
- `test_multi_person.py` - Test script to demonstrate functionality

### Original Files (Still Available)

- `pose_activity_detector.py` - Original single-person detector
- `demo.py` - Original demo script

## Usage

### Running Multi-Person Detection

```bash
python multi_person_activity_detector.py
```

### Running the Test Script

```bash
python test_multi_person.py
```

### Controls

- **'q' key**: Quit the application
- **'r' key**: Reset person tracking (in test script)
- **'s' key**: Show statistics (in test script)
- **Ctrl+C**: Stop the application

## Configuration

### Multi-Person Settings (in config.py)

```python
MULTI_PERSON_CONFIG = {
    'max_persons': 8,                    # Maximum number of persons to track
    'person_association_threshold': 0.1,  # Threshold for associating detections
    'person_stale_timeout': 5.0,         # Seconds before person is considered stale
    'enable_person_tracking': True,       # Enable multi-person tracking
}
```

### Adjustable Parameters

- **max_persons**: Maximum number of persons to track simultaneously (default: 8)
- **person_association_threshold**: Distance threshold for matching new detections with existing persons (default: 0.1)
- **person_stale_timeout**: Time in seconds before a person is removed if not detected (default: 5.0)

## How It Works

### 1. Person Detection

- Uses MediaPipe Holistic for better multi-person detection
- Detects pose landmarks for each person in the frame

### 2. Person Tracking

- Associates new detections with existing persons based on landmark proximity
- Creates new person trackers for new detections
- Removes stale person trackers that haven't been seen recently

### 3. Activity Classification

- Each person has their own activity detection logic
- Tracks movement history and pose patterns individually
- Classifies activities based on pose and movement for each person

### 4. Visual Display

- Draws pose landmarks with person-specific colors
- Shows individual activity status for each person
- Displays summary information about total and active persons

## Technical Details

### Person Association Algorithm

- Uses nose landmark position as the primary metric for person association
- Calculates Euclidean distance between current and previous detections
- Associates new detections with the closest existing person if within threshold

### Color Assignment

- Each person gets a unique color from a predefined palette
- Colors cycle through 8 different options
- Colors are used for both landmarks and text display

### Performance Considerations

- Limits maximum number of persons to maintain performance
- Automatically removes stale trackers to prevent memory buildup
- Uses efficient distance calculations for person association

## Comparison with Single-Person Version

| Feature               | Single-Person | Multi-Person      |
| --------------------- | ------------- | ----------------- |
| Max Persons           | 1             | 8                 |
| Person Tracking       | No            | Yes               |
| Individual Activities | No            | Yes               |
| Color Coding          | Single color  | Unique per person |
| Person Management     | Manual        | Automatic         |
| Performance           | Higher        | Slightly lower    |

## Troubleshooting

### Common Issues

1. **No persons detected**

   - Check camera connection
   - Ensure good lighting
   - Verify camera permissions

2. **Persons not being tracked correctly**

   - Adjust `person_association_threshold` in config
   - Ensure persons are visible and well-lit
   - Check if `max_persons` limit is reached

3. **Performance issues**
   - Reduce `max_persons` in config
   - Lower camera resolution
   - Close other applications

### Performance Tips

- Use good lighting for better detection
- Ensure persons are clearly visible in the frame
- Adjust thresholds based on your specific use case
- Monitor FPS and adjust settings if needed

## Future Enhancements

Potential improvements for future versions:

- Face recognition for persistent person identification
- Improved person association algorithms
- Activity history and analytics
- Export functionality for activity data
- Integration with external systems
