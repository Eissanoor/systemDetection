# System Detection API Documentation

## Overview
This API provides unified detection capabilities for both glove and helmet detection using YOLO models. It supports both image and video processing with a single endpoint for each.

## Endpoints

### 1. Image Detection
**POST** `/detect`

Detects gloves and/or helmets in uploaded images.

#### Parameters
- `file` (UploadFile): Image file to process (required)
- `model_type` (Query string): Detection model to use (optional, default: "both")
  - `"glove"`: Only glove detection
  - `"helmet"`: Only helmet detection  
  - `"both"`: Both glove and helmet detection

#### Example Request
```bash
curl -X POST "http://localhost:8000/detect?model_type=both" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

#### Response
```json
{
  "model_type": "both",
  "detection_summary": {
    "total_detections": 3,
    "glove_detections": 2,
    "helmet_detections": 1
  },
  "glove_detections": [
    {
      "box": [100, 150, 200, 250],
      "score": 0.95,
      "class": 0,
      "class_name": "glove",
      "model_type": "glove"
    }
  ],
  "helmet_detections": [
    {
      "box": [300, 100, 400, 200],
      "score": 0.87,
      "class": 1,
      "class_name": "Safety helmet",
      "model_type": "helmet"
    }
  ],
  "combined_detections": [...],
  "image_saved": "output/images/detected_20250101_120000.jpg",
  "prediction_saved": "output/predictions/prediction_20250101_120000.json",
  "timestamp": "20250101_120000"
}
```

### 2. Video Detection
**POST** `/detect-video`

Detects gloves and/or helmets in uploaded videos frame by frame.

#### Parameters
- `file` (UploadFile): Video file to process (required)
- `model_type` (Query string): Detection model to use (optional, default: "both")
  - `"glove"`: Only glove detection
  - `"helmet"`: Only helmet detection
  - `"both"`: Both glove and helmet detection

#### Example Request
```bash
curl -X POST "http://localhost:8000/detect-video?model_type=both" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@video.mp4"
```

#### Response
```json
{
  "message": "Video processing completed successfully",
  "model_type": "both",
  "video_saved": "output/videos/detected_20250101_120000.mp4",
  "prediction_saved": "output/predictions/video_prediction_20250101_120000.json",
  "timestamp": "20250101_120000",
  "video_properties": {
    "fps": 30,
    "width": 1920,
    "height": 1080,
    "total_frames": 900,
    "duration_seconds": 30.0
  },
  "detection_summary": {
    "total_detections": 150,
    "glove_detections": 100,
    "helmet_detections": 50,
    "glove_count": 80,
    "no_glove_count": 20,
    "safety_helmet_count": 45,
    "no_helmet_count": 5,
    "frames_with_detections": 300
  },
  "total_frames_processed": 900
}
```

## Model Classes

### Glove Model Classes
- `glove`: Person wearing gloves
- `no_glove`: Person not wearing gloves

### Helmet Model Classes
- `Safety helmet`: Person wearing safety helmet
- `No helmet`: Person not wearing helmet

## Color Coding

### Image Detection
- **Green boxes**: Glove detected
- **Red boxes**: No glove detected
- **Blue boxes**: Safety helmet detected
- **Orange boxes**: No helmet detected

### Video Detection
- Same color coding as images, with labels showing `[model_type] class_name: confidence`

## Output Files

### Images
- Annotated images saved to `output/images/detected_TIMESTAMP.jpg`
- Prediction data saved to `output/predictions/prediction_TIMESTAMP.json`

### Videos
- Annotated videos saved to `output/videos/detected_TIMESTAMP.mp4`
- Prediction data saved to `output/predictions/video_prediction_TIMESTAMP.json`

## Running the Server

```bash
python app.py
```

The server will start on `http://localhost:8000`

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
