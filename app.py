# server.py (FastAPI)
# Fix for PyTorch 2.6 weights_only issue - monkey patch torch.load
import torch

def _torch_load_fix(f, *args, **kwargs):
    """Monkey patch to add weights_only=False for PyTorch 2.6 compatibility"""
    kwargs['weights_only'] = False
    return torch._orig_load(f, *args, **kwargs)

# Apply the monkey patch
torch._orig_load = torch.load
torch.load = _torch_load_fix

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse
import uvicorn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import json
import cv2
from datetime import datetime
from ultralytics import YOLO
from typing import Optional
import threading
import time

app = FastAPI()

# Load both models
glove_model = YOLO('./gloveTrained_best.pt')
helmet_model = YOLO('./helmetTrained_best.pt')

# Class names from your data.yaml files
glove_class_names = ['glove', 'no_glove']
helmet_class_names = ['No helmet', 'Safety helmet']

# Create output directories
os.makedirs('output/images', exist_ok=True)
os.makedirs('output/predictions', exist_ok=True)
os.makedirs('output/videos', exist_ok=True)

def run_detection(image, model_type: str = "both", device: str = "auto"):
    """
    Run detection on image using specified model(s)
    
    Args:
        image: PIL Image or numpy array
        model_type: "glove", "helmet", or "both"
        device: "auto", "cpu", or "cuda"
    
    Returns:
        dict: Combined detection results
    """
    if device == "auto":
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    results = {
        'glove_detections': [],
        'helmet_detections': [],
        'combined_detections': []
    }
    
    # Run glove detection
    if model_type in ["glove", "both"]:
        glove_res = glove_model.predict(image, imgsz=640, conf=0.25, device=device, verbose=False)
        for r in glove_res:
            for box in r.boxes.data.tolist():
                x1, y1, x2, y2, score, cls = box
                cls_int = int(cls)
                detection = {
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(score),
                    'class': cls_int,
                    'class_name': glove_class_names[cls_int] if cls_int < len(glove_class_names) else f'unknown_{cls_int}',
                    'model_type': 'glove'
                }
                results['glove_detections'].append(detection)
                results['combined_detections'].append(detection)
    
    # Run helmet detection
    if model_type in ["helmet", "both"]:
        helmet_res = helmet_model.predict(image, imgsz=640, conf=0.25, device=device, verbose=False)
        for r in helmet_res:
            for box in r.boxes.data.tolist():
                x1, y1, x2, y2, score, cls = box
                cls_int = int(cls)
                detection = {
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(score),
                    'class': cls_int,
                    'class_name': helmet_class_names[cls_int] if cls_int < len(helmet_class_names) else f'unknown_{cls_int}',
                    'model_type': 'helmet'
                }
                results['helmet_detections'].append(detection)
                results['combined_detections'].append(detection)
    
    return results

def draw_detections_on_image(image, detections):
    """
    Draw bounding boxes on image for all detections
    """
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        score = detection['score']
        cls_name = detection['class_name']
        model_type = detection['model_type']
        
        # Set color based on model type and class
        if model_type == 'glove':
            color = 'green' if cls_name == 'glove' else 'red'
        else:  # helmet
            color = 'green' if cls_name == 'Safety helmet' else 'red'
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label with model type, class name and confidence score
        label = f"[{model_type}] {cls_name}: {score:.2f}"
        draw.text((x1, y1-25), label, fill=color, font=font)
    
    return img_with_boxes

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    model_type: str = Query("both", description="Detection model: 'glove', 'helmet', or 'both'")
):
    """
    Unified endpoint for image detection using glove and/or helmet models
    
    Args:
        file: Image file to process
        model_type: "glove", "helmet", or "both" (default: "both")
    
    Returns:
        Combined detection results from specified model(s)
    """
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert('RGB')
    
    # Generate timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run detection using the unified function
    detection_results = run_detection(img, model_type=model_type)
    
    # Draw bounding boxes on the image
    img_with_boxes = draw_detections_on_image(img, detection_results['combined_detections'])
    
    # Save the annotated image
    img_filename = f"detected_{timestamp}.jpg"
    img_path = os.path.join('output/images', img_filename)
    img_with_boxes.save(img_path, 'JPEG')
    
    # Calculate statistics
    total_detections = len(detection_results['combined_detections'])
    glove_count = len(detection_results['glove_detections'])
    helmet_count = len(detection_results['helmet_detections'])
    
    # Save prediction data as JSON
    prediction_data = {
        'timestamp': timestamp,
        'original_filename': file.filename,
        'model_type': model_type,
        'detection_summary': {
            'total_detections': total_detections,
            'glove_detections': glove_count,
            'helmet_detections': helmet_count
        },
        'glove_detections': detection_results['glove_detections'],
        'helmet_detections': detection_results['helmet_detections'],
        'combined_detections': detection_results['combined_detections'],
        'image_path': img_path
    }
    
    prediction_filename = f"prediction_{timestamp}.json"
    prediction_path = os.path.join('output/predictions', prediction_filename)
    
    with open(prediction_path, 'w') as f:
        json.dump(prediction_data, f, indent=2)
    
    return {
        'model_type': model_type,
        'detection_summary': prediction_data['detection_summary'],
        'glove_detections': detection_results['glove_detections'],
        'helmet_detections': detection_results['helmet_detections'],
        'combined_detections': detection_results['combined_detections'],
        'image_saved': img_path,
        'prediction_saved': prediction_path,
        'timestamp': timestamp
    }

@app.post("/detect-video")
async def detect_video(
    file: UploadFile = File(...),
    model_type: str = Query("both", description="Detection model: 'glove', 'helmet', or 'both'")
):
    """
    Unified endpoint for video detection using glove and/or helmet models
    
    Args:
        file: Video file to process
        model_type: "glove", "helmet", or "both" (default: "both")
    
    Returns:
        Combined detection results from specified model(s) for the entire video
    """
    data = await file.read()
    
    # Generate timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save uploaded video temporarily
    temp_video_path = f"temp_video_{timestamp}.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(data)
    
    try:
        # Open video
        cap = cv2.VideoCapture(temp_video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer for output
        output_video_path = os.path.join('output/videos', f"detected_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Process video frame by frame
        frame_detections = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection using unified function
            detection_results = run_detection(frame_rgb, model_type=model_type)
            
            # Process detections for this frame
            frame_detection_data = {
                'glove_detections': detection_results['glove_detections'],
                'helmet_detections': detection_results['helmet_detections'],
                'combined_detections': detection_results['combined_detections']
            }
            
            # Draw bounding boxes on frame
            for detection in detection_results['combined_detections']:
                x1, y1, x2, y2 = detection['box']
                score = detection['score']
                cls_name = detection['class_name']
                model_type_det = detection['model_type']
                
                # Set color based on model type and class
                if model_type_det == 'glove':
                    color = (0, 255, 0) if cls_name == 'glove' else (0, 0, 255)  # Green for glove, Red for no_glove
                else:  # helmet
                    color = (0, 255, 0) if cls_name == 'Safety helmet' else (0, 0, 255)  # Green for helmet, Red for no helmet
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label
                label = f"[{model_type_det}] {cls_name}: {score:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Store frame detection data
            frame_detections.append({
                'frame_number': frame_count,
                'timestamp_seconds': frame_count / fps,
                'glove_detections': detection_results['glove_detections'],
                'helmet_detections': detection_results['helmet_detections'],
                'combined_detections': detection_results['combined_detections'],
                'total_detections': len(detection_results['combined_detections'])
            })
            
            # Write frame to output video
            out.write(frame)
            frame_count += 1
        
        # Release everything
        cap.release()
        out.release()
        
        # Calculate video statistics
        total_detections = sum(frame['total_detections'] for frame in frame_detections)
        glove_detections = sum(1 for frame in frame_detections for det in frame['glove_detections'])
        helmet_detections = sum(1 for frame in frame_detections for det in frame['helmet_detections'])
        
        # Count specific classes
        glove_count = sum(1 for frame in frame_detections for det in frame['glove_detections'] if det['class_name'] == 'glove')
        no_glove_count = sum(1 for frame in frame_detections for det in frame['glove_detections'] if det['class_name'] == 'no_glove')
        safety_helmet_count = sum(1 for frame in frame_detections for det in frame['helmet_detections'] if det['class_name'] == 'Safety helmet')
        no_helmet_count = sum(1 for frame in frame_detections for det in frame['helmet_detections'] if det['class_name'] == 'No helmet')
        
        # Save video prediction data as JSON
        video_prediction_data = {
            'timestamp': timestamp,
            'original_filename': file.filename,
            'model_type': model_type,
            'video_properties': {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames,
                'duration_seconds': total_frames / fps
            },
            'detection_summary': {
                'total_detections': total_detections,
                'glove_detections': glove_detections,
                'helmet_detections': helmet_detections,
                'glove_count': glove_count,
                'no_glove_count': no_glove_count,
                'safety_helmet_count': safety_helmet_count,
                'no_helmet_count': no_helmet_count,
                'frames_with_detections': len([f for f in frame_detections if f['total_detections'] > 0])
            },
            'frame_detections': frame_detections,
            'output_video_path': output_video_path
        }
        
        prediction_filename = f"video_prediction_{timestamp}.json"
        prediction_path = os.path.join('output/predictions', prediction_filename)
        
        with open(prediction_path, 'w') as f:
            json.dump(video_prediction_data, f, indent=2)
        
        return {
            'message': 'Video processing completed successfully',
            'model_type': model_type,
            'video_saved': output_video_path,
            'prediction_saved': prediction_path,
            'timestamp': timestamp,
            'video_properties': video_prediction_data['video_properties'],
            'detection_summary': video_prediction_data['detection_summary'],
            'total_frames_processed': frame_count
        }
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

# Global variable to control camera streaming
camera_active = False
camera_thread = None

def generate_frames():
    """Generate video frames with real-time detection"""
    global camera_active
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use default camera (index 0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        while camera_active:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Convert BGR to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection using both models
            detection_results = run_detection(frame_rgb, model_type="both")
            
            # Draw bounding boxes on frame
            for detection in detection_results['combined_detections']:
                x1, y1, x2, y2 = detection['box']
                score = detection['score']
                cls_name = detection['class_name']
                model_type_det = detection['model_type']
                
                # Set color based on model type and class
                if model_type_det == 'glove':
                    color = (0, 255, 0) if cls_name == 'glove' else (0, 0, 255)  # Green for glove, Red for no_glove
                else:  # helmet
                    color = (0, 255, 0) if cls_name == 'Safety helmet' else (0, 0, 255)  # Green for helmet, Red for no helmet
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label
                label = f"[{model_type_det}] {cls_name}: {score:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add detection summary to frame
            glove_count = len(detection_results['glove_detections'])
            helmet_count = len(detection_results['helmet_detections'])
            summary_text = f"Gloves: {glove_count} | Helmets: {helmet_count}"
            cv2.putText(frame, summary_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to control frame rate
            time.sleep(0.033)  # ~30 FPS
    
    finally:
        cap.release()

@app.get("/live-camera")
async def live_camera():
    """
    Live camera detection endpoint that streams real-time video with both glove and helmet detection
    
    Returns:
        StreamingResponse: MJPEG video stream with detection overlays
    """
    global camera_active, camera_thread
    
    # Start camera if not already active
    if not camera_active:
        camera_active = True
        camera_thread = threading.Thread(target=generate_frames)
        camera_thread.daemon = True
        camera_thread.start()
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/stop-camera")
async def stop_camera():
    """
    Stop the live camera detection
    
    Returns:
        dict: Status message
    """
    global camera_active
    
    camera_active = False
    return {"message": "Camera stopped successfully"}

@app.get("/camera-status")
async def camera_status():
    """
    Get the current status of the camera
    
    Returns:
        dict: Camera status information
    """
    return {
        "camera_active": camera_active,
        "message": "Camera is running" if camera_active else "Camera is stopped"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
