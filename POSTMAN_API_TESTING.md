# API Testing Guide for YOLO Detection API

## Endpoint Details

- **URL**: `http://localhost:8000/detect`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

## Postman Setup

### Step 1: Create a new POST request
1. Open Postman
2. Create a new request
3. Set method to **POST**
4. Enter URL: `http://localhost:8000/detect`

### Step 2: Configure the body
1. Go to the **Body** tab
2. Select **form-data** (not raw or binary)
3. Add a new key:
   - Key: `file` (dropdown should be set to **File**)
   - Value: Select an image file from your computer
4. Click **Send**

### Visual Guide

```
┌────────────────────────────────────────┐
│ POST  http://localhost:8000/detect     │
├────────────────────────────────────────┤
│ Body (tab)                             │
│  ○ none  ○ form-data  ○ x-www-form... │
│                                        │
│  Key          Value  Description       │
│  ─────────────────────────────────────│
│  file    [📎 Choose Files]  File       │
└────────────────────────────────────────┘
```

## Expected Response

### Success Response (200 OK)
```json
{
  "detections": [
    {
      "box": [100, 150, 300, 400],
      "score": 0.95,
      "class": 0
    },
    {
      "box": [500, 200, 700, 500],
      "score": 0.87,
      "class": 1
    }
  ]
}
```

### Response Fields
- **detections**: Array of detected objects
- **box**: `[x1, y1, x2, y2]` - Bounding box coordinates
- **score**: Confidence score (0.0 to 1.0)
- **class**: Class ID of the detected object

## cURL Example

If you prefer using cURL from terminal:

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@path/to/your/image.jpg"
```

## Python Example

```python
import requests

url = "http://localhost:8000/detect"
files = {"file": open("path/to/your/image.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

## JavaScript/Fetch Example

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/detect', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Notes

- Supported image formats: JPG, PNG, JPEG
- The API returns detections with confidence > 0.25 (as configured)
- Model input size: 640x640 pixels
- If using GPU, it will automatically use device=0
