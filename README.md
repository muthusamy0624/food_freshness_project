# Food Freshness Classification API

A production-ready FastAPI inference service for classifying food freshness using deep learning.

## Features

- 🍎 **Food Classification**: Classifies food as fresh, nearly expiry, or spoiled
- 🤖 **Deep Learning**: Powered by TensorFlow/Keras ResNet50 model
- 📱 **Mobile Ready**: CORS enabled for mobile app integration
- 📊 **Confidence Scoring**: Provides prediction confidence and probabilities
- 🛡️ **Error Handling**: Comprehensive error handling and validation
- 📝 **API Documentation**: Auto-generated Swagger/OpenAPI docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and status |
| GET | `/health` | Health check endpoint |
| POST | `/predict` | Predict food freshness from image |
| GET | `/docs` | Interactive API documentation |

## Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone or download this project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**:
   - Make sure `model.keras` is in the project directory
   - The model should be trained for food classification

4. **Start the server**:
   ```bash
   python start_server.py
   ```
   
   Or alternatively:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access the API**:
   - API Base URL: `http://localhost:8000`
   - Documentation: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/health`

## Usage Examples

### Python Client Example

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Image file to upload
with open("food_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Probabilities: {result['probabilities']}")
else:
    print(f"Error: {response.json()}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@food_image.jpg"
```

### Mobile App Integration

The API is configured with CORS to work with mobile apps:

```javascript
// Flutter/Dart example
import 'dart:io';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> predictFood(File imageFile) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('http://your-server-url:8000/predict'),
  );
  
  request.files.add(
    await http.MultipartFile.fromPath('file', imageFile.path),
  );
  
  var response = await request.send();
  var responseData = await response.stream.bytesToString();
  
  return json.decode(responseData);
}
```

## API Response Format

```json
{
  "predicted_class": "fresh",
  "confidence": 0.892,
  "probabilities": {
    "fresh": 0.892,
    "nearly_expiry": 0.087,
    "spoiled": 0.021
  },
  "reasoning": "High confidence (0.892 >= 0.600)",
  "threshold_used": 0.6,
  "model_info": {
    "classes": ["fresh", "nearly_expiry", "spoiled"],
    "input_shape": [224, 224, 3],
    "preprocessing": "ResNet50 standard"
  },
  "filename": "apple.jpg",
  "content_type": "image/jpeg",
  "file_size": 245678
}
```

## Model Information

- **Architecture**: ResNet50-based CNN
- **Input Size**: 224x224 RGB images
- **Classes**: fresh, nearly_expiry, spoiled
- **Confidence Threshold**: 0.6 (configurable)
- **Preprocessing**: Standard ResNet50 preprocessing

## Error Handling

The API provides comprehensive error handling:

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid file type, etc.) |
| 413 | Payload Too Large (file > 10MB) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (model not loaded) |

## Configuration

### Environment Variables

- `TF_CPP_MIN_LOG_LEVEL`: Set to '2' to reduce TensorFlow logging
- Model path can be changed in `load_model_on_startup()` function

### Customization

You can customize:
- Confidence threshold (`confidence_threshold` variable)
- Class names (`class_names` list)
- Maximum file size (in `/predict` endpoint)
- CORS settings in middleware configuration

## Deployment

### Production Deployment

1. **Use production app**:
   ```bash
   uvicorn app_production:app --host 0.0.0.0 --port 8000
   ```

2. **Environment setup**:
   - Set appropriate environment variables
   - Configure reverse proxy (nginx/Apache)
   - Set up SSL certificates

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app_production:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Render Deployment

The `app_production.py` file is optimized for Render deployment:
- Model loads on startup
- Proper error handling
- Production-ready logging

## Troubleshooting

### Common Issues

1. **Model loading fails**:
   - Ensure `model.keras` exists in the directory
   - Check TensorFlow version compatibility
   - Verify model file integrity

2. **Import errors**:
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **PIL image processing errors**:
   - The code handles both old and new PIL versions
   - Ensure Pillow is properly installed

4. **Memory issues**:
   - TensorFlow may require significant RAM
   - Consider using smaller batch sizes for large models

### Logging

The application includes comprehensive logging:
- Model loading status
- Prediction errors
- Request information
- Debug information for development

## Development

### Running in Development Mode

```bash
python start_server.py
```

This enables:
- Auto-reload on code changes
- Detailed logging
- Debug information

### Testing the API

1. **Health check**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Test prediction**:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -F "file=@test_image.jpg"
   ```

## License

This project is part of the Food Freshness Classification system.

## Support

For issues and support:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check server logs for detailed error information
