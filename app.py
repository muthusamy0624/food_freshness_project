#!/usr/bin/env python3
"""
Food Freshness Classification API
Production-ready FastAPI inference service for Render deployment
"""

import os
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from typing import Dict, Any

# Initialize FastAPI application
app = FastAPI(
    title="Food Freshness Classification API",
    description="Production-ready inference service for food freshness classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
class_names = ["fresh", "nearly_expiry", "spoiled"]
confidence_threshold = 0.6

def load_model_on_startup():
    """Load model on application startup"""
    global model
    try:
        model_path = "model.keras"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = load_model(model_path)
        print(f"✅ Model loaded successfully: {model_path}")
        print(f"✅ Model parameters: {model.count_params():,}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

def preprocess_image(img_data: bytes) -> np.ndarray:
    """
    Preprocess image consistently with training methodology
    
    Args:
        img_data: Raw image bytes
        
    Returns:
        Preprocessed numpy array ready for model inference
    """
    try:
        # Load image using PIL for better format handling
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to match training input size (224x224)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = image.img_to_array(img)
        
        # Apply ResNet50 preprocessing (same as training)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

def predict_with_confidence(processed_image: np.ndarray) -> Dict[str, Any]:
    """
    Perform model inference with confidence scoring
    
    Args:
        processed_image: Preprocessed image array
        
    Returns:
        Dictionary containing prediction, confidence, and probabilities
    """
    try:
        # Model inference
        predictions = model.predict(processed_image, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get per-class probabilities
        probabilities = {
            class_names[i]: float(predictions[0][i]) 
            for i in range(len(class_names))
        }
        
        # Apply confidence-based decision logic
        if confidence < confidence_threshold:
            predicted_class = "uncertain"
            reasoning = f"Low confidence ({confidence:.3f} < {confidence_threshold})"
        else:
            predicted_class = class_names[predicted_class_idx]
            reasoning = f"High confidence ({confidence:.3f} >= {confidence_threshold})"
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "reasoning": reasoning,
            "threshold_used": confidence_threshold,
            "model_info": {
                "classes": class_names,
                "input_shape": (224, 224, 3),
                "preprocessing": "ResNet50 standard"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "title": "Food Freshness Classification API",
        "description": "Production-ready inference service for food freshness classification",
        "version": "1.0.0",
        "model_path": "model.keras",
        "classes": class_names,
        "confidence_threshold": confidence_threshold,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "Food Freshness Classification API"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        global model

        # Lazy load model (only when needed)
        if model is None:
            load_model_on_startup()

        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image."
            )

        # Read image data
        img_data = await file.read()

        # Preprocess image
        processed_image = preprocess_image(img_data)

        # Perform prediction
        result = predict_with_confidence(processed_image)

        # Add metadata
        result.update({
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(img_data)
        })

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during prediction: {str(e)}"
        )
