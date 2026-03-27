"""
YAMNet Sound Classification API - Modal Deployment
Serverless API for environmental sound classification using Google's YAMNet model
"""

import os
import base64
import numpy as np
import librosa
import tensorflow_hub as hub
import logging
from typing import Dict, List, Optional, Tuple
import modal

# Configure Modal app
app = modal.App("environment-sound-api")

# Define Modal image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "ffmpeg", "libsndfile1", "libgomp1"
).pip_install(
    "flask==2.3.3",
    "flask-cors==4.0.0",
    "tensorflow==2.15.0", 
    "tensorflow-hub==0.15.0",
    "librosa==0.10.1",
    "numpy==1.24.3",
    "scipy==1.11.4",
    "soundfile==0.12.1",
    "audioread==3.0.1",
    "psutil==5.9.6",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Volume for model caching
model_volume = modal.Volume.from_name("yamnet-model-cache", create_if_missing=True)

# Global model variables (will be loaded in Modal container)
yamnet_model = None
yamnet_class_names = None


def load_yamnet_classes():
    """Load YAMNet's 521 class names from CSV"""
    import csv
    import io
    
    csv_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    
    try:
        import urllib.request
        response = urllib.request.urlopen(csv_url)
        lines = response.read().decode('utf-8').strip().split('\n')
        
        classes = []
        reader = csv.reader(io.StringIO('\n'.join(lines)))
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 3:
                classes.append(row[2])  # Display name is 3rd column
        
        return classes
    except Exception as e:
        logger.warning(f"Could not load YAMNet classes from URL: {e}")
        return [f"class_{i}" for i in range(521)]


def load_yamnet():
    """Load YAMNet model with caching"""
    global yamnet_model, yamnet_class_names
    
    try:
        logger.info("Loading YAMNet from TensorFlow Hub...")
        
        # Configure TensorFlow for GPU
        import tensorflow as tf
        
        # Check for GPU availability
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU detected: {len(gpus)} device(s)")
            try:
                # Enable memory growth to avoid allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("GPU memory growth enabled")
            except RuntimeError as e:
                logger.warning(f"GPU configuration error: {e}")
        else:
            logger.info("No GPU detected, using CPU")
        
        # Set cache directory to persistent volume
        cache_dir = "/models"
        os.environ['TFHUB_CACHE_DIR'] = cache_dir
        
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        logger.info("YAMNet loaded successfully!")
        
        yamnet_class_names = load_yamnet_classes()
        logger.info(f"Loaded {len(yamnet_class_names)} YAMNet class names")
        
    except Exception as e:
        logger.error(f"Error loading YAMNet: {e}")
        raise


def preprocess_audio(audio_data: str, sample_rate: int = 22050) -> Optional[np.ndarray]:
    """Preprocess audio for YAMNet (expects 16kHz)"""
    try:
        # Decode base64
        try:
            audio_bytes = base64.b64decode(audio_data)
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            return None
        
        # Check if this is raw file data (marker 888.888 as first float)
        if len(audio_bytes) > 4:
            try:
                first_float = np.frombuffer(audio_bytes[:4], dtype=np.float32)[0]
                
                if abs(first_float - 888.888) < 0.001:
                    # This is file data - rest are byte values as floats
                    float_array = np.frombuffer(audio_bytes[4:], dtype=np.float32)
                    file_bytes = bytes([int(min(255, max(0, f))) for f in float_array])
                    
                    # Detect file format from header
                    ext = '.m4a'
                    if file_bytes[:4] == b'RIFF':
                        ext = '.wav'
                    elif b'ftyp' in file_bytes[:100]:
                        ext = '.m4a'
                    
                    # Save to temp file and load with librosa
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    
                    try:
                        audio_array, sr = librosa.load(tmp_path, sr=16000, mono=True)
                        os.unlink(tmp_path)
                        return audio_array
                    except Exception as e:
                        logger.error(f"Error loading audio file: {e}")
                        os.unlink(tmp_path)
                        return None
            except Exception as e:
                logger.debug(f"Marker check failed: {e}")
        
        # Standard path: Float32Array audio samples from web
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Resample to 16kHz for YAMNet
            if sample_rate != 16000 and len(audio_array) > 0:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
            # Normalize to [-1, 1]
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
            
            return audio_array
        except Exception as e:
            logger.error(f"Standard audio processing error: {e}")
            return None
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return None


@app.function(
    image=image,
    volumes={"/models": model_volume},
    timeout=300,
    scaledown_window=300,
    gpu="T4",
    memory=8192
)
@modal.concurrent(max_inputs=10)
def classify_audio(audio_data: str) -> Dict:
    """Classify audio using YAMNet model"""
    global yamnet_model, yamnet_class_names
    
    # Load model if not already loaded
    if yamnet_model is None:
        load_yamnet()
    
    # Preprocess audio
    audio = preprocess_audio(audio_data)
    if audio is None:
        return {"error": "Preprocessing failed"}
    
    try:
        # Run YAMNet inference
        scores, embeddings, _ = yamnet_model(audio)
        avg_scores = scores.numpy().mean(axis=0)
        
        # Get top prediction
        predicted_idx = np.argmax(avg_scores)
        confidence = float(avg_scores[predicted_idx])
        predicted_class = yamnet_class_names[predicted_idx] if yamnet_class_names and predicted_idx < len(yamnet_class_names) else "unknown"
        
        # Get top 5 predictions
        top_5_indices = avg_scores.argsort()[-5:][::-1]
        top_predictions = {
            yamnet_class_names[i] if yamnet_class_names and i < len(yamnet_class_names) else f"class_{i}": float(avg_scores[i])
            for i in top_5_indices
        }
        
        return {
            'className': predicted_class if confidence >= 0.6 else 'uncertain',
            'confidence': confidence,
            'allProbabilities': top_predictions,
            'model': 'yamnet',
            'classIndex': int(predicted_idx),
            'totalClasses': len(yamnet_class_names) if yamnet_class_names else 521
        }
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {"error": f"Classification failed: {str(e)}"}




@app.function(
    image=image,
    timeout=30
)
def health_check() -> Dict:
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'environment-sound-api',
        'version': '1.0.0'
    }


# FastAPI web server for HTTP endpoints
@app.function(
    image=image,
    timeout=30,
    gpu="T4",
    memory=4096
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def fastapi_app():
    # Import FastAPI inside the function to ensure it's available in the Modal container
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    
    web_app = FastAPI(title="Environment Sound API")

    # Add CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class AudioRequest(BaseModel):
        audio: str

    @web_app.get("/health")
    async def health():
        """Health check endpoint"""
        result = health_check.remote()
        return result

    @web_app.get("/info")
    async def info():
        """API information endpoint"""
        return {
            'name': 'Environment Sound API',
            'version': '1.0.0',
            'model': 'YAMNet',
            'classes': 521,
            'endpoints': {
                '/health': 'GET - Health check',
                '/info': 'GET - API information', 
                '/classify': 'POST - Classify audio (returns top prediction)',
                '/classify/raw': 'POST - Classify audio (returns top 5 predictions)'
            }
        }

    @web_app.post("/classify")
    async def classify(request: AudioRequest):
        """Main classification endpoint - returns top YAMNet AudioSet class"""
        if not request.audio:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        result = await classify_audio.remote.aio(request.audio)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result

    @web_app.post("/classify/raw")
    async def classify_raw(request: AudioRequest):
        """Raw YAMNet classification (521 classes)"""
        if not request.audio:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        result = await classify_audio.remote.aio(request.audio)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Format for raw endpoint
        top_5 = [
            {
                'className': class_name,
                'confidence': confidence
            }
            for class_name, confidence in list(result['allProbabilities'].items())[:5]
        ]
        
        return {
            'topPredictions': top_5,
            'model': 'yamnet',
            'totalClasses': result['totalClasses']
        }

    
    return web_app


if __name__ == "__main__":
    # For local testing - create a simple Flask app
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy', 'service': 'environment-sound-api'})
    
    @app.route('/info')
    def info():
        return jsonify({
            'name': 'Environment Sound API',
            'version': '1.0.0',
            'message': 'Use Modal deployment for full functionality'
        })
    
    app.run(host="0.0.0.0", port=8000)
