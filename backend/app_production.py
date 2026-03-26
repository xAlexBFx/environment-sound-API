"""
YAMNet Sound Classification API
Production-ready Flask API for environmental sound classification
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import base64
import numpy as np
import librosa
import tensorflow_hub as hub
import os
import logging
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
allowed_origins = os.getenv('ALLOWED_ORIGINS', '*').split(',')
if '*' in allowed_origins:
    CORS(app)
else:
    CORS(app, origins=allowed_origins)

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[f"{os.getenv('RATE_LIMIT_PER_MINUTE', '60')} per minute"]
)

# Global model variables
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
    global yamnet_model, yamnet_class_names
    try:
        logger.info("Loading YAMNet from TensorFlow Hub...")
        cache_dir = os.getenv('MODEL_CACHE_DIR')
        if cache_dir:
            os.environ['TFHUB_CACHE_DIR'] = cache_dir
        
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        logger.info("YAMNet loaded successfully!")
        
        yamnet_class_names = load_yamnet_classes()
        logger.info(f"Loaded {len(yamnet_class_names)} YAMNet class names")
    except Exception as e:
        logger.error(f"Error loading YAMNet: {e}")
        raise


def preprocess_audio(audio_data, sample_rate=22050):
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


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'yamnet_loaded': yamnet_model is not None,
        'yamnet_classes': len(yamnet_class_names) if yamnet_class_names else 0,
        'version': '1.0.0'
    })


@app.route('/info', methods=['GET'])
def info():
    """API information endpoint"""
    return jsonify({
        'name': 'YAMNet Sound Classification API',
        'version': '1.0.0',
        'model': 'YAMNet',
        'classes': len(yamnet_class_names) if yamnet_class_names else 521,
        'endpoints': {
            '/health': 'GET - Health check',
            '/info': 'GET - API information',
            '/classify': 'POST - Classify audio (returns top prediction)',
            '/classify/raw': 'POST - Classify audio (returns top 5 predictions)',
            '/embeddings': 'POST - Extract YAMNet embeddings'
        }
    })


@app.route('/classify', methods=['POST'])
@limiter.limit("30 per minute")
def classify():
    """Main classification endpoint - returns top YAMNet AudioSet class"""
    if yamnet_model is None:
        return jsonify({'error': 'YAMNet not loaded'}), 500
    
    data = request.get_json()
    if not data or 'audio' not in data:
        return jsonify({'error': 'No audio data'}), 400
    
    audio = preprocess_audio(data['audio'])
    if audio is None:
        return jsonify({'error': 'Preprocessing failed'}), 400
    
    # Run YAMNet inference
    scores, embeddings, _ = yamnet_model(audio)
    avg_scores = scores.numpy().mean(axis=0)
    
    # Get top prediction from YAMNet's 521 classes
    predicted_idx = np.argmax(avg_scores)
    confidence = float(avg_scores[predicted_idx])
    predicted_class = yamnet_class_names[predicted_idx] if yamnet_class_names and predicted_idx < len(yamnet_class_names) else "unknown"
    
    # Get top 5 YAMNet predictions for display
    top_5_indices = avg_scores.argsort()[-5:][::-1]
    top_predictions = {
        yamnet_class_names[i] if yamnet_class_names and i < len(yamnet_class_names) else f"class_{i}": float(avg_scores[i])
        for i in top_5_indices
    }
    
    return jsonify({
        'className': predicted_class if confidence >= 0.6 else 'uncertain',
        'confidence': confidence,
        'allProbabilities': top_predictions,
        'model': 'yamnet',
        'classIndex': int(predicted_idx),
        'totalClasses': len(yamnet_class_names) if yamnet_class_names else 521
    })


@app.route('/classify/raw', methods=['POST'])
@limiter.limit("30 per minute")
def classify_raw():
    """Raw YAMNet classification (521 classes)"""
    if yamnet_model is None:
        return jsonify({'error': 'YAMNet not loaded'}), 500
    
    data = request.get_json()
    if not data or 'audio' not in data:
        return jsonify({'error': 'No audio data'}), 400
    
    audio = preprocess_audio(data['audio'])
    if audio is None:
        return jsonify({'error': 'Preprocessing failed'}), 400
    
    scores, embeddings, _ = yamnet_model(audio)
    avg_scores = scores.numpy().mean(axis=0)
    
    # Get top 5 predictions with class names
    top_5_indices = avg_scores.argsort()[-5:][::-1]
    top_5 = [
        {
            'className': yamnet_class_names[i] if yamnet_class_names and i < len(yamnet_class_names) else f"class_{i}",
            'classIndex': int(i),
            'confidence': float(avg_scores[i])
        }
        for i in top_5_indices
    ]
    
    return jsonify({
        'topPredictions': top_5,
        'model': 'yamnet',
        'totalClasses': len(yamnet_class_names) if yamnet_class_names else 521
    })


@app.route('/embeddings', methods=['POST'])
@limiter.limit("30 per minute")
def embeddings():
    """Extract YAMNet embeddings for transfer learning"""
    if yamnet_model is None:
        return jsonify({'error': 'YAMNet not loaded'}), 500
    
    data = request.get_json()
    if not data or 'audio' not in data:
        return jsonify({'error': 'No audio data'}), 400
    
    audio = preprocess_audio(data['audio'])
    if audio is None:
        return jsonify({'error': 'Preprocessing failed'}), 400
    
    _, emb, _ = yamnet_model(audio)
    avg_embedding = emb.numpy().mean(axis=0).tolist()
    
    return jsonify({
        'embedding': avg_embedding,
        'dimensions': len(avg_embedding)
    })


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded', 'retry_after': e.description}), 429


if __name__ == '__main__':
    load_yamnet()
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    app.run(host=host, port=port, debug=debug)
