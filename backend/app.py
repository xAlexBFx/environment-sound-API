from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import librosa
import tensorflow_hub as hub

app = Flask(__name__)
CORS(app)

yamnet_model = None
yamnet_class_names = None

# Urban sound classes (10 classes for UrbanSound8K)
CLASS_NAMES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 
    'siren', 'street_music'
]

# Mapping from YAMNet classes to our urban classes (simplified)
YAMNET_TO_URBAN = {
    'air_conditioner': [384, 385],
    'car_horn': [390, 391],
    'children_playing': [420, 421],
    'dog_bark': [80, 81],
    'drilling': [347, 348],
    'engine_idling': [343, 344],
    'gun_shot': [399, 400],
    'jackhammer': [401, 402],
    'siren': [389, 390],
    'street_music': [100, 101]
}


def load_yamnet_classes():
    """Load YAMNet's 521 class names from CSV"""
    import csv
    import io
    
    # YAMNet class map CSV content
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
        print(f"Could not load YAMNet classes from URL: {e}")
        # Fallback: return default class count
        return [f"class_{i}" for i in range(521)]


def load_yamnet():
    global yamnet_model, yamnet_class_names
    try:
        print("Loading YAMNet from TensorFlow Hub...")
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("YAMNet loaded successfully!")
        
        # Load class names
        yamnet_class_names = load_yamnet_classes()
        print(f"Loaded {len(yamnet_class_names)} YAMNet class names")
    except Exception as e:
        print(f"Error loading YAMNet: {e}")
        raise


def preprocess_audio(audio_data, sample_rate=22050):
    """Preprocess audio for YAMNet (expects 16kHz)"""
    try:
        print(f"Received audio data length: {len(audio_data)}")
        
        # Decode base64
        try:
            audio_bytes = base64.b64decode(audio_data)
            print(f"Decoded bytes length: {len(audio_bytes)}")
        except Exception as e:
            print(f"Base64 decode error: {e}")
            return None
        
        # Check if this is raw file data (marker 888.888 as first float)
        if len(audio_bytes) > 4:
            try:
                first_float = np.frombuffer(audio_bytes[:4], dtype=np.float32)[0]
                print(f"First float value: {first_float}")
                
                if abs(first_float - 888.888) < 0.001:
                    print("Detected file data marker")
                    # This is file data - rest are byte values as floats
                    float_array = np.frombuffer(audio_bytes[4:], dtype=np.float32)
                    print(f"Float array length: {len(float_array)}")
                    
                    file_bytes = bytes([int(min(255, max(0, f))) for f in float_array])
                    print(f"Reconstructed file bytes: {len(file_bytes)}")
                    print(f"First 20 bytes (hex): {file_bytes[:20].hex()}")
                    
                    # Detect file format from header
                    ext = '.m4a'
                    if file_bytes[:4] == b'RIFF':
                        ext = '.wav'
                        print("Detected WAV format")
                    elif file_bytes[:4] == b'\x00\x00\x00\x20' or b'ftyp' in file_bytes[:100]:
                        ext = '.m4a'
                        print("Detected MP4/M4A format")
                    
                    # Save to temp file and load with librosa
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                        print(f"Saved to temp file: {tmp_path}")
                    
                    try:
                        # Load with librosa (supports MP4/AAC, WAV, etc.)
                        audio_array, sr = librosa.load(tmp_path, sr=16000, mono=True)
                        print(f"Loaded audio with librosa: {len(audio_array)} samples at {sr}Hz")
                        os.unlink(tmp_path)
                        return audio_array
                    except Exception as e:
                        print(f"Error loading audio file with librosa: {e}")
                        # Save failed file for debugging
                        debug_path = f"failed_audio{ext}"
                        with open(debug_path, 'wb') as f:
                            f.write(file_bytes)
                        print(f"Saved failed audio to {debug_path} for debugging")
                        os.unlink(tmp_path)
                        return None
            except Exception as e:
                print(f"Marker check failed: {e}")
        
        # Standard path: Float32Array audio samples from web
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            print(f"Audio array length: {len(audio_array)}")
            
            # Resample to 16kHz for YAMNet
            if sample_rate != 16000 and len(audio_array) > 0:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
            # Normalize to [-1, 1]
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
            
            return audio_array
        except Exception as e:
            print(f"Standard audio processing error: {e}")
            return None
    except Exception as e:
        print(f"Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return None


def map_yamnet_to_urban(yamnet_scores):
    """Map YAMNet AudioSet scores to urban sound classes"""
    urban_scores = np.zeros(len(CLASS_NAMES))
    
    for i, class_name in enumerate(CLASS_NAMES):
        indices = YAMNET_TO_URBAN.get(class_name, [])
        if indices:
            # Average the scores for mapped YAMNet classes
            valid_scores = [yamnet_scores[idx] for idx in indices if idx < len(yamnet_scores)]
            if valid_scores:
                urban_scores[i] = np.mean(valid_scores)
    
    # If all scores are low, use a fallback heuristic
    if urban_scores.sum() == 0:
        # Use some reasonable defaults based on common YAMNet classes
        urban_scores[0] = yamnet_scores[0] * 0.1  # Speech -> air_conditioner (weak)
        urban_scores[4] = yamnet_scores[347] * 0.8 if 347 < len(yamnet_scores) else 0  # drilling
        urban_scores[5] = yamnet_scores[343] * 0.8 if 343 < len(yamnet_scores) else 0  # engine_idling
        urban_scores[8] = yamnet_scores[389] * 0.9 if 389 < len(yamnet_scores) else 0  # siren
    
    # Normalize to sum to 1 (softmax-like)
    urban_scores = urban_scores / (urban_scores.sum() + 1e-8)
    
    return urban_scores


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'yamnet_loaded': yamnet_model is not None,
        'yamnet_classes': len(yamnet_class_names) if yamnet_class_names else 0
    })


@app.route('/classify', methods=['POST'])
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


if __name__ == '__main__':
    load_yamnet()
    app.run(host='0.0.0.0', port=5000, debug=True)
