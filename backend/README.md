# Sound Classification Backend

This backend provides audio classification using a Keras model for the environment sound classifier frontend.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your trained `.keras` model file in the `backend/models/` directory:
```
backend/models/sound_classifier.keras
```

3. Run the backend:
```bash
python app.py
```

The backend will start on `http://localhost:5000`

## API Endpoints

### Health Check
- **GET** `/health`
- Returns: Backend status, model loading status, and available classes

### Audio Classification
- **POST** `/classify`
- Body: 
```json
{
  "audio": "base64_encoded_audio_data"
}
```
- Returns: Classification result with confidence and all probabilities

## Model Requirements

The backend expects a Keras model trained on:
- Input shape: `(None, None, 13, 1)` (batch, time, MFCC features, channel)
- Output: 10 classes corresponding to:
  - air_conditioner
  - car_horn
  - children_playing
  - dog_bark
  - drilling
  - engine_idling
  - gun_shot
  - jackhammer
  - siren
  - street_music

## Audio Processing

- Sample rate: 22050 Hz
- Duration: 2 seconds
- Features: 13 MFCC coefficients
- Format: Base64 encoded audio data
