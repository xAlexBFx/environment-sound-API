---
title: YAMNet Sound Classifier
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 5000
---

# Environment Sound Classification API

[![GitHub](https://img.shields.io/badge/GitHub-xAlexBFx%2Fenvironment--sound--API-blue)](https://github.com/xAlexBFx/environment-sound-API)

A RESTful API for classifying environmental sounds using Google's YAMNet deep learning model. The API can identify 521 different audio classes including urban sounds like sirens, car horns, engine noise, and more.

## Overview

This project provides a Flask-based backend service that accepts audio data via HTTP requests and returns sound classification predictions using the pre-trained YAMNet model from TensorFlow Hub.

**Related Projects:**
- [Environment Sound Classifier (Frontend)](https://github.com/xAlexBFx/environment-sound-classifier) - Web app that uses this API

**Key Features:**
- **521 Audio Classes**: Leverages YAMNet's extensive AudioSet training
- **Multiple Audio Formats**: Supports raw audio samples, WAV, M4A/AAC via base64 encoding
- **REST API**: Simple HTTP endpoints for classification and health checks
- **Embeddings Extraction**: Get YAMNet feature embeddings for transfer learning
- **Docker Support**: Ready-to-use containerization with Docker Compose
- **Production Ready**: Gunicorn configuration with rate limiting and CORS

## Project Structure

```
environment-sound-API/
├── backend/
│   ├── app.py                 # Development Flask application
│   ├── app_production.py      # Production Flask application
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile             # Docker container configuration
│   ├── docker-compose.yml     # Docker Compose setup
│   ├── gunicorn.conf.py       # Gunicorn production server config
│   ├── render.yaml            # Render deployment blueprint
│   ├── DEPLOYMENT.md          # Detailed deployment guide
│   └── .env.example           # Environment variables template
└── README.md                  # This file
```

## Quick Start

### Prerequisites

- Python 3.9+ or Docker
- FFmpeg (for audio format support)
- 2GB+ RAM (TensorFlow model loading)

### Option 1: Docker (Recommended)

```bash
cd backend

# Copy environment file
cp .env.example .env

# Build and run
docker-compose up -d

# Check health
curl http://localhost:5000/health
```

### Option 2: Local Development

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/classify` | POST | Classify audio (returns top prediction) |
| `/classify/raw` | POST | Classify audio (returns top 5 predictions) |
| `/embeddings` | POST | Extract 1024-dimensional YAMNet embeddings |

### Example: Classify Audio

```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "base64_encoded_audio_data_here"
  }'
```

**Response:**
```json
{
  "className": "Siren",
  "confidence": 0.89,
  "allProbabilities": {
    "Siren": 0.89,
    "Police car": 0.05,
    "Ambulance": 0.03
  },
  "model": "yamnet",
  "classIndex": 389,
  "totalClasses": 521
}
```

## Supported Audio Formats

The API accepts base64-encoded audio in multiple formats:

1. **Raw Float32 Samples** - Standard web audio (recommended)
2. **WAV Files** - Auto-detected via RIFF header
3. **M4A/AAC Files** - Auto-detected and processed via librosa

Audio is automatically resampled to 16kHz (YAMNet requirement) and normalized.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 5000 | Server port |
| `HOST` | 0.0.0.0 | Server host |
| `DEBUG` | false | Enable debug mode |
| `LOG_LEVEL` | INFO | Logging level |
| `ALLOWED_ORIGINS` | * | CORS allowed origins |
| `RATE_LIMIT_PER_MINUTE` | 60 | API rate limit |
| `MODEL_CACHE_DIR` | - | Model cache directory |

See `.env.example` for all options.

## Deployment

The API supports multiple deployment platforms:

- **Docker** - Use `docker-compose up`
- **Render** - Blueprint included (`render.yaml`)
- **Heroku** - Push to deploy
- **Google Cloud Run** - Container-based deployment
- **AWS Elastic Beanstalk** - Standard Python platform

See [`backend/DEPLOYMENT.md`](backend/DEPLOYMENT.md) for detailed instructions.

### Render (Recommended for Quick Deploy)

```bash
# Push to GitHub, then in Render Dashboard:
# New + → Blueprint → Connect repo → Auto-deploys
```

## Model Information

- **Model**: YAMNet (from TensorFlow Hub)
- **Classes**: 521 AudioSet classes
- **Input**: 16kHz mono audio
- **Output**: Class scores + 1024-dim embeddings
- **Size**: ~15MB (downloads on first use)

## Dependencies

```
flask>=2.3.3
tensorflow>=2.15.0
tensorflow-hub>=0.15.0
librosa>=0.10.1
numpy>=1.24.3
gunicorn>=21.2.0
```

See `backend/requirements.txt` for complete list.

## Troubleshooting

### FFmpeg Not Found
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Memory Issues
- Reduce Gunicorn workers in `gunicorn.conf.py`
- Ensure 2GB+ RAM available
- Set `MODEL_CACHE_DIR` to persistent storage

### Slow First Request
YAMNet downloads from TensorFlow Hub on first use (30-60s). Subsequent requests are fast.

## License

MIT License - See LICENSE file for details.

## Credits

- YAMNet model by Google Research
- AudioSet dataset by Google
- Flask framework by Pallets Projects
