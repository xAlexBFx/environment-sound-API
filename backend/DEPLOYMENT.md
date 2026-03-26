# YAMNet Sound Classification API - Deployment Guide

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and navigate to backend
cd backend

# Create environment file
cp .env.example .env
# Edit .env with your settings

# Build and run
docker-compose up -d
```

### Option 2: Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your settings

# Run with production server
gunicorn -c gunicorn.conf.py app_production:app

# Or run development server
python app_production.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 5000 | Server port |
| `HOST` | 0.0.0.0 | Server host |
| `DEBUG` | false | Enable debug mode |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `ALLOWED_ORIGINS` | * | CORS allowed origins (comma-separated) |
| `RATE_LIMIT_PER_MINUTE` | 60 | API rate limit per minute |
| `MODEL_CACHE_DIR` | - | Directory to cache YAMNet model |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | API information |
| `/classify` | POST | Classify audio (top prediction) |
| `/classify/raw` | POST | Classify audio (top 5 predictions) |
| `/embeddings` | POST | Extract YAMNet embeddings |

## Deployment Options

### 1. Heroku

```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create your-app-name

# Set environment variables
heroku config:set PORT=5000
heroku config:set DEBUG=false

# Deploy
git push heroku main
```

### 2. Railway

1. Connect GitHub repo to Railway
2. Add environment variables in Railway dashboard
3. Deploy automatically on push

### 3. AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.11 your-app-name

# Create environment
eb create your-env-name

# Deploy
eb deploy
```

### 4. Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/your-project/yamnet-api

# Deploy
gcloud run deploy yamnet-api \
  --image gcr.io/your-project/yamnet-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 6. Render (Recommended for Beginners)

**Option A: Deploy via Blueprint (Infrastructure as Code)**

1. Push your code to GitHub
2. In Render Dashboard, click "New +" → "Blueprint"
3. Connect your GitHub repo
4. Render will use `render.yaml` to configure the service automatically

**Option B: Manual Deploy**

1. In Render Dashboard, click "New +" → "Web Service"
2. Connect your GitHub repo
3. Select "Docker" as runtime
4. Configure:
   - **Name**: `yamnet-sound-classifier`
   - **Region**: Choose closest to your users
   - **Plan**: Standard ($7/month minimum for persistent disk)
5. Add environment variables:
   ```
   PORT=5000
   HOST=0.0.0.0
   DEBUG=false
   LOG_LEVEL=INFO
   ALLOWED_ORIGINS=*
   RATE_LIMIT_PER_MINUTE=60
   MODEL_CACHE_DIR=/app/models
   ```
6. Add Disk:
   - **Name**: `model-cache`
   - **Mount Path**: `/app/models`
   - **Size**: 2 GB
7. Set health check path: `/health`
8. Click "Create Web Service"

**Important Render Notes:**
- Free tier doesn't support persistent disks - model will re-download on every restart
- Use Standard tier ($7/month) with disk for production
- First deploy takes 5-10 minutes (model download + Docker build)
- Health check prevents traffic until model is loaded

## System Requirements

- **RAM**: 2GB minimum, 4GB recommended
- **CPU**: 2 cores minimum
- **Storage**: 1GB for model cache
- **OS**: Linux (Ubuntu 20.04+ recommended)

## Important Notes

1. **FFmpeg Required**: FFmpeg must be installed for audio processing
2. **First Start**: Model downloads on first request (30-60 seconds)
3. **Memory**: TensorFlow models use significant memory
4. **Preloading**: Use `--preload` with Gunicorn to share model memory
5. **Rate Limiting**: Default is 60 requests/minute per IP

## Troubleshooting

### FFmpeg not found
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Memory issues
- Reduce Gunicorn workers to 1 or 2
- Increase server RAM
- Use model caching (set MODEL_CACHE_DIR)

### Slow first request
- Model downloads from TensorFlow Hub on first use
- Pre-warm by calling `/health` after startup
- Use `docker-compose` with volume mount for model cache

## Security

- Set `ALLOWED_ORIGINS` to specific domains in production
- Use HTTPS in production
- Consider adding API key authentication for public APIs
- Rate limiting is enabled by default (60 req/min)
