# YAMNet Sound Classification API - Modal Deployment

A serverless API for classifying environmental sounds using Google's YAMNet deep learning model, deployed on Modal.

## Quick Start

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal setup
```

### 3. Deploy the API

```bash
modal deploy modal_app.py
```

### 4. Test the API

```bash
# Health check
curl https://your-app-name.modal.run/health

# Get API info
curl https://your-app-name.modal.run/info
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and service status |
| `/info` | GET | API information and endpoints |
| `/classify` | POST | Classify audio (returns top prediction) |
| `/classify/raw` | POST | Classify audio (returns top 5 predictions) |
| `/embeddings` | POST | Extract 1024-dimensional YAMNet embeddings |

## Example Usage

### Classify Audio

```bash
curl -X POST https://your-app-name.modal.run/classify \
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

### Extract Embeddings

```bash
curl -X POST https://your-app-name.modal.run/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "base64_encoded_audio_data_here"
  }'
```

## Modal Configuration

The app is configured with:

- **Image**: Debian Slim with Python 3.11
- **Dependencies**: TensorFlow, librosa, and audio processing libraries
- **Volumes**: Persistent storage for model caching
- **Timeout**: 300 seconds for classification tasks
- **Concurrency**: Up to 10 concurrent requests
- **Idle Timeout**: 300 seconds (5 minutes)

## Cost Optimization

- **Model Caching**: YAMNet model is cached in persistent volume to avoid re-downloading
- **Container Reuse**: Containers stay warm for 5 minutes after last request
- **Lazy Loading**: Model loads only when first needed
- **Memory Optimization**: Single worker per container to manage memory usage

## Local Development

For local testing:

```bash
# Install dependencies
pip install -r requirements-modal.txt
pip install -r backend/requirements.txt

# Run locally
python modal_app.py
```

The API will be available at `http://localhost:8000`

## Monitoring

Monitor your Modal deployment at:

- [Modal Dashboard](https://modal.com/dashboard)
- View function logs and metrics
- Track usage and costs

## Differences from Docker Version

| Feature | Docker | Modal |
|---------|--------|-------|
| Scaling | Manual | Automatic serverless |
| Model Loading | On startup | Lazy loading with caching |
| Cost | Always running | Pay-per-use |
| Scaling | Limited | Infinite horizontal scaling |
| Maintenance | High | Low (managed) |

## Deployment Commands

```bash
# Deploy to Modal
modal deploy modal_app.py

# View app status
modal app list

# View function logs
modal app logs yamnet-sound-classifier

# Update deployment
modal deploy modal_app.py --force
```

## Troubleshooting

### Common Issues

1. **Model Loading Timeout**: Increase timeout in function definition
2. **Memory Issues**: Reduce concurrent inputs or use smaller model
3. **Cold Starts**: First request may be slower due to model download

### Debug Mode

For debugging, you can run the app in interactive mode:

```bash
modal run modal_app.py
```

## License

MIT License - See LICENSE file for details.
