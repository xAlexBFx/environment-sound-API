# Dockerfile for YAMNet Sound Classification API
FROM python:3.11-slim

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/app_production.py app.py
COPY backend/.env.example .env

# Create directory for model cache
RUN mkdir -p /app/models

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')" || exit 1

# Run with gunicorn (optimized for HF free tier)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "-t", "300", "--preload", "--max-requests", "10", "--max-requests-jitter", "5", "app:app"]
