import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes - use 2 workers (more can cause memory issues with TensorFlow)
workers = 2
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2

# Preload app to share model in memory
preload_app = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "yamnet-api"

# Server mechanics
daemon = False
pidfile = None

# SSL (set these for production)
# keyfile = "/path/to/ssl/key.pem"
# certfile = "/path/to/ssl/cert.pem"
