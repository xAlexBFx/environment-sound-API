import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes - use 1 worker for HuggingFace Spaces memory constraints
workers = 1
worker_class = "sync"
worker_connections = 1000
timeout = 300
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
