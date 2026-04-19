import os

# Server Configuration
PORT = 5000
DEBUG = True

# Database Configuration
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb+srv://mahajanharshal36_db_user:b6TZuju9aksccG5P@cluster0.t4qos1y.mongodb.net/?appName=Cluster0")

# Security Configuration
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-super-secret-key-change-this-in-production")
JWT_ACCESS_TOKEN_EXPIRES = 15 * 60  # 15 minutes in seconds

# Proxy Configuration (DEPRECATED: Now handled natively)
NODE_SERVER_URL = os.environ.get("NODE_SERVER_URL", "http://127.0.0.1:5001")

# Media Configuration
ALLOWED_TYPES = {
    "image/jpeg", "image/png", "image/jpg", "image/webp", 
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/webm",
    "audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp3", "audio/ogg"
}

ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png", "webp", 
    "mp4", "mov", "avi", "webm", 
    "mp3", "wav", "ogg"
}

# Model Paths
VISION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model-v3.pt")
