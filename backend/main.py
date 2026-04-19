from flask import Flask, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import os
import static_ffmpeg
from .config import PORT, DEBUG, JWT_SECRET_KEY, JWT_ACCESS_TOKEN_EXPIRES
from .routes import api

def create_app():
    # Set up static folder to point to the frontend directory
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
    
    # Enable CORS
    CORS(app, supports_credentials=True)

    # JWT Configuration
    app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = JWT_ACCESS_TOKEN_EXPIRES
    jwt = JWTManager(app)

    # Initialize FFmpeg for audio processing
    try:
        static_ffmpeg.add_paths()
        print("[Verionyx AI] FFmpeg paths initialized.")
    except Exception as e:
        print(f"[Verionyx AI] Warning: Could not initialize static-ffmpeg: {e}")

    # Register API blueprint at root
    app.register_blueprint(api, url_prefix='/api') # Moved to /api for better organization

    # Explicitly serve index.html at root
    @app.route("/")
    def serve_index():
        return send_file(os.path.join(frontend_dir, "index.html"))

    # Serve other HTML files directly for easy navigation
    @app.route("/<page>.html")
    def serve_html(page):
        return send_file(os.path.join(frontend_dir, f"{page}.html"))

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=DEBUG, port=PORT)
