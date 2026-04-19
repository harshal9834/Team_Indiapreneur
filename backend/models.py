import torch
import cv2
from PIL import Image
from torchvision.models import efficientnet_b0
from torchvision import transforms
from transformers import pipeline
import os
from .config import VISION_MODEL_PATH

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face Detection Setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_vision_model():
    print("[Verionyx AI] Loading Vision Forensic Engine...")
    model = efficientnet_b0()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    
    if os.path.exists(VISION_MODEL_PATH):
        model.load_state_dict(torch.load(VISION_MODEL_PATH, map_location=DEVICE))
    else:
        print(f"[Verionyx AI] Warning: Vision model weights not found at {VISION_MODEL_PATH}")
        
    model.eval()
    model.to(DEVICE)
    print(f"[Verionyx AI] Vision model loaded [OK]")
    return model

def load_audio_model():
    print("[Verionyx AI] Initializing Audio Forensic Engine...")
    # Hemgg/Deepfake-audio-detection usually uses 'FAKE' and 'REAL' labels.
    audio_pipe = pipeline("audio-classification", 
                          model="Hemgg/Deepfake-audio-detection", 
                          device=0 if torch.cuda.is_available() else -1)
    print(f"[Verionyx AI] Audio model loaded [OK]")
    return audio_pipe

# Initialize models
vision_model = load_vision_model()
audio_pipe = load_audio_model()
