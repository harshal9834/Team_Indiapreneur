import os
import time
import io
import json
import csv
import tempfile
import requests
import cv2
import numpy as np
import torch
from flask import Blueprint, request, jsonify, send_file, Response
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from PIL import Image
from fpdf import FPDF

from .config import ALLOWED_TYPES, ALLOWED_EXTENSIONS
from .models import vision_model, audio_pipe, preprocess, face_cascade, DEVICE
from .utils import generate_heatmap
from .services import (
    generate_multi_layered_forensic, 
    get_recommendations, 
    save_report, 
    session_audit_logs, 
    last_detailed_report
)
from .auth_helpers import (
    find_user_by_email, 
    create_user, 
    check_password, 
    format_user_response
)

api = Blueprint('api', __name__)

# --- AUTH ROUTES ---

@api.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user')

    if not name or not email or not password:
        return jsonify({"message": "Please provide name, email and password"}), 400

    try:
        if find_user_by_email(email):
            return jsonify({"message": "User already exists"}), 400
        
        user = create_user(name, email, password, role)
        access_token = create_access_token(identity={"id": user["_id"], "role": user["role"]})
        
        return jsonify({
            "user": user,
            "accessToken": access_token,
            "message": "User registered successfully"
        }), 201
    except Exception as e:
        return jsonify({"message": str(e)}), 500

@api.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"message": "Please provide email and password"}), 400

    try:
        user = find_user_by_email(email)
        if user and check_password(password, user['password']):
            formatted_user = format_user_response(user)
            access_token = create_access_token(identity={"id": formatted_user["_id"], "role": formatted_user["role"]})
            
            return jsonify({
                "user": formatted_user,
                "accessToken": access_token
            }), 200
        else:
            return jsonify({"message": "Invalid email or password"}), 401
    except Exception as e:
        return jsonify({"message": str(e)}), 500

@api.route('/auth/logout', methods=['POST'])
def logout():
    # With JWT-Extended, logout is usually handled on the frontend by removing the token.
    # If using cookies, we would clear them here.
    return jsonify({"message": "Logged out successfully"}), 200


# --- PROTECTED DETECT ROUTE ---

@api.route("/detect", methods=["POST"])
# @jwt_required() # Uncomment this to enable mandatory login for scans
def detect():
    # Check if a file is provided
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    content_type = file.content_type or ""
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    is_valid_type = content_type in ALLOWED_TYPES or ext in ALLOWED_EXTENSIONS
    if not is_valid_type:
        return jsonify({"error": "Invalid type. Supports IMG/VID/AUD."}), 415

    is_video = content_type.startswith("video") or ext in {"mp4", "mov", "avi", "webm"}
    is_audio = content_type.startswith("audio") or ext in {"mp3", "wav", "ogg"}
    
    media_category = "Video" if is_video else ("Audio" if is_audio else "Image")
    
    try:
        heatmap_b64 = None
        raw_confidence = 0
        is_fake = False
        start = time.time()

        # --- AUDIO PROCESSING ---
        if is_audio:
            temp_path = os.path.join(tempfile.gettempdir(), file.filename)
            file.save(temp_path)
            results = audio_pipe(temp_path)
            os.remove(temp_path)
            
            top_pred = sorted(results, key=lambda x: x['score'], reverse=True)[0]
            label = top_pred['label'].upper()
            is_fake = "FAKE" in label
            raw_confidence = round(top_pred['score'] * 100, 1)

        # --- IMAGE/VIDEO PROCESSING ---
        else:
            if is_video:
                temp_path = os.path.join(tempfile.gettempdir(), file.filename)
                file.save(temp_path)
                cap = cv2.VideoCapture(temp_path)
                ret, frame = cap.read()
                cap.release()
                os.remove(temp_path)
                if not ret: return jsonify({"error": "Error reading video"}), 422
                frame_for_face = frame.copy()
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                img_bytes = file.read()
                frame_for_face = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Face Detection
            detected_faces = []
            if frame_for_face is not None:
                gray = cv2.cvtColor(frame_for_face, cv2.COLOR_BGR2GRAY)
                detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(detected_faces) == 0:
                    return jsonify({"error": "No human face detected. Visual analysis requires a face."}), 422

            # Vision Inference
            tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = vision_model(tensor)
                probs = torch.softmax(out, dim=1)[0]
                conf, pred = torch.max(probs, dim=0)
            
            raw_confidence = round(conf.item() * 100, 1)
            is_fake = (pred.item() == 1)
            
            # Visual Heatmap
            heatmap_b64 = generate_heatmap(frame_for_face, detected_faces, is_fake=is_fake)

        end = time.time()
        
        # Risk Logic
        if is_fake:
            risk_level = "HIGH 🔴" if raw_confidence > 80 else ("MEDIUM 🟡" if raw_confidence >= 60 else "LOW 🟢")
        else:
            risk_level = "LOW 🟢" if raw_confidence > 80 else ("MEDIUM 🟡" if raw_confidence >= 60 else "HIGH 🔴")

        multi_forensic = generate_multi_layered_forensic(raw_confidence, is_fake, media_type=media_category)
        recommendations = get_recommendations(is_fake, risk_level)
        trust_score = round(100 - raw_confidence) if is_fake else round(raw_confidence)
        
        stack_contribution = [
            {"tech": "EfficientNet-B0 (Vision)", "role": "Analyzes facial geometry, textures, and lighting patterns to detect synthetic visual artifacts."},
            {"tech": "Wav2Vec2 (Audio)", "role": "Processes acoustic signals and spectral density to identify synthetic speech and prosody anomalies."},
            {"tech": "OpenCV (Haar Cascades)", "role": "Real-time face detection and isolation for precise regional forensic analysis."},
            {"tech": "PyTorch (Neural Engine)", "role": "High-performance tensor processing for real-time inference across deep learning architectures."}
        ]
        
        output = {
            "verdict_data": multi_forensic["verdict_header"],
            "summary": {
                "prediction": "FAKE" if is_fake else "REAL",
                "confidence": raw_confidence,
                "risk_level": risk_level,
                "trust_score": trust_score
            },
            "fake_signal_strength": raw_confidence if is_fake else round(100 - raw_confidence, 1),
            "forensic_analysis": multi_forensic,
            "recommendation": recommendations,
            "heatmap_base64": heatmap_b64,
            "media_type": media_category,
            "stack_contribution": stack_contribution,
            "system_info": {
                "processing_time": f"{round(end-start, 2)}s",
                "analysis_type": multi_forensic["system_tags"]["analysis_type"],
                "engine": "Wav2Vec2-Forensic" if is_audio else "EfficientNet-B0-Vision"
            }
        }
        
        save_report(output, file.filename)
        return jsonify(output)

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"error": str(e)}), 500


# --- REPORT ROUTES ---

@api.route("/download-report")
def download_report():
    report_path = "report.json"
    with open(report_path, "w", encoding='utf-8') as f:
        json.dump(session_audit_logs, f, indent=4)
    return send_file(report_path, as_attachment=True)

@api.route("/download-report-csv")
def download_csv():
    filename = "report.csv"
    with open(filename, "w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["File Name", "Result", "Confidence", "Risk Level", "Time"])
        for r in session_audit_logs:
            writer.writerow([r["file"], r["result"], r["confidence"], r["risk"], r["time"]])
    return send_file(filename, as_attachment=True)

@api.route("/download-report-pdf")
def download_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(190, 10, "Verionyx AI - Forensic Audit Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "File Name", 1)
    pdf.cell(30, 10, "Result", 1)
    pdf.cell(30, 10, "Confidence", 1)
    pdf.cell(30, 10, "Risk Level", 1)
    pdf.cell(60, 10, "Time", 1)
    pdf.ln()
    
    pdf.set_font("Arial", "", 10)
    for r in session_audit_logs:
        risk_text = str(r["risk"]).split(" ")[0]
        pdf.cell(40, 10, str(r["file"][:20]), 1)
        pdf.cell(30, 10, str(r["result"]), 1)
        pdf.cell(30, 10, f"{r['confidence']}%", 1)
        pdf.cell(30, 10, risk_text, 1)
        pdf.cell(60, 10, str(r["time"]), 1)
        pdf.ln()
        
    filename = "report.pdf"
    pdf.output(filename)
    return send_file(filename, as_attachment=True)

@api.route("/download-latest-report")
def download_latest_report():
    if not last_detailed_report:
        return "No report generated yet", 404
        
    rep = last_detailed_report
    d = rep["data"]
    forensic = d["forensic_analysis"]
    
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 22)
    pdf.set_text_color(22, 160, 133)
    pdf.cell(190, 15, "Verionyx AI - Forensic Brief", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(190, 8, f"Scan timestamp: {rep['time']}", ln=True, align="C")
    pdf.ln(10)
    
    # Executive Summary
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(190, 10, f" Forensic Analysis for: {rep['file']}", ln=True, fill=True)
    pdf.ln(5)
    
    # Verdict
    pdf.set_font("Arial", "B", 14)
    res_color = (192, 57, 43) if "FAKE" in d["summary"]["prediction"].upper() else (39, 174, 96)
    pdf.set_text_color(*res_color)
    pdf.cell(95, 10, f"VERDICT: {d['summary']['prediction']}", 0)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(95, 10, f"CONFIDENCE: {d['summary']['confidence']}%", ln=True, align="R")
    
    pdf.ln(5)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(190, 6, forensic["explanation"])
    pdf.ln(5)

    # Technical Table
    pdf.set_font("Arial", "B", 12)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(190, 10, " Forensic Feature metrics", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(60, 8, "Feature", 1)
    pdf.cell(30, 8, "Value", 1)
    pdf.cell(100, 8, "Forensic Interpretation", 1)
    pdf.ln()
    pdf.set_font("Arial", "", 9)
    for f_row in forensic["feature_table"]:
        pdf.cell(60, 8, str(f_row["feature"]), 1)
        pdf.cell(30, 8, str(f_row.get("value", "")), 1)
        pdf.cell(100, 8, str(f_row["interpretation"]), 1)
        pdf.ln()

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, "Verionyx AI - Digital Trust Platform | Confidential Forensic Audit", align="C")
        
    filename = f"forensic_report_{rep['file'].replace('.', '_')}.pdf"
    pdf.output(filename)
    return send_file(filename, as_attachment=True)
