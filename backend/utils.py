import cv2
import numpy as np
import base64

def generate_heatmap(image_bgr, faces, is_fake=True):
    """
    Generates a forensic heatmap overlay for detected faces.
    """
    if faces is None or len(faces) == 0:
        return None

    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    overlay_color = [0, 0, 255] if is_fake else [0, 255, 0]
    
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        cv2.ellipse(mask, center, (w // 2, h // 2), 0, 0, 360, 255, -1)
        
        glow_mask = np.zeros_like(mask)
        cv2.ellipse(glow_mask, center, (int(w * 0.65), int(h * 0.65)), 0, 0, 360, 180, -1)
        mask = cv2.max(mask, glow_mask)
        
    mask = cv2.GaussianBlur(mask, (81, 81), 0)
    blurred_bg = cv2.GaussianBlur(image_bgr, (25, 25), 0)
    
    overlay = np.zeros_like(image_bgr)
    overlay[:, :] = overlay_color
    
    alpha = 0.45
    heatmap_mix = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    focused_img = (heatmap_mix * mask_3d + blurred_bg * (1 - mask_3d)).astype(np.uint8)
    
    _, buffer = cv2.imencode('.jpg', focused_img)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
