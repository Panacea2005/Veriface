"""
Visual test to demonstrate the difference between:
- Old method: Analyze cropped face only (fails to detect spoofs)
- New method: Analyze full frame with context (detects spoofs correctly)
"""
import numpy as np
import cv2

def simulate_webcam_real():
    """
    Simulate a real webcam frame with:
    - Person in environment
    - Complex background
    - Natural lighting
    """
    # Create 640x480 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Background: Room with varying textures (walls, furniture)
    # Left side: darker (wall)
    frame[:, :320] = np.random.randint(40, 70, (480, 320, 3), dtype=np.uint8)
    # Right side: lighter (window/light)
    frame[:, 320:] = np.random.randint(120, 160, (480, 320, 3), dtype=np.uint8)
    
    # Add some "furniture" (darker rectangles)
    cv2.rectangle(frame, (50, 300), (150, 450), (30, 25, 20), -1)  # Chair
    cv2.rectangle(frame, (450, 250), (600, 400), (35, 30, 25), -1)  # Table
    
    # Face region: 200x250 in center-ish
    face_x, face_y = 220, 115
    face_w, face_h = 200, 250
    
    # Face with skin tone and texture
    face = np.random.randint(180, 220, (face_h, face_w, 3), dtype=np.uint8)
    face[:, :, 0] = np.random.randint(140, 180, (face_h, face_w))  # B
    face[:, :, 1] = np.random.randint(160, 200, (face_h, face_w))  # G
    face[:, :, 2] = np.random.randint(180, 220, (face_h, face_w))  # R (more red for skin)
    
    # Add facial features (darker regions for eyes, mouth)
    # Eyes
    cv2.ellipse(face, (60, 80), (20, 12), 0, 0, 360, (50, 40, 30), -1)
    cv2.ellipse(face, (140, 80), (20, 12), 0, 0, 360, (50, 40, 30), -1)
    # Mouth
    cv2.ellipse(face, (100, 180), (40, 20), 0, 0, 180, (80, 60, 50), -1)
    
    # Add lighting gradient (natural light from one side)
    gradient = np.linspace(1.0, 1.2, face_w).reshape(1, -1, 1)  # Add channel dimension
    face = np.clip(face.astype(float) * gradient, 0, 255).astype(np.uint8)
    
    # Place face in frame
    frame[face_y:face_y+face_h, face_x:face_x+face_w] = face
    
    bbox = (face_x, face_y, face_w, face_h)
    return frame, bbox

def simulate_photo_spoof():
    """
    Simulate holding up a phone/paper with a photo:
    - Simple background (hand/wall)
    - Rectangular border (phone/paper edge)
    - Uniform lighting (screen glow or flat paper)
    """
    # Create 640x480 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Background: Simple uniform (hand or wall)
    frame[:] = np.random.randint(90, 110, (480, 640, 3), dtype=np.uint8)
    
    # Phone/paper region: 300x400 rectangle in center
    phone_x, phone_y = 170, 40
    phone_w, phone_h = 300, 400
    
    # Phone border (dark edge)
    cv2.rectangle(frame, (phone_x-5, phone_y-5), 
                  (phone_x+phone_w+5, phone_y+phone_h+5), 
                  (20, 20, 20), -1)
    
    # Phone screen with uniform lighting (screen glow)
    phone_screen = np.random.randint(140, 160, (phone_h, phone_w, 3), dtype=np.uint8)
    
    # Face on phone: 150x200 in center of phone
    face_x_on_phone = 75
    face_y_on_phone = 100
    face_w, face_h = 150, 200
    
    # Face with skin tone (but more uniform due to screen/print)
    face = np.random.randint(180, 200, (face_h, face_w, 3), dtype=np.uint8)
    face[:, :, 0] = 150  # More uniform B
    face[:, :, 1] = 170  # More uniform G
    face[:, :, 2] = 190  # More uniform R
    
    # Eyes and mouth (but less detailed due to photo quality)
    cv2.ellipse(face, (45, 60), (15, 10), 0, 0, 360, (60, 50, 40), -1)
    cv2.ellipse(face, (105, 60), (15, 10), 0, 0, 360, (60, 50, 40), -1)
    cv2.ellipse(face, (75, 140), (30, 15), 0, 0, 180, (90, 70, 60), -1)
    
    # Place face on phone screen
    phone_screen[face_y_on_phone:face_y_on_phone+face_h, 
                 face_x_on_phone:face_x_on_phone+face_w] = face
    
    # Place phone in frame
    frame[phone_y:phone_y+phone_h, phone_x:phone_x+phone_w] = phone_screen
    
    # Face bbox in FULL FRAME coordinates
    bbox = (phone_x + face_x_on_phone, phone_y + face_y_on_phone, face_w, face_h)
    return frame, bbox

def analyze_frame(frame, bbox, label):
    """Analyze frame with context-based features"""
    x, y, w, h = bbox
    img_h, img_w = frame.shape[:2]
    
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    
    # Extract regions
    face = frame[y:y+h, x:x+w]
    
    # Context region (1.25x face size)
    ctx_x1 = max(0, int(x - w * 0.25))
    ctx_y1 = max(0, int(y - h * 0.25))
    ctx_x2 = min(img_w, int(x + w * 1.25))
    ctx_y2 = min(img_h, int(y + h * 1.25))
    context = frame[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
    
    # Background (everything except face)
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    mask[y:y+h, x:x+w] = 0
    background = frame[mask > 0]
    
    # Convert to grayscale
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    context_gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
    
    # === ANALYSIS ===
    
    # 1. Background Complexity
    bg_std = np.std(background)
    bg_complexity = min(bg_std / 30.0, 1.0)
    print(f"\n1. Background Complexity:")
    print(f"   Std Dev: {bg_std:.2f}")
    print(f"   Score: {bg_complexity:.3f} {'✅' if bg_complexity > 0.5 else '❌'}")
    
    # 2. Lighting Consistency
    face_brightness = np.mean(face_gray)
    context_brightness = np.mean(context_gray)
    brightness_diff = abs(face_brightness - context_brightness)
    lighting_score = 1.0 - min(brightness_diff / 100.0, 1.0)
    print(f"\n2. Lighting Consistency:")
    print(f"   Face brightness: {face_brightness:.1f}")
    print(f"   Context brightness: {context_brightness:.1f}")
    print(f"   Difference: {brightness_diff:.1f}")
    print(f"   Score: {lighting_score:.3f} {'✅' if lighting_score > 0.5 else '❌'}")
    
    # 3. Edge Artifacts
    edges = cv2.Canny(context_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    edge_artifact_score = 1.0 - min(edge_density / 0.15, 1.0)
    print(f"\n3. Edge Artifacts (phone/paper borders):")
    print(f"   Edge density: {edge_density:.4f}")
    print(f"   Score: {edge_artifact_score:.3f} {'✅' if edge_artifact_score > 0.5 else '❌'}")
    
    # 4. Face-to-Frame Ratio
    face_area = w * h
    frame_area = img_w * img_h
    face_ratio = face_area / frame_area
    if face_ratio < 0.05:
        ratio_score = face_ratio / 0.05
    elif face_ratio > 0.25:
        ratio_score = max(0.0, 1.0 - (face_ratio - 0.25) / 0.25)
    else:
        ratio_score = 1.0
    print(f"\n4. Face-to-Frame Ratio:")
    print(f"   Face area: {face_area} pixels")
    print(f"   Frame area: {frame_area} pixels")
    print(f"   Ratio: {face_ratio:.3f} ({face_ratio*100:.1f}%)")
    print(f"   Score: {ratio_score:.3f} {'✅' if ratio_score > 0.5 else '❌'}")
    
    # 5. Depth Cues
    blur_face = cv2.GaussianBlur(face_gray, (5, 5), 0)
    blur_context = cv2.GaussianBlur(context_gray, (5, 5), 0)
    blur_diff_face = np.mean(np.abs(face_gray.astype(float) - blur_face.astype(float)))
    blur_diff_ctx = np.mean(np.abs(context_gray.astype(float) - blur_context.astype(float)))
    depth_score = min((blur_diff_face + blur_diff_ctx) / 20.0, 1.0)
    print(f"\n5. Depth/3D Cues:")
    print(f"   Face blur diff: {blur_diff_face:.2f}")
    print(f"   Context blur diff: {blur_diff_ctx:.2f}")
    print(f"   Score: {depth_score:.3f} {'✅' if depth_score > 0.5 else '❌'}")
    
    # === FINAL SCORE ===
    context_score = (
        0.25 * bg_complexity +
        0.20 * lighting_score +
        0.20 * edge_artifact_score +
        0.15 * depth_score +
        0.10 * ratio_score +
        0.10 * 0.5  # Color score placeholder
    )
    
    # Texture score (simplified)
    face_variance = np.var(face_gray)
    texture_score = min(face_variance / 1500.0, 1.0)
    
    final_score = 0.70 * context_score + 0.30 * texture_score
    
    print(f"\n{'='*60}")
    print(f"SCORES:")
    print(f"  Context Score: {context_score:.3f} (70% weight)")
    print(f"  Texture Score: {texture_score:.3f} (30% weight)")
    print(f"  FINAL SCORE: {final_score:.3f}")
    print(f"  Threshold: 0.500")
    print(f"  Result: {'✅ PASS (REAL)' if final_score > 0.5 else '❌ FAIL (SPOOF)'}")
    print(f"{'='*60}")
    
    return final_score

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ANTISPOOF CONTEXT-BASED DETECTION - VISUAL DEMONSTRATION")
    print("="*80)
    
    # Test 1: Real webcam
    print("\n\n### TEST 1: REAL WEBCAM (should PASS) ###")
    frame_real, bbox_real = simulate_webcam_real()
    score_real = analyze_frame(frame_real, bbox_real, "Real Webcam - Person in Environment")
    
    # Test 2: Photo spoof
    print("\n\n### TEST 2: PHOTO SPOOF (should FAIL) ###")
    frame_fake, bbox_fake = simulate_photo_spoof()
    score_fake = analyze_frame(frame_fake, bbox_fake, "Photo Spoof - Phone/Paper with Face Image")
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Real Webcam Score: {score_real:.3f} → {'✅ PASS' if score_real > 0.5 else '❌ FAIL'}")
    print(f"Photo Spoof Score: {score_fake:.3f} → {'✅ CORRECTLY REJECTED' if score_fake <= 0.5 else '❌ WRONGLY ACCEPTED'}")
    
    if score_real > 0.5 and score_fake <= 0.5:
        print("\n✅✅✅ SUCCESS! Context-based detection works correctly! ✅✅✅")
        print("Real faces pass, fake photos fail.")
    else:
        print("\n❌ ISSUE: Detection not working as expected.")
        if score_real <= 0.5:
            print("   - Real face incorrectly rejected")
        if score_fake > 0.5:
            print("   - Fake photo incorrectly accepted")
    
    print("="*80)
