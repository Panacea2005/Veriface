"""
Quick Webcam Capture for Anti-Spoof Testing

This script helps you quickly capture test images:
1. Press 'r' to capture REAL face from webcam
2. Press 's' to capture SPOOF attack (photo/screen)
3. Press 'q' to quit

Images are automatically saved to Test/real_faces/ or Test/spoof_attacks/
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Create directories
real_faces_dir = Path("d:/Swinburne/COS30082 - Applied Machine Learning/Project/Test/real_faces")
spoof_attacks_dir = Path("d:/Swinburne/COS30082 - Applied Machine Learning/Project/Test/spoof_attacks")

real_faces_dir.mkdir(parents=True, exist_ok=True)
spoof_attacks_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("WEBCAM CAPTURE FOR ANTI-SPOOF TESTING")
print("="*80)
print()
print("INSTRUCTIONS:")
print("-" * 80)
print("Mode 1: REAL FACE capture")
print("  - Look directly at webcam")
print("  - Press 'r' to capture")
print("  - Try different lighting, angles, expressions")
print("  - Capture at least 20 images")
print()
print("Mode 2: SPOOF ATTACK capture")
print("  - Hold a printed photo in front of webcam, OR")
print("  - Display your photo on phone/tablet screen")
print("  - Press 's' to capture")
print("  - Try different photo qualities, distances")
print("  - Capture at least 20 images")
print()
print("Press 'q' to quit")
print("-" * 80)
print()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam!")
    print("Check if:")
    print("  1. Webcam is connected")
    print("  2. No other app is using the webcam")
    print("  3. Webcam drivers are installed")
    exit(1)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

real_count = 0
spoof_count = 0
mode = "REAL"  # Current capture mode

print("[OK] Webcam opened successfully")
print(f"[TIP] Current mode: {mode} FACE - Press 'r' to capture, 'm' to switch mode, 'q' to quit")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from webcam")
        break
    
    # Display frame with info
    display = frame.copy()
    
    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0) if mode == "REAL" else (0, 0, 255)
    
    cv2.putText(display, f"Mode: {mode} FACE", (10, 30), font, 0.8, color, 2)
    cv2.putText(display, f"Real captured: {real_count}", (10, 60), font, 0.6, (255, 255, 255), 2)
    cv2.putText(display, f"Spoof captured: {spoof_count}", (10, 85), font, 0.6, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(display, "Press 'r': Capture REAL face", (10, 430), font, 0.5, (255, 255, 255), 1)
    cv2.putText(display, "Press 's': Capture SPOOF attack", (10, 450), font, 0.5, (255, 255, 255), 1)
    cv2.putText(display, "Press 'm': Switch mode | 'q': Quit", (10, 470), font, 0.5, (255, 255, 255), 1)
    
    # Show frame
    cv2.imshow('Webcam - Anti-Spoof Test Capture', display)
    
    # Handle key press
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n[OK] Quitting...")
        break
    
    elif key == ord('m'):
        # Switch mode
        mode = "SPOOF" if mode == "REAL" else "REAL"
        print(f"[MODE] Switched to: {mode} FACE")
    
    elif key == ord('r'):
        # Capture REAL face
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"real_face_{timestamp}.jpg"
        filepath = real_faces_dir / filename
        cv2.imwrite(str(filepath), frame)
        real_count += 1
        print(f"[SAVE] Real face #{real_count}: {filename}")
    
    elif key == ord('s'):
        # Capture SPOOF attack
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"spoof_attack_{timestamp}.jpg"
        filepath = spoof_attacks_dir / filename
        cv2.imwrite(str(filepath), frame)
        spoof_count += 1
        print(f"[SAVE] Spoof attack #{spoof_count}: {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()

print()
print("="*80)
print("CAPTURE SUMMARY")
print("="*80)
print(f"Real faces captured:   {real_count}")
print(f"Spoof attacks captured: {spoof_count}")
print()
print(f"Saved to:")
print(f"  - {real_faces_dir}")
print(f"  - {spoof_attacks_dir}")
print()

if real_count >= 20 and spoof_count >= 20:
    print("[PASS] Good! You have enough images for threshold testing.")
    print()
    print("Next steps:")
    print("  1. Run: python test_antispoof_threshold.py")
    print("  2. Check the recommended threshold")
    print("  3. Update thresholds.yaml")
elif real_count < 20:
    print(f"[WARN] Need {20 - real_count} more REAL face images")
    print("       Run this script again and press 'r' to capture more")
elif spoof_count < 20:
    print(f"[WARN] Need {20 - spoof_count} more SPOOF attack images")
    print("       Run this script again with printed photo and press 's'")

print("="*80)
