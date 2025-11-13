"""
Visual comparison: Training preprocessing vs Backend preprocessing
Shows side-by-side comparison of how images are processed.
"""
import numpy as np
import cv2
from PIL import Image
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Veriface', 'backend'))

def training_preprocessing(img_rgb_pil):
    """Simulate training preprocessing (eval mode - no augmentation)"""
    # PIL Image in RGB
    # Resize to 112x112
    img_resized = img_rgb_pil.resize((112, 112), Image.BILINEAR)
    
    # ToTensor: converts [0,255] -> [0,1] and HWC -> CHW
    img_array = np.array(img_resized).astype(np.float32) / 255.0  # [0, 1]
    
    # Lambda: (x*255 - 127.5) / 128.0
    img_normalized = (img_array * 255.0 - 127.5) / 128.0
    
    # HWC -> CHW
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    
    return img_chw, img_normalized

def backend_preprocessing(img_bgr_cv2):
    """Backend preprocessing (current implementation)"""
    # OpenCV image in BGR
    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr_cv2, cv2.COLOR_BGR2RGB)
    
    # Resize to 112x112
    if img_rgb.shape[:2] != (112, 112):
        img_rgb = cv2.resize(img_rgb, (112, 112))
    
    # Normalize: (pixel - 127.5) / 128.0
    img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0
    
    # HWC -> CHW
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    
    return img_chw, img_normalized

def compare_preprocessing():
    """Compare preprocessing pipelines with synthetic test image"""
    print("=" * 80)
    print("PREPROCESSING COMPARISON: Training vs Backend")
    print("=" * 80)
    
    # Create synthetic test image (gradient pattern)
    # This helps visualize normalization differences
    height, width = 112, 112
    test_image_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient pattern: R channel 0->255, G channel 128, B channel 255->0
    for i in range(height):
        for j in range(width):
            test_image_rgb[i, j, 0] = int(255 * i / height)  # R: top=0, bottom=255
            test_image_rgb[i, j, 1] = 128                     # G: constant 128
            test_image_rgb[i, j, 2] = int(255 * (1 - j / width))  # B: left=255, right=0
    
    print("\n1. TEST IMAGE CREATED")
    print(f"   Shape: {test_image_rgb.shape}")
    print(f"   Dtype: {test_image_rgb.dtype}")
    print(f"   Range: [{test_image_rgb.min()}, {test_image_rgb.max()}]")
    print(f"   Mean: {test_image_rgb.mean():.2f}")
    
    # Process with training pipeline
    print("\n2. TRAINING PREPROCESSING")
    img_pil = Image.fromarray(test_image_rgb, mode='RGB')
    train_chw, train_hwc = training_preprocessing(img_pil)
    print(f"   Output shape (CHW): {train_chw.shape}")
    print(f"   Output dtype: {train_chw.dtype}")
    print(f"   Output range: [{train_chw.min():.6f}, {train_chw.max():.6f}]")
    print(f"   Output mean: {train_chw.mean():.6f}")
    print(f"   Output std: {train_chw.std():.6f}")
    
    # Process with backend pipeline
    print("\n3. BACKEND PREPROCESSING")
    # Convert RGB -> BGR for OpenCV (simulate webcam input)
    img_bgr = cv2.cvtColor(test_image_rgb, cv2.COLOR_RGB2BGR)
    backend_chw, backend_hwc = backend_preprocessing(img_bgr)
    print(f"   Output shape (CHW): {backend_chw.shape}")
    print(f"   Output dtype: {backend_chw.dtype}")
    print(f"   Output range: [{backend_chw.min():.6f}, {backend_chw.max():.6f}]")
    print(f"   Output mean: {backend_chw.mean():.6f}")
    print(f"   Output std: {backend_chw.std():.6f}")
    
    # Compare outputs
    print("\n4. COMPARISON")
    print("-" * 80)
    
    # Pixel-wise difference
    diff = np.abs(train_chw - backend_chw)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"   Max absolute difference: {max_diff:.10f}")
    print(f"   Mean absolute difference: {mean_diff:.10f}")
    print(f"   Relative error: {mean_diff / (np.abs(train_chw).mean() + 1e-8) * 100:.6f}%")
    
    if max_diff < 1e-6:
        print("\n   ✅ PERFECT MATCH! Preprocessing is identical.")
    elif max_diff < 1e-3:
        print("\n   ✅ VERY CLOSE! Difference is negligible (< 0.001).")
    else:
        print(f"\n   ⚠️ WARNING! Significant difference detected: {max_diff:.6f}")
    
    # Channel-wise comparison
    print("\n5. CHANNEL-WISE ANALYSIS")
    print("-" * 80)
    for channel, name in enumerate(['R', 'G', 'B']):
        train_ch = train_chw[channel]
        backend_ch = backend_chw[channel]
        ch_diff = np.abs(train_ch - backend_ch)
        
        print(f"   {name} Channel:")
        print(f"      Training - min: {train_ch.min():.6f}, max: {train_ch.max():.6f}, mean: {train_ch.mean():.6f}")
        print(f"      Backend  - min: {backend_ch.min():.6f}, max: {backend_ch.max():.6f}, mean: {backend_ch.mean():.6f}")
        print(f"      Diff     - max: {ch_diff.max():.10f}, mean: {ch_diff.mean():.10f}")
    
    # Test with specific pixel values
    print("\n6. SPECIFIC PIXEL VALUE TESTS")
    print("-" * 80)
    test_pixels = [0, 64, 127, 128, 192, 255]
    
    for pixel_val in test_pixels:
        # Create uniform image with this pixel value
        uniform_img = np.full((112, 112, 3), pixel_val, dtype=np.uint8)
        
        # Training pipeline
        img_pil = Image.fromarray(uniform_img, mode='RGB')
        train_result, _ = training_preprocessing(img_pil)
        train_value = train_result[0, 0, 0]  # First channel, first pixel
        
        # Backend pipeline
        img_bgr = cv2.cvtColor(uniform_img, cv2.COLOR_RGB2BGR)
        backend_result, _ = backend_preprocessing(img_bgr)
        backend_value = backend_result[0, 0, 0]
        
        diff = abs(train_value - backend_value)
        match = "✓" if diff < 1e-6 else "✗"
        
        print(f"   Pixel {pixel_val:3d}: Train={train_value:8.6f}, Backend={backend_value:8.6f}, Diff={diff:.10f} {match}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    
    if max_diff < 1e-6:
        print("✅ SUCCESS: Backend preprocessing matches training EXACTLY!")
        print("   Models trained with this preprocessing will work correctly in production.")
        return True
    else:
        print("❌ FAILURE: Preprocessing mismatch detected!")
        print(f"   Max difference: {max_diff:.10f}")
        print("   This will cause accuracy degradation in production.")
        return False

if __name__ == "__main__":
    success = compare_preprocessing()
    sys.exit(0 if success else 1)
