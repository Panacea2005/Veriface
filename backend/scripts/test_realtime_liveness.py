"""
Test script for real-time liveness detection endpoint

This script tests the new /api/liveness/realtime endpoint with sample images.
"""

import requests
import sys
from pathlib import Path
import time

# API endpoint
API_BASE = "http://localhost:8000"
LIVENESS_ENDPOINT = f"{API_BASE}/api/liveness/realtime"
HEALTH_ENDPOINT = f"{API_BASE}/api/liveness/health"

def test_health():
    """Test the liveness health endpoint."""
    print("=" * 80)
    print("Testing Liveness Health Endpoint")
    print("=" * 80)
    
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {data}")
        print(f"✓ Health check {'PASSED' if data.get('ready') else 'FAILED'}")
        print()
        return data.get('ready', False)
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        print()
        return False

def test_liveness(image_path: Path, description: str):
    """Test liveness detection with a single image."""
    print("-" * 80)
    print(f"Testing: {description}")
    print(f"Image: {image_path.name}")
    print("-" * 80)
    
    if not image_path.exists():
        print(f"✗ Image not found: {image_path}")
        print()
        return
    
    try:
        # Measure time
        start_time = time.time()
        
        with open(image_path, 'rb') as f:
            files = {'image': (image_path.name, f, 'image/jpeg')}
            response = requests.post(LIVENESS_ENDPOINT, files=files, timeout=10)
        
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        
        # Parse response
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {data}")
        print()
        print(f"Score: {data.get('score', 0):.4f}")
        print(f"Passed: {data.get('passed', False)}")
        print(f"Is Real: {data.get('is_real', False)}")
        print(f"Status: {data.get('status', 'unknown')}")
        print(f"Message: {data.get('message', 'N/A')}")
        print(f"Processing Time (Server): {data.get('processing_time_ms', 0):.2f} ms")
        print(f"Total Round-trip Time: {elapsed:.2f} ms")
        
        # Verdict
        if data.get('status') == 'success':
            if data.get('passed'):
                print("✓ REAL FACE DETECTED")
            else:
                print("✗ SPOOF DETECTED")
        else:
            print(f"⚠ ERROR: {data.get('message', 'Unknown error')}")
        
        print()
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print()

def main():
    """Run all liveness tests."""
    print("\n" + "=" * 80)
    print("REAL-TIME LIVENESS DETECTION TEST SUITE")
    print("=" * 80)
    print()
    
    # Check health first
    if not test_health():
        print("⚠ Warning: Health check failed, but continuing with tests...")
        print()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Test with real faces
    real_faces_dir = project_root / "Test" / "real_faces"
    if real_faces_dir.exists():
        print("=" * 80)
        print("TESTING REAL FACES")
        print("=" * 80)
        print()
        
        real_images = list(real_faces_dir.glob("*.jpg")) + list(real_faces_dir.glob("*.png"))
        for i, img_path in enumerate(real_images[:3], 1):  # Test first 3
            test_liveness(img_path, f"Real Face #{i}")
    else:
        print(f"⚠ Real faces directory not found: {real_faces_dir}")
        print()
    
    # Test with spoof attacks
    spoof_dir = project_root / "Test" / "spoof_attacks"
    if spoof_dir.exists():
        print("=" * 80)
        print("TESTING SPOOF ATTACKS")
        print("=" * 80)
        print()
        
        spoof_images = list(spoof_dir.glob("*.jpg")) + list(spoof_dir.glob("*.png"))
        for i, img_path in enumerate(spoof_images[:3], 1):  # Test first 3
            test_liveness(img_path, f"Spoof Attack #{i}")
    else:
        print(f"⚠ Spoof attacks directory not found: {spoof_dir}")
        print()
    
    print("=" * 80)
    print("TEST SUITE COMPLETED")
    print("=" * 80)
    print()
    print("Notes:")
    print("- Real faces should have high scores (>0.5) and passed=True")
    print("- Spoof attacks should have low scores (<0.5) and passed=False")
    print("- Processing time should be ~300-500ms for good performance")
    print()

if __name__ == "__main__":
    main()
