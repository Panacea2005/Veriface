"""
Quick test to verify Top-K aggregation is working.
Run: python Veriface/backend/scripts/test_topk.py
"""

import requests
import base64
from pathlib import Path

# Test image
test_img = Path("Test/1.jpg")
if not test_img.exists():
    print("❌ Test image not found: Test/1.jpg")
    exit(1)

# Encode image
with open(test_img, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

# Send request
print("Testing Top-K aggregation...")
print("="*50)

response = requests.post(
    "http://localhost:8000/api/verify",
    data={
        'model': 'A',
        'metric': 'cosine',
        'image_b64': img_b64
    },
    timeout=30
)

if response.status_code == 200:
    result = response.json()
    print(f"✅ Success!")
    print(f"  Matched: {result.get('matched_id', 'None')}")
    print(f"  Score: {result['score']:.4f} ({result['score']*100:.2f}%)")
    print(f"  Threshold: {result['threshold']:.4f}")
    print(f"  Method: Top-K (k=3)")
    print(f"\n  Top 3 users:")
    for i, score_info in enumerate(result.get('all_scores', [])[:3], 1):
        print(f"    {i}. {score_info['user_id']}: {score_info['score']:.4f} ({score_info['percentage']:.2f}%)")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
