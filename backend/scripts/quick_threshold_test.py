"""
Quick threshold comparison - Test Model A with different thresholds.

Shows how threshold affects matching behavior with your registry.

Usage:
    python quick_threshold_test.py
"""

import requests
import base64
from pathlib import Path
import yaml

# API endpoint
API_URL = "http://localhost:8000/api/verify"
THRESHOLDS_PATH = Path("Veriface/backend/app/core/thresholds.yaml")


def load_current_threshold():
    """Load current threshold from config."""
    with open(THRESHOLDS_PATH) as f:
        config = yaml.safe_load(f)
    return config['similarity']['cosine']['threshold']


def update_threshold(new_threshold: float):
    """Update threshold in config file."""
    with open(THRESHOLDS_PATH) as f:
        config = yaml.safe_load(f)
    
    config['similarity']['cosine']['threshold'] = new_threshold
    
    with open(THRESHOLDS_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Updated threshold to {new_threshold}")


def test_image_with_threshold(image_path: Path, threshold: float):
    """Test an image with a specific threshold."""
    # Update threshold
    update_threshold(threshold)
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    image_b64 = base64.b64encode(image_data).decode('utf-8')
    
    # Send request
    data = {
        'model': 'A',
        'metric': 'cosine',
        'image_b64': image_b64
    }
    
    try:
        response = requests.post(API_URL, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return {
                'matched_id': result.get('matched_id'),
                'score': result.get('score'),
                'liveness_passed': result['liveness']['passed'],
                'liveness_score': result['liveness']['score'],
                'all_scores': result.get('all_scores', [])
            }
        else:
            return {'error': response.text}
    except Exception as e:
        return {'error': str(e)}


def main():
    """Main testing function."""
    print("="*70)
    print("QUICK THRESHOLD TESTING - MODEL A")
    print("="*70)
    
    # Get test image
    test_image = input("\nEnter path to test image (or press Enter for demo): ").strip()
    
    if not test_image:
        # Use default test image
        test_dir = Path("Test")
        test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        
        if not test_images:
            print("\n❌ No test images found in Test/ directory")
            print("Please provide an image path or add images to Test/")
            return
        
        test_image = test_images[0]
        print(f"Using test image: {test_image}")
    else:
        test_image = Path(test_image)
        if not test_image.exists():
            print(f"❌ Image not found: {test_image}")
            return
    
    # Test with different thresholds
    thresholds = [0.20, 0.23, 0.25, 0.30, 0.35, 0.40]
    
    print("\n" + "="*70)
    print("TESTING WITH DIFFERENT THRESHOLDS")
    print("="*70)
    print(f"\nTest image: {test_image.name}")
    print("\nMake sure backend is running: cd Veriface/backend && run.bat")
    print()
    
    results = []
    
    for threshold in thresholds:
        print(f"\nThreshold {threshold:.2f}:")
        print("-" * 50)
        
        result = test_image_with_threshold(test_image, threshold)
        
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
            continue
        
        # Store result
        results.append({
            'threshold': threshold,
            'result': result
        })
        
        # Display result
        if not result['liveness_passed']:
            print(f"  🚫 Spoof detected (liveness: {result['liveness_score']:.2f})")
        elif result['matched_id']:
            print(f"  ✅ MATCHED: {result['matched_id']}")
            print(f"     Score: {result['score']:.4f} ({result['score']*100:.2f}%)")
        else:
            print(f"  ❌ No match")
            if result['all_scores']:
                best = result['all_scores'][0]
                print(f"     Best score: {best['score']:.4f} ({best['percentage']:.2f}%)")
                print(f"     User: {best['user_id']}")
        
        # Show top 3 scores
        if result['all_scores'] and len(result['all_scores']) > 1:
            print(f"     Top 3 scores:")
            for i, score_info in enumerate(result['all_scores'][:3], 1):
                print(f"       {i}. {score_info['user_id']}: {score_info['score']:.4f} ({score_info['percentage']:.2f}%)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if not results:
        print("\n❌ No successful tests!")
        return
    
    print(f"\nTest image: {test_image.name}")
    print("\n{:<12} {:<15} {:<20} {:<10}".format("Threshold", "Result", "User ID", "Score (%)"))
    print("-" * 70)
    
    for r in results:
        threshold = r['threshold']
        result = r['result']
        
        if not result['liveness_passed']:
            status = "🚫 SPOOF"
            user_id = "-"
            score_pct = f"{result['liveness_score']*100:.1f}"
        elif result['matched_id']:
            status = "✅ MATCH"
            user_id = result['matched_id']
            score_pct = f"{result['score']*100:.1f}"
        else:
            status = "❌ NO MATCH"
            if result['all_scores']:
                user_id = result['all_scores'][0]['user_id']
                score_pct = f"{result['all_scores'][0]['percentage']:.1f}"
            else:
                user_id = "-"
                score_pct = "0.0"
        
        print(f"{threshold:.2f}         {status:<15} {user_id:<20} {score_pct}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    matched_thresholds = [r['threshold'] for r in results 
                         if r['result'].get('matched_id')]
    
    if matched_thresholds:
        print(f"\n✅ Image matched at thresholds: {matched_thresholds}")
        print(f"   Lowest threshold that accepts: {min(matched_thresholds):.2f}")
        print(f"   Highest threshold that accepts: {max(matched_thresholds):.2f}")
    else:
        print("\n❌ Image did not match at any threshold")
        if results and results[0]['result']['all_scores']:
            best_score = results[0]['result']['all_scores'][0]['score']
            print(f"   Best similarity score: {best_score:.4f} ({best_score*100:.2f}%)")
            print(f"   Would need threshold ≤ {best_score:.2f} to match")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
Based on your testing:

1. Threshold TOO LOW (0.20-0.23):
   - Accepts almost everyone (high false accepts)
   - ⚠️ Security risk
   - Only use if user convenience is critical

2. BALANCED (0.25-0.30):
   - Good trade-off between security and convenience
   - ✅ Recommended for most cases
   - Current production setting: 0.30

3. Threshold TOO HIGH (0.35-0.40):
   - Very strict (high false rejects)
   - 🔒 Use for high-security scenarios
   - Legitimate users may be rejected

NEXT STEPS:
1. Test with multiple registered users
2. Test with imposters (similar-looking people)
3. Monitor FAR (False Accept Rate) in production
4. Adjust threshold based on real usage patterns
    """)
    
    # Restore original threshold
    original_threshold = 0.30
    update_threshold(original_threshold)
    print(f"\n✅ Restored threshold to {original_threshold}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        # Restore threshold
        update_threshold(0.30)
        print("✅ Restored threshold to 0.30")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
