"""
Test different thresholds for Model A in production-like scenario.

This script helps you find the optimal threshold by:
1. Testing multiple threshold values (0.20 - 0.40)
2. Calculating FAR (False Accept Rate) and FRR (False Reject Rate)
3. Simulating real-world conditions

Usage:
    python test_threshold_production.py
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2

# Add backend to path
backend_path = Path(__file__).parent / "Veriface" / "backend"
sys.path.insert(0, str(backend_path))

from app.pipelines.arcface_model import iresnet100
from app.pipelines.detector import FaceDetector
from app.core.config import MODELS_DIR


class ThresholdTester:
    """Test different thresholds for face verification."""
    
    def __init__(self, model_path: Path):
        """Initialize model and detector."""
        print("Loading Model A (iResNet100)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = iresnet100(num_features=512)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Face detector
        self.detector = FaceDetector()
        
        print(f"✅ Model loaded on {self.device}")
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image (matches training)."""
        # Detect and align face
        bbox = self.detector.detect(img)
        if bbox is None:
            raise ValueError("No face detected")
        
        face = self.detector.align(img, bbox)
        
        # Convert to float32 and normalize
        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0  # Range: [-0.996, 0.996]
        
        # CHW format
        face = np.transpose(face, (2, 0, 1))
        
        return face
    
    def extract_embedding(self, img_path: Path) -> np.ndarray:
        """Extract embedding from image."""
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        face = self.preprocess(img)
        
        # Extract embedding
        with torch.no_grad():
            face_tensor = torch.from_numpy(face).unsqueeze(0).to(self.device)
            embedding = self.model(face_tensor)
            
            # L2 normalization
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            embedding = embedding.cpu().numpy()[0]
        
        return embedding
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        # Embeddings are already L2-normalized, so dot product = cosine similarity
        return float(np.dot(emb1, emb2))
    
    def test_threshold_range(self, positive_pairs: list, negative_pairs: list, 
                            thresholds: list):
        """
        Test multiple thresholds.
        
        Args:
            positive_pairs: List of (emb1, emb2) for same person
            negative_pairs: List of (emb1, emb2) for different people
            thresholds: List of threshold values to test
        
        Returns:
            results: Dict with FAR, FRR for each threshold
        """
        results = {}
        
        print("\n" + "="*60)
        print("THRESHOLD TESTING")
        print("="*60)
        print(f"Positive pairs (same person): {len(positive_pairs)}")
        print(f"Negative pairs (different people): {len(negative_pairs)}")
        print()
        
        for threshold in thresholds:
            # Test positive pairs (should accept)
            false_rejects = 0
            positive_sims = []
            for emb1, emb2 in positive_pairs:
                sim = self.cosine_similarity(emb1, emb2)
                positive_sims.append(sim)
                if sim < threshold:  # Rejected (bad!)
                    false_rejects += 1
            
            # Test negative pairs (should reject)
            false_accepts = 0
            negative_sims = []
            for emb1, emb2 in negative_pairs:
                sim = self.cosine_similarity(emb1, emb2)
                negative_sims.append(sim)
                if sim >= threshold:  # Accepted (bad!)
                    false_accepts += 1
            
            # Calculate rates
            frr = false_rejects / len(positive_pairs) if positive_pairs else 0
            far = false_accepts / len(negative_pairs) if negative_pairs else 0
            
            results[threshold] = {
                'FAR': far,
                'FRR': frr,
                'positive_sims': positive_sims,
                'negative_sims': negative_sims,
                'false_accepts': false_accepts,
                'false_rejects': false_rejects
            }
            
            print(f"Threshold {threshold:.2f}:")
            print(f"  FRR (False Reject Rate): {frr*100:.2f}% ({false_rejects}/{len(positive_pairs)})")
            print(f"  FAR (False Accept Rate): {far*100:.2f}% ({false_accepts}/{len(negative_pairs)})")
            print(f"  Positive sims: {np.mean(positive_sims):.4f} ± {np.std(positive_sims):.4f}")
            print(f"  Negative sims: {np.mean(negative_sims):.4f} ± {np.std(negative_sims):.4f}")
            print()
        
        return results
    
    def plot_results(self, results: dict):
        """Plot FAR/FRR curves."""
        thresholds = sorted(results.keys())
        fars = [results[t]['FAR'] * 100 for t in thresholds]
        frrs = [results[t]['FRR'] * 100 for t in thresholds]
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: FAR/FRR curves
        plt.subplot(1, 2, 1)
        plt.plot(thresholds, fars, 'r-o', label='FAR (False Accept)', linewidth=2)
        plt.plot(thresholds, frrs, 'b-o', label='FRR (False Reject)', linewidth=2)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Error Rate (%)', fontsize=12)
        plt.title('FAR/FRR vs Threshold', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Find EER (where FAR ≈ FRR)
        eer_idx = np.argmin([abs(f - r) for f, r in zip(fars, frrs)])
        eer_threshold = thresholds[eer_idx]
        eer_rate = (fars[eer_idx] + frrs[eer_idx]) / 2
        
        plt.axvline(eer_threshold, color='g', linestyle='--', alpha=0.5, 
                   label=f'EER: {eer_rate:.2f}% @ {eer_threshold:.2f}')
        plt.axvline(0.23, color='orange', linestyle='--', alpha=0.5,
                   label='Training optimal (0.23)')
        plt.axvline(0.30, color='purple', linestyle='--', alpha=0.5,
                   label='Production recommended (0.30)')
        plt.legend(fontsize=9)
        
        # Plot 2: Similarity distributions
        plt.subplot(1, 2, 2)
        
        # Get all similarities from one threshold (e.g., 0.30)
        test_threshold = 0.30 if 0.30 in results else thresholds[len(thresholds)//2]
        positive_sims = results[test_threshold]['positive_sims']
        negative_sims = results[test_threshold]['negative_sims']
        
        plt.hist(positive_sims, bins=30, alpha=0.6, color='blue', 
                label=f'Same Person (n={len(positive_sims)})', density=True)
        plt.hist(negative_sims, bins=30, alpha=0.6, color='red',
                label=f'Different People (n={len(negative_sims)})', density=True)
        
        plt.axvline(0.23, color='orange', linestyle='--', linewidth=2,
                   label='Training (0.23)')
        plt.axvline(0.30, color='purple', linestyle='--', linewidth=2,
                   label='Production (0.30)')
        
        plt.xlabel('Cosine Similarity', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Similarity Score Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
        print("✅ Plot saved as 'threshold_analysis.png'")
        plt.show()


def main():
    """Main testing function."""
    print("="*60)
    print("MODEL A THRESHOLD TESTING FOR PRODUCTION")
    print("="*60)
    print()
    
    # Model path
    model_path = MODELS_DIR / "modelA_best.pth"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please ensure modelA_best.pth is in Veriface/backend/app/models/")
        return
    
    # Initialize tester
    tester = ThresholdTester(model_path)
    
    print("\n" + "="*60)
    print("INSTRUCTIONS FOR TESTING")
    print("="*60)
    print("""
To test thresholds effectively, you need:

1. POSITIVE PAIRS (Same Person):
   - Multiple photos of the same person
   - Different angles, lighting, expressions
   - Place in: Test/positive_pairs/person1/, person2/, etc.
   - At least 2 images per person, 5+ people recommended

2. NEGATIVE PAIRS (Different People):
   - Photos of different people
   - Place in: Test/negative_pairs/
   - Should include people who look similar (hard negatives)
   - At least 20+ different people recommended

Directory structure:
    Test/
        positive_pairs/
            person1/
                img1.jpg
                img2.jpg
            person2/
                img1.jpg
                img2.jpg
        negative_pairs/
            person_A.jpg
            person_B.jpg
            person_C.jpg
            ...
    """)
    
    # Check if test data exists
    test_dir = Path(__file__).parent / "Test"
    positive_dir = test_dir / "positive_pairs"
    negative_dir = test_dir / "negative_pairs"
    
    if not positive_dir.exists() or not negative_dir.exists():
        print("\n⚠️  Test directories not found!")
        print("Please create test data following the structure above.")
        
        # Create directories
        positive_dir.mkdir(parents=True, exist_ok=True)
        negative_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✅ Created directories:")
        print(f"   - {positive_dir}")
        print(f"   - {negative_dir}")
        print("\nPlease add test images and run again.")
        return
    
    # Load positive pairs
    print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)
    
    positive_pairs = []
    person_dirs = [d for d in positive_dir.iterdir() if d.is_dir()]
    
    if not person_dirs:
        print("⚠️  No person directories found in positive_pairs/")
        print("Please add folders with multiple images per person.")
        return
    
    print(f"\nFound {len(person_dirs)} people in positive_pairs/")
    
    for person_dir in tqdm(person_dirs, desc="Processing positive pairs"):
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        
        if len(images) < 2:
            print(f"⚠️  {person_dir.name} has only {len(images)} image(s), need at least 2")
            continue
        
        # Extract embeddings
        try:
            embeddings = [tester.extract_embedding(img) for img in images]
            
            # Create all pairs
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    positive_pairs.append((embeddings[i], embeddings[j]))
        except Exception as e:
            print(f"⚠️  Error processing {person_dir.name}: {e}")
            continue
    
    # Load negative pairs
    negative_images = list(negative_dir.glob("*.jpg")) + list(negative_dir.glob("*.png"))
    
    if len(negative_images) < 2:
        print(f"⚠️  Only {len(negative_images)} images in negative_pairs/, need at least 2")
        return
    
    print(f"Found {len(negative_images)} images in negative_pairs/")
    
    negative_pairs = []
    try:
        embeddings = [tester.extract_embedding(img) 
                     for img in tqdm(negative_images, desc="Processing negative pairs")]
        
        # Create all pairs
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                negative_pairs.append((embeddings[i], embeddings[j]))
    except Exception as e:
        print(f"⚠️  Error processing negative pairs: {e}")
        return
    
    if not positive_pairs or not negative_pairs:
        print("\n❌ Not enough pairs generated!")
        return
    
    # Test thresholds
    thresholds = [0.20, 0.23, 0.25, 0.27, 0.30, 0.32, 0.35, 0.38, 0.40]
    
    results = tester.test_threshold_range(positive_pairs, negative_pairs, thresholds)
    
    # Plot results
    tester.plot_results(results)
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    fars = [results[t]['FAR'] for t in thresholds]
    frrs = [results[t]['FRR'] for t in thresholds]
    
    # Find best threshold (minimize total error)
    total_errors = [f + r for f, r in zip(fars, frrs)]
    best_idx = np.argmin(total_errors)
    best_threshold = thresholds[best_idx]
    
    print(f"""
Based on your test data:

1. Training Optimal (0.23):
   - FAR: {results[0.23]['FAR']*100:.2f}%
   - FRR: {results[0.23]['FRR']*100:.2f}%
   - Total Error: {(results[0.23]['FAR'] + results[0.23]['FRR'])*100:.2f}%

2. Production Recommended (0.30):
   - FAR: {results[0.30]['FAR']*100:.2f}%
   - FRR: {results[0.30]['FRR']*100:.2f}%
   - Total Error: {(results[0.30]['FAR'] + results[0.30]['FRR'])*100:.2f}%

3. Optimal for Your Data ({best_threshold:.2f}):
   - FAR: {results[best_threshold]['FAR']*100:.2f}%
   - FRR: {results[best_threshold]['FRR']*100:.2f}%
   - Total Error: {(results[best_threshold]['FAR'] + results[best_threshold]['FRR'])*100:.2f}%

SUGGESTIONS:
- For HIGH SECURITY (minimize false accepts): Use 0.35-0.40
- For BALANCED (minimize total error): Use {best_threshold:.2f}
- For USER CONVENIENCE (minimize false rejects): Use 0.25-0.27

Current setting in thresholds.yaml: 0.30 (Production recommended)
    """)


if __name__ == "__main__":
    main()
