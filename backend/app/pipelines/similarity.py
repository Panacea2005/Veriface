import numpy as np
from typing import Dict, Optional, Tuple
from app.core.config import THRESHOLDS_PATH
import yaml

class SimilarityMatcher:
    """Similarity computation and matching."""
    
    def __init__(self):
        with open(THRESHOLDS_PATH) as f:
            self.config = yaml.safe_load(f)
        self.cosine_threshold = self.config.get("similarity", {}).get("cosine", {}).get("threshold", 0.75)
        self.euclidean_threshold = self.config.get("similarity", {}).get("euclidean", {}).get("threshold", 5.0)
        self._warned_about_old_embeddings = False
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity."""
        # Ensure vectors are normalized
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))
    
    def euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute euclidean distance."""
        return float(np.linalg.norm(vec1 - vec2))
    
    def get_all_scores(self, query_vec: np.ndarray, registry: Dict[str, list], metric: str = "cosine") -> list:
        """
        Get similarity scores for all users in registry.
        Returns list of (user_id, best_score, embeddings_count) sorted by score (desc for cosine, asc for euclidean).
        """
        scores = []
        
        for user_id, vectors in registry.items():
            best_score = None
            for vec in vectors:
                vec_array = np.array(vec, dtype=np.float32)
                query_vec_normalized = query_vec.astype(np.float32)
                
                if len(query_vec_normalized) != len(vec_array):
                    print(f"[DEBUG] Warning: Dimension mismatch for user {user_id} - query: {len(query_vec_normalized)}, registry: {len(vec_array)}", file=__import__('sys').stderr)
                    continue
                
                # Normalize registry embeddings if needed (for compatibility with old embeddings)
                vec_norm = np.linalg.norm(vec_array)
                if vec_norm > 0.1 and vec_norm < 0.9:  # Likely not normalized
                    vec_array = vec_array / vec_norm
                
                if metric == "cosine":
                    score = self.cosine_similarity(query_vec_normalized, vec_array)
                    if best_score is None or score > best_score:
                        best_score = score
                else:  # euclidean
                    dist = self.euclidean_distance(query_vec_normalized, vec_array)
                    if best_score is None or dist < best_score:
                        best_score = dist
            
            if best_score is not None:
                scores.append((user_id, float(best_score), len(vectors)))
        
        # Sort: descending for cosine (higher is better), ascending for euclidean (lower is better)
        if metric == "cosine":
            scores.sort(key=lambda x: x[1], reverse=True)
        else:
            scores.sort(key=lambda x: x[1])
        
        return scores
    
    def match(self, query_vec: np.ndarray, registry: Dict[str, list], metric: str = "cosine") -> Optional[Tuple[str, float]]:
        """
        Find best match from registry.
        Returns (matched_user_id, score) or None if below threshold.
        """
        import sys
        best_user = None
        best_score = -1.0 if metric == "cosine" else float('inf')
        threshold = self.cosine_threshold if metric == "cosine" else self.euclidean_threshold
        
        print(f"[DEBUG] Matching with metric={metric}, threshold={threshold}", file=sys.stderr)
        print(f"[DEBUG] Registry has {len(registry)} users", file=sys.stderr)
        
        if not registry:
            print(f"[DEBUG] Registry is empty!", file=sys.stderr)
            return None
        
        for user_id, vectors in registry.items():
            print(f"[DEBUG] Checking user '{user_id}' with {len(vectors)} embeddings", file=sys.stderr)
            for idx, vec in enumerate(vectors):
                vec_array = np.array(vec, dtype=np.float32)
                query_vec_normalized = query_vec.astype(np.float32)
                
                # Ensure same length
                if len(query_vec_normalized) != len(vec_array):
                    print(f"[DEBUG] Warning: Dimension mismatch - query: {len(query_vec_normalized)}, registry: {len(vec_array)}", file=sys.stderr)
                    continue
                
                # Normalize registry embeddings if needed (for compatibility with old embeddings)
                vec_norm = np.linalg.norm(vec_array)
                if vec_norm > 0.1 and vec_norm < 0.9:  # Likely not normalized or from different model
                    vec_array = vec_array / vec_norm
                    if not self._warned_about_old_embeddings:
                        print(f"[WARNING] Registry embeddings appear to be from a different model (norm: {vec_norm:.4f}). Normalizing for compatibility, but match scores may be low. Consider re-registering with current model.", file=sys.stderr)
                        self._warned_about_old_embeddings = True
                    print(f"[DEBUG]   Normalized registry embedding {idx+1} (original norm: {vec_norm:.4f})", file=sys.stderr)
                
                if metric == "cosine":
                    score = self.cosine_similarity(query_vec_normalized, vec_array)
                    # Debug: Check if embeddings are identical
                    vec_diff = np.linalg.norm(query_vec_normalized - vec_array)
                    print(f"[DEBUG]   Embedding {idx+1} cosine score: {score:.4f}, L2 distance: {vec_diff:.6f}", file=sys.stderr)
                    if vec_diff < 1e-6:
                        print(f"[WARNING]   Embeddings are nearly identical (distance: {vec_diff:.6f}) - possible duplicate registration!", file=sys.stderr)
                    if score > best_score:
                        best_score = score
                        best_user = user_id
                else:  # euclidean
                    dist = self.euclidean_distance(query_vec_normalized, vec_array)
                    print(f"[DEBUG]   Embedding {idx+1} euclidean distance: {dist:.4f}", file=sys.stderr)
                    if dist < best_score:
                        best_score = dist
                        best_user = user_id
        
        print(f"[DEBUG] Best match: user='{best_user}', score={best_score:.4f}, threshold={threshold}", file=sys.stderr)
        
        # Check threshold
        if metric == "cosine":
            if best_score >= threshold:
                print(f"[DEBUG] Match PASSED (score {best_score:.4f} >= threshold {threshold})", file=sys.stderr)
                return (best_user, best_score)
            else:
                print(f"[DEBUG] Match FAILED (score {best_score:.4f} < threshold {threshold})", file=sys.stderr)
        else:  # euclidean
            if best_score <= threshold:
                print(f"[DEBUG] Match PASSED (distance {best_score:.4f} <= threshold {threshold})", file=sys.stderr)
                return (best_user, best_score)
            else:
                print(f"[DEBUG] Match FAILED (distance {best_score:.4f} > threshold {threshold})", file=sys.stderr)
        
        return None

