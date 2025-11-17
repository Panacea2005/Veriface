import numpy as np
from typing import Dict, Optional, Tuple
from app.core.config import THRESHOLDS_PATH
import yaml

class SimilarityMatcher:
    """Similarity computation and matching with advanced multi-embedding aggregation."""
    
    def __init__(self):
        import os
        with open(THRESHOLDS_PATH) as f:
            self.config = yaml.safe_load(f)
        
        # Auto-select threshold based on model type (PyTorch or DeepFace)
        use_deepface = os.environ.get("DEEPFACE_ONLY", "0") == "1"
        cosine_config = self.config.get("similarity", {}).get("cosine", {})
        
        if use_deepface:
            # Use DeepFace ArcFace threshold (standard: 0.68)
            self.cosine_threshold = cosine_config.get("threshold_deepface", 0.68)
            print(f"[INFO] Using DeepFace ArcFace threshold: {self.cosine_threshold}", file=__import__('sys').stderr)
        else:
            # Use PyTorch trained model threshold (validated: 0.4)
            self.cosine_threshold = cosine_config.get("threshold_pytorch", cosine_config.get("threshold", 0.4))
            print(f"[INFO] Using PyTorch trained model threshold: {self.cosine_threshold}", file=__import__('sys').stderr)
        
        self.euclidean_threshold = self.config.get("similarity", {}).get("euclidean", {}).get("threshold", 5.0)
        
        # Aggregation method: "max", "mean", "median", "top_k", "delta_margin", "hybrid"
        self.method = self.config.get("similarity", {}).get("cosine", {}).get("method", "top_k")
        self.top_k = self.config.get("similarity", {}).get("cosine", {}).get("k", 3)
        self.vote_threshold = self.config.get("similarity", {}).get("cosine", {}).get("vote_threshold", 0.60)
        self.margin_threshold = self.config.get("similarity", {}).get("cosine", {}).get("margin_threshold", 0.15)
        
        self._warned_about_old_embeddings = False
        
        print(f"[INFO] SimilarityMatcher initialized: method={self.method}, k={self.top_k}, threshold={self.cosine_threshold}", file=__import__('sys').stderr)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity."""
        # Ensure vectors are normalized
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))
    
    def euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute euclidean distance."""
        return float(np.linalg.norm(vec1 - vec2))
    
    def aggregate_scores(self, scores: list, method: str = None) -> float:
        """
        Aggregate multiple similarity scores using specified method.
        
        Args:
            scores: List of similarity scores
            method: "max", "mean", "median", "top_k"
        
        Returns:
            Aggregated score
        """
        if not scores:
            return 0.0
        
        if method is None:
            method = self.method
        
        if method == "max":
            return max(scores)
        elif method == "mean":
            return float(np.mean(scores))
        elif method == "median":
            return float(np.median(scores))
        elif method == "top_k":
            # Sort descending and take top-k
            sorted_scores = sorted(scores, reverse=True)
            top_k_scores = sorted_scores[:min(self.top_k, len(scores))]
            return float(np.mean(top_k_scores))
        elif method == "delta_margin":
            # Delta-Margin: Best for sibling discrimination
            # Requires clear winner (best_score - second_best_score > margin)
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) >= 2:
                best = sorted_scores[0]
                second_best = sorted_scores[1]
                margin = best - second_best
                
                # If margin is too small, faces are too similar (siblings)
                # Return average of top-2 to be conservative
                if margin < self.margin_threshold:
                    # Ambiguous match - return lower score
                    return float(np.mean([best, second_best]))
                else:
                    # Clear winner
                    return best
            else:
                return float(np.mean(sorted_scores))
        elif method == "hybrid":
            # Hybrid: Max score + Voting (both conditions must pass)
            # Always return the actual max score, but validation happens elsewhere
            max_score = max(scores)
            
            # Store voting metadata for logging/debugging
            passing_count = sum(1 for s in scores if s >= self.cosine_threshold)
            vote_ratio = passing_count / len(scores)
            
            # Return max_score regardless - threshold check happens in match() method
            # This ensures frontend always shows actual similarity score
            return max_score
        else:
            # Default to top_k
            return self.aggregate_scores(scores, "top_k")
    
    def get_all_scores(self, query_vec: np.ndarray, registry: Dict[str, dict], metric: str = "cosine") -> list:
        """
        Get similarity scores for all users in registry using aggregation method.
        
        Args:
            query_vec: Query embedding vector
            registry: {user_id: {name, embeddings}} or {user_id: [[embeddings]]} (legacy)
            metric: "cosine" or "euclidean"
        
        Returns:
            list of (user_id, aggregated_score, embeddings_count) sorted by score.
        """
        scores = []
        
        for user_id, user_data in registry.items():
            # Handle both new format {name, embeddings} and legacy format [[embeddings]]
            if isinstance(user_data, dict) and "embeddings" in user_data:
                vectors = user_data["embeddings"]
            elif isinstance(user_data, list):
                vectors = user_data  # Legacy format
            else:
                continue
            
            user_scores = []
            
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
                    user_scores.append(score)
                else:  # euclidean
                    dist = self.euclidean_distance(query_vec_normalized, vec_array)
                    user_scores.append(dist)
            
            if user_scores:
                # Aggregate scores based on method
                if metric == "cosine":
                    aggregated_score = self.aggregate_scores(user_scores, self.method)
                else:
                    # For euclidean, use minimum distance (best match)
                    aggregated_score = min(user_scores)
                
                scores.append((user_id, float(aggregated_score), len(vectors)))
        
        # Sort: descending for cosine (higher is better), ascending for euclidean (lower is better)
        if metric == "cosine":
            scores.sort(key=lambda x: x[1], reverse=True)
        else:
            scores.sort(key=lambda x: x[1])
        
        return scores
    
    def match(self, query_vec: np.ndarray, registry: Dict[str, dict], metric: str = "cosine") -> Optional[Tuple[str, float]]:
        """
        Find best match from registry using aggregation method.
        
        Args:
            query_vec: Query embedding vector
            registry: {user_id: {name, embeddings}} or {user_id: [[embeddings]]} (legacy)
            metric: "cosine" or "euclidean"
        
        Returns:
            (matched_user_id, aggregated_score) or None if below threshold.
        """
        import sys
        best_user = None
        best_score = -1.0 if metric == "cosine" else float('inf')
        threshold = self.cosine_threshold if metric == "cosine" else self.euclidean_threshold
        
        print(f"[DEBUG] Matching with metric={metric}, method={self.method}, threshold={threshold}", file=sys.stderr)
        print(f"[DEBUG] Registry has {len(registry)} users", file=sys.stderr)
        
        if not registry:
            print(f"[DEBUG] Registry is empty!", file=sys.stderr)
            return None
        
        for user_id, user_data in registry.items():
            # Handle both new format {name, embeddings} and legacy format [[embeddings]]
            if isinstance(user_data, dict) and "embeddings" in user_data:
                vectors = user_data["embeddings"]
            elif isinstance(user_data, list):
                vectors = user_data  # Legacy format
            else:
                continue
            
            print(f"[DEBUG] Checking user '{user_id}' with {len(vectors)} embeddings", file=sys.stderr)
            
            user_scores = []
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
                    user_scores.append(score)
                    # Debug: Check if embeddings are identical
                    vec_diff = np.linalg.norm(query_vec_normalized - vec_array)
                    print(f"[DEBUG]   Embedding {idx+1} cosine score: {score:.4f}, L2 distance: {vec_diff:.6f}", file=sys.stderr)
                    if vec_diff < 1e-6:
                        print(f"[WARNING]   Embeddings are nearly identical (distance: {vec_diff:.6f}) - possible duplicate registration!", file=sys.stderr)
                else:  # euclidean
                    dist = self.euclidean_distance(query_vec_normalized, vec_array)
                    user_scores.append(dist)
                    print(f"[DEBUG]   Embedding {idx+1} euclidean distance: {dist:.4f}", file=sys.stderr)
            
            # Aggregate scores for this user
            if user_scores:
                if metric == "cosine":
                    aggregated_score = self.aggregate_scores(user_scores, self.method)
                    print(f"[DEBUG]   Raw scores: {[f'{s:.4f}' for s in user_scores]}", file=sys.stderr)
                    print(f"[DEBUG]   Aggregated score ({self.method}): {aggregated_score:.4f}", file=sys.stderr)
                    
                    # Hybrid method: check voting
                    if self.method == "hybrid":
                        votes = sum(1 for s in user_scores if s >= threshold)
                        vote_confidence = votes / len(user_scores)
                        print(f"[DEBUG]   Voting: {votes}/{len(user_scores)} ({vote_confidence*100:.1f}%) passed threshold", file=sys.stderr)
                        
                        # Only accept if both score AND voting pass
                        if vote_confidence >= self.vote_threshold and aggregated_score >= threshold:
                            if aggregated_score > best_score:
                                best_score = aggregated_score
                                best_user = user_id
                    else:
                        # Normal aggregation
                        if aggregated_score > best_score:
                            best_score = aggregated_score
                            best_user = user_id
                else:  # euclidean
                    aggregated_score = min(user_scores)  # Best distance
                    print(f"[DEBUG]   Aggregated distance (min): {aggregated_score:.4f}", file=sys.stderr)
                    if aggregated_score < best_score:
                        best_score = aggregated_score
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

