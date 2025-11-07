import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from app.core.config import REGISTRY_PATH

class FaceRegistry:
    """Face embedding registry (JSON store)."""
    
    def __init__(self):
        self.path = REGISTRY_PATH
        self.registry: Dict[str, List[List[float]]] = self._load()
    
    def _load(self) -> Dict[str, List[List[float]]]:
        """Load registry from JSON."""
        if self.path.exists():
            with open(self.path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save(self):
        """Save registry to JSON."""
        with open(self.path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def add_user(self, user_id: str, embedding: np.ndarray):
        """Add or update user embedding."""
        # Always reload from file first to ensure we have latest state
        self.registry = self._load()
        vec = embedding.tolist()
        if user_id not in self.registry:
            self.registry[user_id] = []
        self.registry[user_id].append(vec)
        self._save()
    
    def get_all(self) -> Dict[str, List[List[float]]]:
        """Get all embeddings. Reloads from file to get latest data."""
        self.registry = self._load()  # Reload to get latest registrations
        return self.registry
    
    def get_user_vectors(self, user_id: str) -> List[List[float]]:
        """Get all vectors for a user."""
        # Always reload from file first to ensure we have latest state
        self.registry = self._load()
        return self.registry.get(user_id, [])
    
    def remove_user(self, user_id: str) -> bool:
        """Remove user from registry. Returns True if removed."""
        # Always reload from file first to ensure we have latest state
        self.registry = self._load()
        # normalize id
        target = user_id.strip()
        if target in self.registry:
            del self.registry[target]
            self._save()
            return True
        # fallback: case-insensitive match
        lower_map = {k.lower(): k for k in self.registry.keys()}
        key = lower_map.get(target.lower())
        if key is not None:
            del self.registry[key]
            self._save()
            return True
        return False

    def remove_embedding(self, user_id: str, index: int) -> bool:
        """Remove a single embedding by index for a user. Returns True if removed."""
        # Always reload from file first to ensure we have latest state
        self.registry = self._load()
        if user_id not in self.registry:
            return False
        vectors = self.registry[user_id]
        if index < 0 or index >= len(vectors):
            return False
        del vectors[index]
        if len(vectors) == 0:
            # Remove user if no embeddings left
            del self.registry[user_id]
        self._save()
        return True

    def replace_user_embeddings(self, user_id: str, embeddings: List[List[float]]):
        """Replace all embeddings for a given user with provided list."""
        # Always reload from file first to ensure we have latest state
        self.registry = self._load()
        self.registry[user_id] = embeddings
        self._save()
    
    def clear_all(self) -> int:
        """Clear all users and embeddings from registry. Returns number of users deleted."""
        # Reload first to get accurate count
        self.registry = self._load()
        count = len(self.registry)
        self.registry = {}
        self._save()
        
        # Also delete backup file if it exists
        backup_path = self.path.parent / f"{self.path.name}.backup"
        if backup_path.exists():
            try:
                backup_path.unlink()
            except Exception:
                pass  # Ignore errors deleting backup
        
        return count

