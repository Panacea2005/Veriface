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
        return self.registry.get(user_id, [])
    
    def remove_user(self, user_id: str):
        """Remove user from registry."""
        if user_id in self.registry:
            del self.registry[user_id]
            self._save()

