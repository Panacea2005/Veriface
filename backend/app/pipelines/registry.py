import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from app.core.config import REGISTRY_PATH

class FaceRegistry:
    """Face embedding registry (JSON store).
    
    Structure: {
        "user_id": {
            "name": str,
            "embeddings": [[float]]
        }
    }
    """
    
    def __init__(self):
        self.path = REGISTRY_PATH
        self.registry: Dict[str, dict] = self._load()
    
    def _load(self) -> Dict[str, dict]:
        """Load registry from JSON."""
        if self.path.exists():
            with open(self.path, 'r') as f:
                data = json.load(f)
                # Handle legacy format (direct embeddings) and migrate
                if data and isinstance(list(data.values())[0], list):
                    # Legacy format: {user_id: [[embeddings]]}
                    # Migrate to new format: {user_id: {name, embeddings}}
                    migrated = {}
                    for user_id, embeddings in data.items():
                        migrated[user_id] = {
                            "name": user_id,  # Use user_id as name for legacy data
                            "embeddings": embeddings
                        }
                    # Save migrated data
                    with open(self.path, 'w') as f:
                        json.dump(migrated, f, indent=2)
                    return migrated
                return data
        return {}
    
    def _save(self):
        """Save registry to JSON."""
        with open(self.path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _generate_user_id(self) -> str:
        """Generate next user ID in format SWS00001, SWS00002, etc."""
        self.registry = self._load()
        if not self.registry:
            return "SWS00001"
        
        # Extract numeric parts from existing IDs
        max_num = 0
        for uid in self.registry.keys():
            if uid.startswith("SWS") and len(uid) == 8:
                try:
                    num = int(uid[3:])
                    max_num = max(max_num, num)
                except ValueError:
                    continue
        
        next_num = max_num + 1
        return f"SWS{next_num:05d}"
    
    def add_user(self, name: str, embedding: np.ndarray, user_id: Optional[str] = None) -> str:
        """Add or update user embedding. Returns user_id.
        
        Args:
            name: User's name
            embedding: Face embedding vector
            user_id: Optional user_id. If None, auto-generate new ID. If exists, add embedding to existing user.
        
        Returns:
            user_id (generated or provided)
        """
        # Always reload from file first to ensure we have latest state
        self.registry = self._load()
        vec = embedding.tolist()
        
        # If user_id provided and exists, add embedding to existing user
        if user_id and user_id in self.registry:
            self.registry[user_id]["embeddings"].append(vec)
            self._save()
            return user_id
        
        # If user_id provided but doesn't exist, create new user with that ID
        if user_id and user_id not in self.registry:
            self.registry[user_id] = {
                "name": name,
                "embeddings": [vec]
            }
            self._save()
            return user_id
        
        # No user_id provided: generate new ID
        new_user_id = self._generate_user_id()
        self.registry[new_user_id] = {
            "name": name,
            "embeddings": [vec]
        }
        self._save()
        return new_user_id
    
    def get_all(self) -> Dict[str, dict]:
        """Get all user data. Reloads from file to get latest data.
        
        Returns:
            {user_id: {name, embeddings}}
        """
        self.registry = self._load()  # Reload to get latest registrations
        return self.registry
    
    def get_all_embeddings(self) -> Dict[str, List[List[float]]]:
        """Get embeddings only (legacy format for compatibility).
        
        Returns:
            {user_id: [[embeddings]]}
        """
        self.registry = self._load()
        return {uid: data["embeddings"] for uid, data in self.registry.items()}
    
    def get_user_name(self, user_id: str) -> Optional[str]:
        """Get user's name."""
        self.registry = self._load()
        user_data = self.registry.get(user_id)
        return user_data["name"] if user_data else None
    
    def get_user_vectors(self, user_id: str) -> List[List[float]]:
        """Get all embedding vectors for a user."""
        # Always reload from file first to ensure we have latest state
        self.registry = self._load()
        user_data = self.registry.get(user_id)
        return user_data["embeddings"] if user_data else []
    
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
        if user_id in self.registry:
            self.registry[user_id]["embeddings"] = embeddings
            self._save()
    
    def update_user_name(self, user_id: str, name: str) -> bool:
        """Update user's name. Returns True if successful."""
        self.registry = self._load()
        if user_id in self.registry:
            self.registry[user_id]["name"] = name
            self._save()
            return True
        return False
    
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

