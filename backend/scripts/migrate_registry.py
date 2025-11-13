"""
Migrate registry from legacy format to new format with auto-generated user IDs.

This script:
1. Reads current registry.json
2. Detects if keys are names (legacy) or user IDs (SWS00001)
3. Migrates to new format: {SWS00001: {name, embeddings}}
4. Backs up old registry before migration
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import REGISTRY_PATH, STORE_DIR

def is_legacy_format(data: dict) -> bool:
    """Check if registry uses legacy format (name as key instead of user_id)"""
    if not data:
        return False
    
    # Check first key - if it starts with "SWS" and has 8 chars, it's new format
    first_key = list(data.keys())[0]
    if first_key.startswith("SWS") and len(first_key) == 8:
        try:
            int(first_key[3:])  # Try to parse number part
            return False  # New format
        except ValueError:
            pass
    
    return True  # Legacy format (name as key)

def migrate_registry():
    """Migrate registry to new format"""
    if not REGISTRY_PATH.exists():
        print("❌ No registry.json found")
        return
    
    # Load current registry
    with open(REGISTRY_PATH, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("✅ Registry is empty, no migration needed")
        return
    
    # Check if migration needed
    if not is_legacy_format(data):
        print("✅ Registry already in new format (user IDs: SWS00001, SWS00002...)")
        print(f"   Users: {len(data)}")
        for user_id, user_data in list(data.items())[:3]:
            name = user_data.get("name", user_id) if isinstance(user_data, dict) else user_id
            emb_count = len(user_data.get("embeddings", [])) if isinstance(user_data, dict) else len(user_data)
            print(f"   - {user_id}: {name} ({emb_count} embeddings)")
        return
    
    print("🔄 Migrating registry from legacy format...")
    print(f"   Found {len(data)} users")
    
    # Backup old registry
    backup_path = STORE_DIR / f"registry_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Backed up to: {backup_path}")
    
    # Migrate to new format
    migrated = {}
    user_counter = 1
    
    for old_key, user_data in data.items():
        # Generate new user_id
        new_user_id = f"SWS{user_counter:05d}"
        
        # Handle both formats
        if isinstance(user_data, dict) and "embeddings" in user_data:
            # Already has structure, just need new key
            name = user_data.get("name", old_key)
            embeddings = user_data["embeddings"]
        elif isinstance(user_data, list):
            # Direct embeddings list (legacy)
            name = old_key
            embeddings = user_data
        else:
            print(f"⚠️  Skipping invalid entry: {old_key}")
            continue
        
        migrated[new_user_id] = {
            "name": name,
            "embeddings": embeddings
        }
        
        print(f"   {old_key} → {new_user_id}: {name} ({len(embeddings)} embeddings)")
        user_counter += 1
    
    # Save migrated registry
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(migrated, f, indent=2)
    
    print(f"\n✅ Migration complete!")
    print(f"   Migrated {len(migrated)} users")
    print(f"   New format: SWS00001, SWS00002, ...")
    print(f"\n📝 Registry structure:")
    print(f"   {{")
    print(f"     \"SWS00001\": {{")
    print(f"       \"name\": \"User Name\",")
    print(f"       \"embeddings\": [[...], [...]]")
    print(f"     }}")
    print(f"   }}")

if __name__ == "__main__":
    print("="*70)
    print("REGISTRY MIGRATION SCRIPT")
    print("="*70)
    print()
    
    migrate_registry()
    
    print()
    print("="*70)
    print("Done!")
    print("="*70)
