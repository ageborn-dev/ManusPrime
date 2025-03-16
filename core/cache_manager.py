from typing import Dict, Any, Optional
import time
from collections import OrderedDict
import json
import os
import hashlib

class LRUCache:
    """LRU Cache for model responses with disk persistence."""
    
    def __init__(self, capacity: int = 1000, ttl: int = 3600):
        """Initialize LRU cache.
        
        Args:
            capacity: Maximum number of items to store
            ttl: Time to live in seconds (default 1 hour)
        """
        self.capacity = capacity
        self.ttl = ttl
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.cache_dir = "cache"
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Load persisted cache
        self._load_cache()
    
    def _generate_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model."""
        combined = f"{prompt}:{model}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _is_valid(self, key: str) -> bool:
        """Check if cache entry is still valid based on TTL."""
        if key not in self.timestamps:
            return False
        return (time.time() - self.timestamps[key]) < self.ttl
    
    def get(self, prompt: str, model: str) -> Optional[Dict]:
        """Get cached response if available and valid."""
        key = self._generate_key(prompt, model)
        
        if key in self.cache and self._is_valid(key):
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, prompt: str, model: str, response: Dict):
        """Add response to cache."""
        key = self._generate_key(prompt, model)
        
        # Add/update cache
        self.cache[key] = response
        self.timestamps[key] = time.time()
        self.cache.move_to_end(key)
        
        # Remove oldest if capacity exceeded
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
            
        # Persist to disk
        self._save_cache()
    
    def _save_cache(self):
        """Persist cache to disk."""
        cache_data = {
            "cache": list(self.cache.items()),
            "timestamps": self.timestamps
        }
        
        cache_file = os.path.join(self.cache_dir, "model_cache.json")
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
    
    def _load_cache(self):
        """Load cache from disk."""
        cache_file = os.path.join(self.cache_dir, "model_cache.json")
        if not os.path.exists(cache_file):
            return
            
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                
            # Restore cache and timestamps
            self.cache = OrderedDict(data["cache"])
            self.timestamps = data["timestamps"]
            
            # Remove expired entries
            current_time = time.time()
            expired = [
                key for key, timestamp in self.timestamps.items()
                if current_time - timestamp > self.ttl
            ]
            
            for key in expired:
                del self.cache[key]
                del self.timestamps[key]
                
        except Exception as e:
            print(f"Error loading cache: {e}")
