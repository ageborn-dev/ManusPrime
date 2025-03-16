# utils/cache.py
import os
import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from functools import wraps

logger = logging.getLogger("manusprime.utils.cache")

class Cache:
    """Simple file-based cache system."""
    
    def __init__(self, cache_dir: str = "cache", max_age: int = 3600):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_age: Maximum age of cache entries in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.max_age = max_age
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value or None if not found or expired
        """
        cache_file = self._get_cache_file(key)
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
                
            # Check if cache entry is expired
            if time.time() - cache_data["timestamp"] > self.max_age:
                # Remove expired entry
                os.unlink(cache_file)
                return None
                
            return cache_data["value"]
            
        except Exception as e:
            logger.warning(f"Error reading cache for key '{key}': {e}")
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            bool: True if successful, False otherwise
        """
        cache_file = self._get_cache_file(key)
        
        try:
            cache_data = {
                "timestamp": time.time(),
                "value": value
            }
            
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
                
            return True
            
        except Exception as e:
            logger.warning(f"Error writing cache for key '{key}': {e}")
            return False
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful, False otherwise
        """
        cache_file = self._get_cache_file(key)
        
        if not cache_file.exists():
            return True
            
        try:
            os.unlink(cache_file)
            return True
        except Exception as e:
            logger.warning(f"Error invalidating cache for key '{key}': {e}")
            return False
    
    def _get_cache_file(self, key: str) -> Path:
        """Get the path to a cache file.
        
        Args:
            key: Cache key
            
        Returns:
            Path: Cache file path
        """
        # Create a hash of the key to use as filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"


# Cache decorator
def cached(cache: Optional[Cache] = None, max_age: int = 3600):
    """Decorator to cache function results.
    
    Args:
        cache: Cache instance to use
        max_age: Maximum age of cache entries in seconds
        
    Returns:
        Callable: Decorated function
    """
    # Create cache if not provided
    cache_instance = cache or Cache(max_age=max_age)
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            for arg in args:
                key_parts.append(str(arg))
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
            cache_key = "|".join(key_parts)
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache_instance.set(cache_key, result)
            
            return result
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            for arg in args:
                key_parts.append(str(arg))
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
            cache_key = "|".join(key_parts)
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache_instance.set(cache_key, result)
            
            return result
        
        # Choose appropriate wrapper based on function type
        if asyncio.__name__ in func.__module__ or "_is_coroutine" in dir(func):
            return async_wrapper
        return sync_wrapper
        
    return decorator

# Create global cache instance
cache = Cache()