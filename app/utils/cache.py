import functools
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.config import PROJECT_ROOT, config
from app.logger import logger


class ContentCache:
    """
    A simple content caching system for ManusPrime.
    
    This class provides functionality to cache results of API calls, web searches,
    and other expensive operations to improve performance and reduce costs.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        expiration_time: Optional[int] = None,
        max_cache_size: Optional[int] = None,
        enabled: Optional[bool] = None
    ):
        """
        Initialize the content cache.
        
        Args:
            cache_dir: Directory to store cache files (default from config)
            expiration_time: Time in seconds before cache entries expire (default from config)
            max_cache_size: Maximum number of cache entries to keep (default from config)
            enabled: Whether caching is enabled (default from config)
        """
        # Load settings from config if not provided
        cache_config = config.cache
        
        self.enabled = enabled if enabled is not None else cache_config.enable
        
        if not self.enabled:
            logger.info("Content cache is disabled")
            return
            
        # Use provided values or defaults from config
        cache_dir_value = cache_dir or cache_config.cache_dir
        self.cache_dir = Path(cache_dir_value) if not isinstance(cache_dir_value, Path) else cache_dir_value
        if not self.cache_dir.is_absolute():
            self.cache_dir = PROJECT_ROOT / self.cache_dir
            
        self.expiration_time = expiration_time or cache_config.expiration_time
        self.max_cache_size = max_cache_size or cache_config.max_cache_size
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for frequently accessed items
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Content cache initialized at {self.cache_dir}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if not self.enabled:
            return None
            
        # First check memory cache
        if key in self.memory_cache:
            cache_entry = self.memory_cache[key]
            # Check if entry is expired
            if time.time() - cache_entry.get("timestamp", 0) < self.expiration_time:
                return cache_entry.get("value")
            # Remove expired entry
            del self.memory_cache[key]
        
        # Then check file cache
        cache_file = self._get_cache_file(key)
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_entry = json.load(f)
                
            # Check if entry is expired
            if time.time() - cache_entry.get("timestamp", 0) >= self.expiration_time:
                # Remove expired cache file
                os.remove(cache_file)
                return None
                
            # Add to memory cache for faster access next time
            self.memory_cache[key] = cache_entry
            
            return cache_entry.get("value")
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.enabled:
            return
            
        # Create cache entry
        cache_entry = {
            "timestamp": time.time(),
            "value": value
        }
        
        # Store in memory cache
        self.memory_cache[key] = cache_entry
        
        # Store in file cache
        cache_file = self._get_cache_file(key)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_entry, f)
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")
        
        # Clean up if needed
        self._cleanup_if_needed()
    
    def invalidate(self, key: str) -> None:
        """
        Remove an item from the cache.
        
        Args:
            key: Cache key to invalidate
        """
        if not self.enabled:
            return
            
        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Remove from file cache
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            try:
                os.remove(cache_file)
            except Exception as e:
                logger.error(f"Error removing cache file: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        if not self.enabled:
            return
            
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear file cache
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                os.remove(cache_file)
                
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def _get_cache_file(self, key: str) -> Path:
        """Get the cache file path for a key."""
        # Create a hash of the key to use as filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _cleanup_if_needed(self) -> None:
        """Clean up old cache entries if we've exceeded max_cache_size."""
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            if len(cache_files) <= self.max_cache_size:
                return
                
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            
            # Remove oldest files to get back to max_cache_size
            for file in cache_files[:len(cache_files) - self.max_cache_size]:
                os.remove(file)
                
            # Also clean memory cache if it's too large
            if len(self.memory_cache) > self.max_cache_size:
                # Sort by timestamp (oldest first)
                sorted_keys = sorted(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k].get("timestamp", 0)
                )
                
                # Remove oldest entries
                for key in sorted_keys[:len(sorted_keys) - self.max_cache_size]:
                    del self.memory_cache[key]
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")


# Decorator for caching function results
def cached(expiration_time: Optional[int] = None, cache_instance: Optional[ContentCache] = None):
    """
    Decorator to cache function results.
    
    Args:
        expiration_time: Cache expiration time in seconds (default from config)
        cache_instance: Optional specific cache instance to use
        
    Returns:
        Decorator function
    """
    # Use provided expiration time or get from config
    actual_expiration = expiration_time or config.cache.expiration_time
    cache = cache_instance or ContentCache(expiration_time=actual_expiration)
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Skip caching if disabled
            if not cache.enabled:
                return await func(*args, **kwargs)
                
            # Create a cache key from function name and arguments
            key_parts = [func.__name__]
            
            # Add positional arguments
            for arg in args:
                key_parts.append(str(arg))
                
            # Add keyword arguments (sorted to ensure consistent keys)
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
                
            cache_key = "|".join(key_parts)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
                
            # Not in cache, call function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result)
            
            return result
            
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Skip caching if disabled
            if not cache.enabled:
                return func(*args, **kwargs)
                
            # Create a cache key from function name and arguments
            key_parts = [func.__name__]
            
            # Add positional arguments
            for arg in args:
                key_parts.append(str(arg))
                
            # Add keyword arguments (sorted to ensure consistent keys)
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
                
            cache_key = "|".join(key_parts)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
                
            # Not in cache, call function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result)
            
            return result
            
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
            
    return decorator
