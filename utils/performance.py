import asyncio
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
import aiohttp

logger = logging.getLogger("manusprime.performance")

@dataclass
class ConnectionPool:
    """Manages connection pooling for API endpoints."""
    max_connections: int
    idle_timeout: float
    retry_limit: int
    base_url: str
    session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self):
        """Initialize connection pool."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                ttl_dns_cache=300,
                keepalive_timeout=self.idle_timeout
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            
    async def close(self):
        """Close connection pool."""
        if self.session:
            await self.session.close()
            self.session = None

class RequestThrottler:
    """Rate limiting for API requests."""
    
    def __init__(self, 
                 requests_per_minute: int,
                 burst_limit: int):
        """Initialize request throttler.
        
        Args:
            requests_per_minute: Maximum requests per minute
            burst_limit: Maximum burst requests allowed
        """
        self.rpm = requests_per_minute
        self.burst_limit = burst_limit
        self.request_times: List[datetime] = []
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire permission to make request."""
        async with self.lock:
            now = datetime.now()
            
            # Remove old requests
            cutoff = now - timedelta(minutes=1)
            self.request_times = [t for t in self.request_times if t > cutoff]
            
            # Check limits
            if len(self.request_times) >= self.rpm:
                # Check burst limit
                if len(self.request_times) >= self.burst_limit:
                    # Calculate delay needed
                    oldest = min(self.request_times)
                    delay = (oldest + timedelta(minutes=1)) - now
                    if delay.total_seconds() > 0:
                        await asyncio.sleep(delay.total_seconds())
                        
            # Add new request
            self.request_times.append(now)

class PromptOptimizer:
    """Optimizes prompts for better performance."""
    
    def __init__(self):
        self.compression_patterns = {
            r"\b(the|a|an)\b\s+": " ",  # Remove articles
            r"\s+": " ",  # Collapse whitespace
            r"([.!?])\s+": r"\1 ",  # Normalize sentence spacing
            r"\s*([,;:])\s*": r"\1 "  # Normalize punctuation spacing
        }
        
        self.cached_compressions: Dict[str, str] = {}
        self.cache_file = "prompt_compressions.json"
        self._load_cache()
        
    def _load_cache(self):
        """Load cached compressions from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cached_compressions = json.load(f)
            except Exception as e:
                logger.error(f"Error loading compression cache: {e}")
                
    def _save_cache(self):
        """Save cached compressions to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cached_compressions, f)
        except Exception as e:
            logger.error(f"Error saving compression cache: {e}")
            
    def compress(self, prompt: str) -> str:
        """Compress a prompt for efficiency.
        
        Args:
            prompt: The prompt to compress
            
        Returns:
            str: Compressed prompt
        """
        # Check cache
        if prompt in self.cached_compressions:
            return self.cached_compressions[prompt]
            
        # Apply compression patterns
        compressed = prompt
        for pattern, replacement in self.compression_patterns.items():
            compressed = pattern.sub(replacement, compressed)
            
        # Cache result
        self.cached_compressions[prompt] = compressed
        self._save_cache()
        
        return compressed

class ConnectionManager:
    """Manages API connections and performance optimization."""
    
    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
        self.throttlers: Dict[str, RequestThrottler] = {}
        self.prompt_optimizer = PromptOptimizer()
        
        # Track active requests
        self.active_requests: Set[str] = set()
        self.request_times: Dict[str, List[datetime]] = {}
        
    async def initialize(self):
        """Initialize connection manager."""
        # Default pools for common providers
        default_pools = {
            "openai": ConnectionPool(
                max_connections=50,
                idle_timeout=300.0,
                retry_limit=3,
                base_url="https://api.openai.com"
            ),
            "anthropic": ConnectionPool(
                max_connections=30,
                idle_timeout=300.0,
                retry_limit=3,
                base_url="https://api.anthropic.com"
            ),
            "mistral": ConnectionPool(
                max_connections=20,
                idle_timeout=300.0,
                retry_limit=3,
                base_url="https://api.mistral.ai"
            )
        }
        
        # Initialize pools
        for name, pool in default_pools.items():
            self.pools[name] = pool
            await pool.initialize()
            
        # Set up throttlers
        self.throttlers = {
            "openai": RequestThrottler(requests_per_minute=3500, burst_limit=3750),
            "anthropic": RequestThrottler(requests_per_minute=300, burst_limit=350),
            "mistral": RequestThrottler(requests_per_minute=150, burst_limit=175)
        }
        
    async def close(self):
        """Close all connection pools."""
        for pool in self.pools.values():
            await pool.close()
            
    def get_pool(self, provider: str) -> Optional[ConnectionPool]:
        """Get connection pool for provider."""
        return self.pools.get(provider)
        
    def get_throttler(self, provider: str) -> Optional[RequestThrottler]:
        """Get request throttler for provider."""
        return self.throttlers.get(provider)
        
    async def optimize_request(self, 
                             provider: str,
                             prompt: str,
                             **kwargs) -> Dict:
        """Optimize and prepare a request.
        
        Args:
            provider: The provider to use
            prompt: The prompt to optimize
            **kwargs: Additional request parameters
            
        Returns:
            Dict: Optimized request parameters
        """
        # Get connection pool
        pool = self.get_pool(provider)
        if not pool:
            raise ValueError(f"No connection pool for provider: {provider}")
            
        # Get throttler
        throttler = self.get_throttler(provider)
        if throttler:
            await throttler.acquire()
            
        # Optimize prompt
        optimized_prompt = self.prompt_optimizer.compress(prompt)
        
        # Track request
        request_id = f"{provider}:{datetime.now().isoformat()}"
        self.active_requests.add(request_id)
        
        if provider not in self.request_times:
            self.request_times[provider] = []
        self.request_times[provider].append(datetime.now())
        
        # Return optimized parameters
        return {
            "session": pool.session,
            "prompt": optimized_prompt,
            "request_id": request_id,
            **kwargs
        }
        
    def complete_request(self, request_id: str):
        """Mark request as completed."""
        if request_id in self.active_requests:
            self.active_requests.remove(request_id)
            
    def get_metrics(self) -> Dict:
        """Get current performance metrics.
        
        Returns:
            Dict: Performance metrics
        """
        now = datetime.now()
        metrics = {
            "active_requests": len(self.active_requests),
            "requests_per_provider": {}
        }
        
        # Calculate requests per minute per provider
        for provider, times in self.request_times.items():
            # Get requests in last minute
            cutoff = now - timedelta(minutes=1)
            recent = [t for t in times if t > cutoff]
            
            metrics["requests_per_provider"][provider] = {
                "last_minute": len(recent),
                "total": len(times)
            }
            
        return metrics

# Global instance
connection_manager = ConnectionManager()
