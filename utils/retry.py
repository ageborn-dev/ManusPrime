import asyncio
import logging
from typing import Any, Callable, Optional, Union
from functools import wraps
import random

logger = logging.getLogger("manusprime.utils.retry")

class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        """Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add ±25% jitter
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)
            
        return max(0, delay)

async def async_retry(
    func: Callable,
    config: Optional[RetryConfig] = None,
    exceptions: tuple = (Exception,),
    *args,
    **kwargs
) -> Any:
    """Retry an async function with exponential backoff.
    
    Args:
        func: The async function to retry
        config: Retry configuration
        exceptions: Tuple of exceptions to retry on
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Any: Result of the function call
        
    Raises:
        Exception: The last exception if all retries fail
    """
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            result = await func(*args, **kwargs)
            if attempt > 1:
                logger.info(f"Function {func.__name__} succeeded on attempt {attempt}")
            return result
            
        except exceptions as e:
            last_exception = e
            
            if attempt == config.max_attempts:
                logger.error(f"Function {func.__name__} failed after {config.max_attempts} attempts: {e}")
                break
            
            delay = config.get_delay(attempt)
            logger.warning(f"Function {func.__name__} failed on attempt {attempt}, retrying in {delay:.2f}s: {e}")
            await asyncio.sleep(delay)
    
    raise last_exception

def retry_on_failure(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for adding retry logic to async functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exceptions: Tuple of exceptions to retry on
        
    Returns:
        Callable: Decorated function with retry logic
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay
            )
            return await async_retry(func, config, exceptions, *args, **kwargs)
        return wrapper
    return decorator

class ProviderError(Exception):
    """Base exception for provider-related errors."""
    pass

class RateLimitError(ProviderError):
    """Exception for rate limit errors."""
    pass

class AuthenticationError(ProviderError):
    """Exception for authentication errors."""
    pass

class ServiceUnavailableError(ProviderError):
    """Exception for service unavailable errors."""
    pass

def get_provider_exceptions():
    """Get tuple of exceptions that should trigger retries for providers."""
    return (
        RateLimitError,
        ServiceUnavailableError,
        ConnectionError,
        TimeoutError,
        OSError,  # Network errors
    )
