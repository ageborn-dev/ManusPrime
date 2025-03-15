from typing import Dict, Any, Optional
import aiohttp
from app.config import config
from app.logger import logger
from .webhook import webhook_handler
from .actions import ZapierActions

class ZapierClient:
    """Client for making outbound requests to Zapier."""
    
    def __init__(self):
        self.api_key = config.zapier.api_key
        self.timeout = aiohttp.ClientTimeout(total=config.zapier.timeout)
        self.max_retries = config.zapier.max_retries
        self.session = None

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def _close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def trigger_webhook(
        self,
        webhook_url: str,
        data: Dict[str, Any],
        retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Trigger a Zapier webhook with data."""
        try:
            await self._ensure_session()
            
            headers = {
                "Content-Type": "application/json",
                "X-Api-Key": self.api_key
            }
            
            async with self.session.post(webhook_url, json=data, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                    
                # Handle retries for 5xx errors
                if response.status >= 500 and retry_count < self.max_retries:
                    logger.warning(f"Zapier webhook failed with {response.status}, retrying...")
                    return await self.trigger_webhook(webhook_url, data, retry_count + 1)
                    
                # Log error response
                error_text = await response.text()
                logger.error(f"Zapier webhook failed: {error_text}")
                return None
                
        except Exception as e:
            logger.error(f"Error triggering Zapier webhook: {str(e)}")
            
            # Retry on connection errors
            if retry_count < self.max_retries:
                logger.warning("Retrying webhook due to connection error...")
                return await self.trigger_webhook(webhook_url, data, retry_count + 1)
            
            return None
            
        finally:
            await self._close_session()

# Create global instances
client = ZapierClient()

__all__ = [
    "webhook_handler",
    "ZapierActions",
    "client"
]
