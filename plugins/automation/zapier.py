# plugins/automation/zapier.py
import logging
import json
import aiohttp
import hmac
import hashlib
from typing import Dict, ClassVar, Optional, Any

from plugins.base import Plugin, PluginCategory

logger = logging.getLogger("manusprime.plugins.zapier")

class ZapierPlugin(Plugin):
    """Plugin for Zapier integration for automation workflows."""
    
    name: ClassVar[str] = "zapier"
    description: ClassVar[str] = "Connects to Zapier to automate workflows across applications"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.AUTOMATION
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Zapier plugin.
        
        Args:
            config: Plugin configuration
        """
        super().__init__(config)
        self.api_key = self.config.get("api_key", "")
        self.webhook_secret = self.config.get("webhook_secret", "")
        self.allowed_actions = self.config.get("allowed_actions", [])
        self.timeout = self.config.get("timeout", 30)
        self.max_retries = self.config.get("max_retries", 3)
        self.session = None
    
    async def initialize(self) -> bool:
        """Initialize the Zapier plugin.
        
        Returns:
            bool: True if initialization was successful
        """
        if not self.api_key:
            logger.error("Zapier API key not provided")
            return False
            
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        logger.info("Zapier plugin initialized successfully")
        return True
    
    async def execute(self, 
                   webhook_url: str,
                   action: str, 
                   data: Dict[str, Any],
                   **kwargs) -> Dict[str, Any]:
        """Execute a Zapier action by triggering a webhook.
        
        Args:
            webhook_url: The webhook URL to trigger
            action: The action to execute
            data: The data to send to the webhook
            **kwargs: Additional parameters
            
        Returns:
            Dict: Webhook response or error message
        """
        if not self.session:
            return {
                "success": False,
                "error": "Zapier plugin not initialized"
            }
            
        # Validate the action
        if self.allowed_actions and action not in self.allowed_actions:
            return {
                "success": False,
                "error": f"Action '{action}' is not allowed. Allowed actions: {', '.join(self.allowed_actions)}"
            }
            
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key
        }
        
        # Prepare the payload
        payload = {
            "action": action,
            "data": data
        }
        
        # Add optional task_id if provided
        if "task_id" in kwargs:
            payload["task_id"] = kwargs["task_id"]
            
        # Calculate signature if webhook secret provided
        if self.webhook_secret:
            payload_bytes = json.dumps(payload).encode('utf-8')
            signature = hmac.new(
                self.webhook_secret.encode(),
                payload_bytes,
                hashlib.sha256
            ).hexdigest()
            headers["X-Zapier-Signature"] = signature
        
        # Send the request with retries
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                async with self.session.post(webhook_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        try:
                            result = await response.json()
                            return {
                                "success": True,
                                "result": result
                            }
                        except:
                            # If not JSON, try to get text
                            text = await response.text()
                            return {
                                "success": True,
                                "result": text
                            }
                            
                    # Handle retries for 5xx errors
                    if response.status >= 500 and retry_count < self.max_retries:
                        retry_count += 1
                        logger.warning(f"Zapier webhook returned {response.status}, retrying ({retry_count}/{self.max_retries})...")
                        continue
                        
                    # Return error for client errors or after max retries
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"Zapier webhook failed with status {response.status}: {error_text}"
                    }
                    
            except aiohttp.ClientError as e:
                if retry_count < self.max_retries:
                    retry_count += 1
                    logger.warning(f"Connection error with Zapier webhook, retrying ({retry_count}/{self.max_retries}): {str(e)}")
                    continue
                    
                return {
                    "success": False,
                    "error": f"Connection error with Zapier webhook: {str(e)}"
                }
                
        # This should not be reached, but just in case
        return {
            "success": False,
            "error": "Max retries exceeded for Zapier webhook"
        }
    
    async def cleanup(self) -> None:
        """Clean up resources used by the Zapier plugin."""
        if self.session:
            await self.session.close()
            self.session = None