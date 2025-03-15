import hmac
import hashlib
import json
from typing import Any, Dict, Optional
from fastapi import HTTPException, Request
from pydantic import BaseModel

from app.config import config
from app.logger import logger

class WebhookPayload(BaseModel):
    action: str
    data: Dict[str, Any]

class ZapierWebhook:
    """Handler for Zapier webhook integrations."""
    
    def __init__(self):
        self.secret = config.zapier.webhook_secret
        self.api_key = config.zapier.api_key
        self.allowed_actions = config.zapier.allowed_actions

    async def validate_signature(self, request: Request) -> bool:
        """Validate the webhook signature."""
        if not self.secret:
            return True  # Skip validation if no secret configured
            
        signature = request.headers.get("X-Zapier-Signature")
        if not signature:
            raise HTTPException(status_code=401, detail="Missing signature")

        # Get raw body
        body = await request.body()
        
        # Calculate expected signature
        expected = hmac.new(
            self.secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()

        # Constant time comparison
        return hmac.compare_digest(signature, expected)

    def validate_api_key(self, api_key: Optional[str]) -> bool:
        """Validate the provided API key."""
        if not self.api_key:
            return True  # Skip validation if no API key configured
        return hmac.compare_digest(api_key or "", self.api_key)

    async def handle_webhook(self, request: Request) -> Dict[str, Any]:
        """Handle incoming webhook request."""
        try:
            # Validate signature
            if not await self.validate_signature(request):
                raise HTTPException(status_code=401, detail="Invalid signature")

            # Validate API key
            api_key = request.headers.get("X-Api-Key")
            if not self.validate_api_key(api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")

            # Parse payload
            body = await request.body()
            try:
                payload = WebhookPayload(**json.loads(body))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid payload: {str(e)}")

            # Validate action
            if payload.action not in self.allowed_actions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Action '{payload.action}' not allowed"
                )

            # Route to appropriate handler
            result = await self._route_action(payload)
            return {"status": "success", "result": result}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Webhook error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def _route_action(self, payload: WebhookPayload) -> Any:
        """Route webhook to appropriate action handler."""
        try:
            from .actions import ZapierActions
            
            handler = getattr(ZapierActions, payload.action, None)
            if not handler:
                raise ValueError(f"No handler for action: {payload.action}")
                
            return await handler(payload.data)
            
        except Exception as e:
            logger.error(f"Action routing error: {str(e)}")
            raise

webhook_handler = ZapierWebhook()
