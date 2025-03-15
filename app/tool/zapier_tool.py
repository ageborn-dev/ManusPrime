from typing import Any, Dict, Optional
import json
from pydantic import BaseModel

from app.tool.base import BaseTool, ToolResult
from app.integrations.zapier import client as zapier_client
from app.config import config
from app.logger import logger

class ZapierToolParameters(BaseModel):
    """Parameters for Zapier tool execution."""
    webhook_url: str
    action: str
    data: Dict[str, Any]

class ZapierTool(BaseTool):
    """Tool for interacting with Zapier integrations."""
    
    name: str = "zapier"
    description: str = "Execute actions through Zapier integrations"
    parameters: dict = {
        "type": "object",
        "properties": {
            "webhook_url": {
                "type": "string",
                "description": "The Zapier webhook URL to trigger"
            },
            "action": {
                "type": "string",
                "description": "The action to execute",
                "enum": config.zapier.allowed_actions
            },
            "data": {
                "type": "object",
                "description": "Data payload for the action"
            }
        },
        "required": ["webhook_url", "action", "data"]
    }
    
    def __init__(self):
        super().__init__()
        self.enabled = config.zapier.enabled
        self.allowed_actions = config.zapier.allowed_actions

    async def execute(self, webhook_url: str, action: str, data: Dict[str, Any], **kwargs) -> str:
        """
        Execute a Zapier action by triggering a webhook.
        
        Args:
            webhook_url: The Zapier webhook URL to trigger
            action: The action to execute
            data: Data payload for the action
            
        Returns:
            Result message from the Zapier webhook
        """
        try:
            # Validate the action is allowed
            if action not in self.allowed_actions:
                return ToolResult(error=f"Action '{action}' is not allowed. Allowed actions: {', '.join(self.allowed_actions)}")
            
            # Check if Zapier integration is enabled
            if not self.enabled:
                return ToolResult(error="Zapier integration is not enabled in configuration")
            
            # Prepare the payload
            payload = {
                "action": action,
                "data": data
            }
            
            # Add task_id if provided
            if "task_id" in kwargs:
                payload["data"]["task_id"] = kwargs["task_id"]
            
            # Trigger the webhook
            logger.info(f"Triggering Zapier webhook for action '{action}'")
            result = await zapier_client.trigger_webhook(webhook_url, payload)
            
            if result is None:
                return ToolResult(error="Zapier webhook failed to return a response")
            
            # Return the result as a string
            if isinstance(result, dict):
                return ToolResult(output=f"Zapier action '{action}' completed successfully: {json.dumps(result, indent=2)}")
            else:
                return ToolResult(output=f"Zapier action '{action}' completed successfully: {result}")
                
        except Exception as e:
            logger.error(f"Error executing Zapier action: {str(e)}")
            return ToolResult(error=f"Error executing Zapier action: {str(e)}")
    
    @property
    def example_usage(self) -> str:
        """Get example usage of the tool."""
        return """
Example usage:
```
# Create a task via Zapier
result = await zapier_tool.execute(
    webhook_url="https://hooks.zapier.com/hooks/catch/123456/abcdef/",
    action="task_create",
    data={
        "title": "New Task",
        "description": "Task created via Zapier integration",
        "priority": 1
    }
)
```
"""

    @property
    def help_text(self) -> str:
        """Get help text for the tool."""
        actions = ", ".join(self.allowed_actions)
        return f"""
Zapier Integration Tool

This tool allows interaction with Zapier through webhooks. It can be used to:
- Trigger Zapier automations
- Create and update tasks
- Execute custom actions

Allowed actions: {actions}

Each action requires specific data parameters. Consult the Zapier integration
documentation for details on required parameters for each action type.
"""