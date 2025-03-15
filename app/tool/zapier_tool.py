from typing import Any, Dict, Optional
import json
from pydantic import BaseModel

from app.tool.base import BaseTool
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
    
    name = "zapier"
    description = "Execute actions through Zapier integrations"
    parameters_schema = ZapierToolParameters
    
    def __init__(self):
        super().__init__()
        self.enabled = config.zapier.enabled
        self.allowed_actions = config.zapier.allowed_actions

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate tool parameters."""
        try:
            params = ZapierToolParameters(**parameters)
            
            # Check if Zapier integration is enabled
            if not self.enabled:
                raise ValueError("Zapier integration is not enabled")
            
            # Validate action is allowed
            if params.action not in self.allowed_actions:
                raise ValueError(f"Action '{params.action}' is not allowed")
                
            return True
            
        except Exception as e:
            logger.error(f"Zapier tool parameter validation failed: {str(e)}")
            return False

    async def _execute(
        self,
        parameters: Dict[str, Any],
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the Zapier tool."""
        try:
            params = ZapierToolParameters(**parameters)
            
            # Add task_id to data if provided
            if task_id:
                params.data["task_id"] = task_id
            
            # Trigger Zapier webhook
            result = await zapier_client.trigger_webhook(
                webhook_url=params.webhook_url,
                data={
                    "action": params.action,
                    "data": params.data
                }
            )
            
            if result is None:
                raise Exception("Zapier webhook execution failed")
                
            return {
                "status": "success",
                "action": params.action,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Zapier tool execution failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def get_tool_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "webhook_url": {
                        "type": "string",
                        "description": "The Zapier webhook URL to trigger"
                    },
                    "action": {
                        "type": "string",
                        "description": "The action to execute",
                        "enum": self.allowed_actions
                    },
                    "data": {
                        "type": "object",
                        "description": "Data payload for the action"
                    }
                },
                "required": ["webhook_url", "action", "data"]
            }
        }

    @property
    def example_usage(self) -> str:
        """Get example usage of the tool."""
        return """
Example usage:
```python
# Create a task via Zapier
result = await zapier_tool.execute({
    "webhook_url": "https://hooks.zapier.com/...",
    "action": "task_create",
    "data": {
        "title": "New Task",
        "description": "Task created via Zapier integration",
        "priority": 1
    }
})
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
