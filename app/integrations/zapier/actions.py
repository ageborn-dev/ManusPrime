from typing import Any, Dict
from pydantic import BaseModel

from app.database.task_manager import TaskManager
from app.tool.tool_collection import ToolCollection
from app.config import config
from app.logger import logger

class TaskCreateData(BaseModel):
    """Data structure for task creation."""
    title: str
    description: str
    priority: int = 1
    tags: list[str] = []

class TaskUpdateData(BaseModel):
    """Data structure for task updates."""
    task_id: str
    status: str
    progress: float = 0.0
    result: Dict[str, Any] = {}

class ToolExecuteData(BaseModel):
    """Data structure for tool execution."""
    tool_name: str
    parameters: Dict[str, Any]
    task_id: str = None

class ZapierActions:
    """Handlers for different Zapier actions."""
    
    task_manager = TaskManager()
    tool_collection = ToolCollection()

    @classmethod
    async def task_create(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new task."""
        try:
            task_data = TaskCreateData(**data)
            task = await cls.task_manager.create_task(
                title=task_data.title,
                description=task_data.description,
                priority=task_data.priority,
                tags=task_data.tags
            )
            return {
                "task_id": task.id,
                "status": task.status,
                "created_at": task.created_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to create task: {str(e)}")
            raise

    @classmethod
    async def task_update(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing task."""
        try:
            update_data = TaskUpdateData(**data)
            task = await cls.task_manager.update_task(
                task_id=update_data.task_id,
                updates={
                    "status": update_data.status,
                    "progress": update_data.progress,
                    "result": update_data.result
                }
            )
            return {
                "task_id": task.id,
                "status": task.status,
                "updated_at": task.updated_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to update task: {str(e)}")
            raise

    @classmethod
    async def tool_execute(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool via Zapier."""
        try:
            tool_data = ToolExecuteData(**data)
            
            # Get tool from collection
            tool = cls.tool_collection.get_tool(tool_data.tool_name)
            if not tool:
                raise ValueError(f"Tool not found: {tool_data.tool_name}")
            
            # Execute tool
            result = await tool.execute(
                parameters=tool_data.parameters,
                task_id=tool_data.task_id
            )
            
            return {
                "tool": tool_data.tool_name,
                "result": result,
                "task_id": tool_data.task_id
            }
        except Exception as e:
            logger.error(f"Failed to execute tool: {str(e)}")
            raise

    @classmethod
    async def get_task_status(cls, task_id: str) -> Dict[str, Any]:
        """Get task status (utility method)."""
        try:
            task = await cls.task_manager.get_task(task_id)
            return {
                "task_id": task.id,
                "status": task.status,
                "progress": task.progress,
                "result": task.result,
                "updated_at": task.updated_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get task status: {str(e)}")
            raise
