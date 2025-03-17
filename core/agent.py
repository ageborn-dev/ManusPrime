import logging
import asyncio
from typing import Dict, Optional, Union
import uuid

from config import config
from plugins.base import PluginCategory
from plugins.registry import registry
from core.batch_processor import BatchProcessor, BatchTask
from core.cache_manager import LRUCache
from core.memory_manager import MemoryManager
from core.tool_manager import ToolManager
from core.sandbox_manager import SandboxManager
from core.execution_handler import ExecutionHandler
from core.ai_planner import AIPlanner, AIPlannerException

logger = logging.getLogger("manusprime.agent")

class ManusPrime:
    """Multi-model AI agent that selects optimal models for different tasks."""
    
    def __init__(self, 
                 default_provider: Optional[str] = None, 
                 budget_limit: Optional[float] = None):
        """Initialize the ManusPrime agent.
        
        Args:
            default_provider: Name of the default provider to use (overrides config)
            budget_limit: Budget limit for token usage (overrides config)
        """
        logger.info("Initializing ManusPrime agent")
        try:
            # Basic configuration
            self.default_provider = default_provider or config.get_value("providers.default")
            logger.debug(f"Using default provider: {self.default_provider}")
            self.budget_limit = budget_limit or config.get_value("budget.limit", 0.0)
            logger.debug(f"Budget limit set to: {self.budget_limit}")
            
            # Initialize components
            self.cache = LRUCache(
                capacity=config.get_value("cache.max_entries", 1000),
                ttl=config.get_value("cache.ttl", 3600)
            )
            
            self.batch_processor = BatchProcessor(
                max_batch_size=config.get_value("batch.max_batch_size", 10),
                max_concurrent=config.get_value("batch.max_concurrent", 3),
                cost_threshold=config.get_value("batch.cost_threshold", 1.0)
            )
            
            # Initialize managers
            self.memory_manager = MemoryManager()
            self.tool_manager = ToolManager()
            self.sandbox_manager = SandboxManager()
            self.ai_planner = AIPlanner()
            
            # Initialize execution handler
            self.execution_handler = ExecutionHandler(
                memory_manager=self.memory_manager,
                tool_manager=self.tool_manager,
                sandbox_manager=self.sandbox_manager,
                ai_planner=self.ai_planner
            )
            
            # Track usage
            self.total_tokens = 0
            self.total_cost = 0.0
            self.model_usage = {}
            
            self.initialized = False
            logger.info("ManusPrime agent initialization completed")
            
        except Exception as e:
            logger.error(f"Error in ManusPrime.__init__: {e}")
            logger.error("Traceback:", exc_info=True)
            raise
    
    async def initialize(self) -> bool:
        """Initialize the agent and required plugins.
        
        Returns:
            bool: True if initialization was successful
        """
        if self.initialized:
            return True
            
        try:
            # Discover available plugins
            registry.discover_plugins()
            
            # Initialize vector memory
            self.memory_manager.vector_memory = registry.get_plugin("vector_memory")
            if self.memory_manager.vector_memory:
                await self.memory_manager.vector_memory.initialize()
            
            # Activate required plugins
            for category, plugin_name in config.active_plugins.items():
                if plugin_name:
                    plugin_config = config.get_plugin_config(plugin_name)
                    await registry.activate_plugin(plugin_name, plugin_config)
            
            # Activate default provider if needed
            provider = registry.get_active_plugin(PluginCategory.PROVIDER)
            if not provider:
                provider_config = config.get_provider_config(self.default_provider)
                await registry.activate_plugin(self.default_provider, provider_config)
            
            self.initialized = True
            logger.info("ManusPrime agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing ManusPrime agent: {e}")
            logger.error("Traceback:", exc_info=True)
            return False
    
    async def execute_task(self, 
                        task: Union[str, BatchTask], 
                        batch_mode: bool = False,
                        **kwargs) -> Dict:
        """Execute a task using AI planning and appropriate models.
        
        Args:
            task: The task to execute
            batch_mode: Whether to use batch processing
            **kwargs: Additional execution parameters
            
        Returns:
            Dict: The execution result
        """
        logger.info(f"Executing task: {task[:100] if isinstance(task, str) else task.prompt[:100]}...")
        
        try:
            # Initialize if needed
            if not self.initialized:
                await self.initialize()
            
            # Convert string task to BatchTask
            if isinstance(task, str):
                task_id = str(uuid.uuid4())
                task = BatchTask(
                    id=task_id,
                    prompt=task,
                    model=kwargs.get("model")
                )
            
            # Use batch processing if enabled
            if batch_mode and config.get_value("batch.enabled", True):
                return await self._execute_batch_task(task, **kwargs)
            
            # Get provider
            provider = registry.get_active_plugin(PluginCategory.PROVIDER)
            if not provider:
                raise ValueError("No provider plugin active")
            
            # Execute task with AI planning
            result = await self.execution_handler.execute(
                prompt=task.prompt,
                provider=provider,
                cache=self.cache,
                **kwargs
            )
            
            # Update metrics
            if result["success"]:
                self.total_tokens += result["tokens"]
                self.total_cost += result["cost"]
                self.model_usage[result["model"]] = self.model_usage.get(result["model"], 0) + 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            logger.error("Traceback:", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0.0,
                "model": task.model if isinstance(task, BatchTask) else None,
                "tokens": 0,
                "cost": 0.0
            }
    
    async def _execute_batch_task(self, task: BatchTask, **kwargs) -> Dict:
        """Execute a task using batch processing."""
        try:
            task_id = await self.batch_processor.add_task(task)
            
            if task.priority > 0:
                return (await self.batch_processor.process_queue(self))[0]
            
            return {
                "success": True,
                "status": "queued",
                "task_id": task_id
            }
        except Exception as e:
            logger.error(f"Error in batch task execution: {e}")
            logger.error("Traceback:", exc_info=True)
            raise
    
    async def cleanup(self):
        """Clean up resources used by the agent."""
        try:
            await registry.cleanup_all()
            self.initialized = False
            logger.info("ManusPrime agent cleanup completed")
        except Exception as e:
            logger.error(f"Error cleaning up ManusPrime agent: {e}")
            logger.error("Traceback:", exc_info=True)

# Helper function for quick task execution
async def execute_task(task: str, **kwargs) -> Dict:
    """Helper function to execute a task with ManusPrime.
    
    Args:
        task: The task to execute
        **kwargs: Additional execution parameters
        
    Returns:
        Dict: The execution result
    """
    try:
        agent = ManusPrime()
        await agent.initialize()
        result = await agent.execute_task(task, **kwargs)
        await agent.cleanup()
        return result
    except Exception as e:
        logger.error(f"Error in execute_task helper: {e}")
        logger.error("Traceback:", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "execution_time": 0.0
        }
