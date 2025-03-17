import logging
import asyncio
from typing import Dict, Optional, Union, Any
import uuid

from config import config
from plugins.base import PluginCategory
from core.plugin_manager import plugin_manager
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
            # Initialize vector memory
            self.memory_manager.vector_memory = await plugin_manager.get_plugin("vector_memory")
            if self.memory_manager.vector_memory:
                await self.memory_manager.vector_memory.initialize()
            
            # Activate required plugins
            for category, plugin_name in config.active_plugins.items():
                if plugin_name:
                    await plugin_manager.get_plugin(plugin_name)
            
            # Activate default provider if needed
            provider = await plugin_manager.get_active_plugin(PluginCategory.PROVIDER)
            if not provider:
                await plugin_manager.get_plugin(self.default_provider)
            
            self.initialized = True
            logger.info("ManusPrime agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing ManusPrime agent: {e}")
            logger.error("Traceback:", exc_info=True)
            return False
    
    async def _analyze_task(self, prompt: str, provider: Any) -> Dict[str, Any]:
        """Analyze task using AI to determine type and requirements.
        
        Args:
            prompt: The task prompt
            provider: The provider plugin instance
            
        Returns:
            Dict[str, Any]: Analysis results containing task type, capabilities needed,
                           suggested models, and required plugins
        """
        logger.debug(f"Analyzing task: {prompt[:100]}...")
        
        try:
            # Get best reasoning model for analysis
            analysis_model = await self.ai_planner.get_analysis_model(provider)
            if not analysis_model:
                raise ValueError("No suitable model available for task analysis")
            
            # Create analysis prompt
            analysis_prompt = f"""
Analyze this task and provide a structured assessment. Task: "{prompt}"

Consider:
1. Core task type (e.g. coding, creative, research, analysis)
2. Required capabilities (e.g. code generation, text analysis, math)
3. Complexity level (simple, moderate, complex)
4. Needed plugins or tools
5. Execution pattern (sequential, parallel)

Respond in JSON format:
{{
    "task_type": "primary task category",
    "capabilities": ["capability1", "capability2"],
    "complexity": "complexity level",
    "plugins": ["plugin1", "plugin2"],
    "execution_pattern": "sequential or parallel",
    "additional_context": "any important notes"
}}
"""
            # Get analysis from model
            response = await provider.generate(
                prompt=analysis_prompt,
                model=analysis_model,
                temperature=0.7,
                response_format={"type": "json"}
            )
            
            try:
                analysis = response.get("content", "{}")
                if isinstance(analysis, str):
                    import json
                    analysis = json.loads(analysis)
                
                logger.debug(f"Task analysis completed: {analysis}")
                return analysis
                
            except Exception as e:
                logger.error(f"Failed to parse task analysis: {e}")
                return {
                    "task_type": "default",
                    "capabilities": [],
                    "complexity": "moderate",
                    "plugins": [],
                    "execution_pattern": "sequential"
                }
                
        except Exception as e:
            logger.error(f"Error in task analysis: {e}")
            return {
                "task_type": "default",
                "capabilities": [],
                "complexity": "moderate",
                "plugins": [],
                "execution_pattern": "sequential"
            }
    
    async def execute_task(self, 
                        task: Union[str, BatchTask], 
                        batch_mode: bool = False,
                        task_type: Optional[str] = None,
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
            provider = await plugin_manager.get_active_plugin(PluginCategory.PROVIDER)
            if not provider:
                raise ValueError("No provider plugin active")
            
            # Analyze task and determine execution strategy
            prompt = task.prompt if isinstance(task, BatchTask) else task
            task_analysis = await self._analyze_task(prompt, provider)
            
            # Execute task with AI planning using analysis results
            result = await self.execution_handler.execute(
                prompt=task.prompt if isinstance(task, BatchTask) else task,
                provider=provider,
                cache=self.cache,
                task_type=task_type,
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
            await plugin_manager.cleanup()
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
