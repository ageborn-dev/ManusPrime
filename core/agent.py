import logging
import asyncio
from typing import Dict, Optional, Any, List, Tuple
import uuid

from config import config
from plugins.base import PluginCategory
from core.plugin_manager import plugin_manager
from utils.cache import LRUCache
from core.memory_manager import MemoryManager
from core.tool_manager import ToolManager
from core.sandbox_manager import SandboxManager
from core.execution_handler import ExecutionHandler
from core.ai_planner import AIPlanner, AIPlannerException

logger = logging.getLogger("manusprime.agent")

class ManusPrime:
    """ManusPrime is a Multi-model AI agent that selects optimal models for different tasks."""
    
    def __init__(self, 
                 default_provider: Optional[str] = None, 
                 budget_limit: Optional[float] = None):
        """Initialize the ManusPrime agent."""
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
        """Initialize the agent and required plugins."""
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
            
            # Ensure default provider is activated
            default_provider = await plugin_manager.get_plugin(self.default_provider)
            if not default_provider:
                raise ValueError(f"Default provider {self.default_provider} could not be activated")
            
            self.initialized = True
            logger.info("ManusPrime agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing ManusPrime agent: {e}")
            logger.error("Traceback:", exc_info=True)
            return False

    async def _setup_providers(self) -> Tuple[Any, Dict[str, List[str]]]:
        """Set up default provider and get available models.
        
        Returns:
            Tuple[Any, Dict[str, List[str]]]: Default provider and available models
        """
        # Get default provider
        default_provider = await plugin_manager.get_plugin(self.default_provider)
        if not default_provider:
            raise ValueError("Default provider not available")
            
        # Get all available models from providers with valid API keys
        available_models = {}
        for p_name, p_config in config.get_value("providers", {}).items():
            if p_name == "default":
                continue
            
            try:
                provider = await plugin_manager.get_plugin(p_name)
                if provider and await provider.has_valid_api_key():
                    models = await provider.get_available_models()
                    if models:  # Only add if provider has available models
                        available_models[p_name] = models
                        logger.debug(f"Provider {p_name} has {len(models)} models available")
            except Exception as e:
                logger.warning(f"Error getting models from provider {p_name}: {e}")
        
        if not available_models:
            logger.warning("No providers with available models found. Using default provider only.")
            # Add default provider's models
            if await default_provider.has_valid_api_key():
                models = await default_provider.get_available_models()
                if models:
                    available_models[self.default_provider] = models
        
        return default_provider, available_models

    async def execute_task(self, task: str, **kwargs) -> Dict:
        """Execute a task using the default provider for orchestration."""
        logger.info(f"Executing task: {task[:100]}...")
        
        try:
            # Initialize if needed
            if not self.initialized:
                await self.initialize()
            
            # Set up providers and get available models
            default_provider, available_models = await self._setup_providers()
            
            # Check cache first
            cached_result = self.cache.get(task, self.default_provider)
            if cached_result:
                logger.debug("Returning cached result")
                return cached_result
            
            # Create execution plan using default provider
            plan = await self.ai_planner.create_execution_plan(
                task=task,
                provider=default_provider,
                available_models=available_models,
                cache=self.cache
            )
            
            # Execute task with plan
            result = await self.execution_handler.execute(
                prompt=task,
                provider=default_provider,
                cache=self.cache,
                execution_plan=plan,
                available_models=available_models,
                **kwargs
            )
            
            # Update metrics
            if result.get("success", False):
                self.total_tokens += result.get("tokens", 0)
                self.total_cost += result.get("cost", 0.0)
                model = result.get("model", "unknown")
                self.model_usage[model] = self.model_usage.get(model, 0) + 1
                
                # Cache successful result
                self.cache.put(task, result, self.default_provider)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            logger.error("Traceback:", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0.0,
                "tokens": 0,
                "cost": 0.0
            }

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
    """Helper function to execute a task with ManusPrime."""
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
