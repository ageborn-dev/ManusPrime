# core/agent.py
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from copy import deepcopy
import uuid
from core.cache_manager import LRUCache
from core.batch_processor import BatchProcessor, BatchTask
from core.fallback_chain import FallbackChain, ModelConfig
from utils.performance import connection_manager

from plugins.base import Plugin, PluginCategory
from plugins.registry import registry
from config import config

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
        self.default_provider = default_provider or config.get_value("providers.default")
        self.budget_limit = budget_limit or config.get_value("budget.limit", 0.0)
        self.vector_memory = None
        self.input_validator = None
        
        # Initialize optimizations
        self.cache = LRUCache(
            capacity=config.get_value("cache.max_entries", 1000),
            ttl=config.get_value("cache.ttl", 3600)
        )
        
        self.batch_processor = BatchProcessor(
            max_batch_size=config.get_value("batch.max_batch_size", 10),
            max_concurrent=config.get_value("batch.max_concurrent", 3),
            cost_threshold=config.get_value("batch.cost_threshold", 1.0)
        )
        
        # Configure fallback chain
        fallback_models = []
        for model_name in config.get_value("fallback.chains.default", []):
            provider = self._get_provider_for_model(model_name)
            if provider:
                fallback_models.append(ModelConfig(
                    name=model_name,
                    provider=provider,
                    timeout=config.get_value("fallback.timeout_base", 10.0),
                    max_retries=config.get_value("fallback.max_total_retries", 5),
                    cost_per_token=config.get_value(f"costs.{model_name}", 0.0),
                    warm_up=True
                ))
        
        self.fallback_chain = FallbackChain(
            models=fallback_models,
            max_total_retries=config.get_value("fallback.max_total_retries", 5),
            cost_threshold=config.get_value("fallback.cost_threshold", 1.0)
        )
        
        # Track usage
        self.total_tokens = 0
        self.total_cost = 0.0
        self.model_usage = {}
        
        # Task analysis patterns
        self.task_patterns = {
            'code': ['code', 'program', 'function', 'script', 'python', 'javascript', 'java', 'cpp'],
            'planning': ['plan', 'organize', 'strategy', 'workflow', 'complex', 'steps'],
            'tool_use': ['search', 'browse', 'file', 'execute', 'run', 'automate', 'zapier'],
            'creative': ['story', 'poem', 'creative', 'write', 'essay', 'blog'],
            'crawl': ['crawl', 'scrape', 'extract', 'website', 'webpage', 'web content']
        }
        
        # Task-specific models
        self.task_models = {
            'code': config.get_value('task_models.code_generation', 'codestral-latest'),
            'planning': config.get_value('task_models.planning', 'claude-3.7-sonnet'),
            'tool_use': config.get_value('task_models.tool_use', 'claude-3.7-sonnet'),
            'creative': config.get_value('task_models.creative', 'claude-3.7-sonnet'),
            'crawl': config.get_value('task_models.crawler', 'gpt-4o'),
            'default': config.get_value('task_models.default', 'mistral-small-latest')
        }
        
        # Plugins
        self.initialized = False
    
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
            
            # Initialize vector memory if available
            self.vector_memory = registry.get_plugin("vector_memory")
            if self.vector_memory:
                await self.vector_memory.initialize()
            
            # Initialize input validator if available
            self.input_validator = registry.get_plugin("input_validator")
            if self.input_validator:
                await self.input_validator.initialize()
                logger.info("Input validator plugin initialized")
            else:
                logger.info("Input validator plugin not found")
            
            # Activate required plugins based on configuration
            for category, plugin_name in config.active_plugins.items():
                if not plugin_name:
                    continue
                    
                plugin_config = config.get_plugin_config(plugin_name)
                await registry.activate_plugin(plugin_name, plugin_config)
            
            # Activate default provider if not already activated
            provider_plugin = registry.get_active_plugin(PluginCategory.PROVIDER)
            if not provider_plugin:
                provider_config = config.get_provider_config(self.default_provider)
                provider_plugins = registry.get_plugin_classes_by_category(PluginCategory.PROVIDER)
                
                for plugin_class in provider_plugins:
                    if plugin_class.name == self.default_provider:
                        await registry.activate_plugin(self.default_provider, provider_config)
                        break
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing ManusPrime agent: {e}")
            return False
    
    def analyze_task(self, task: str) -> str:
        """Analyze a task to determine the best model to use.
        
        Args:
            task: The task description
            
        Returns:
            str: The task category ('code', 'planning', 'tool_use', 'creative', or 'default')
        """
        task_lower = task.lower()
        
        for category, patterns in self.task_patterns.items():
            for pattern in patterns:
                if pattern in task_lower:
                    return category
        
        return 'default'
    
    def select_model(self, task: str) -> str:
        """Select the appropriate model for a given task.
        
        Args:
            task: The task description
            
        Returns:
            str: The selected model name
        """
        category = self.analyze_task(task)
        return self.task_models.get(category, self.task_models['default'])
    
    def _get_provider_for_model(self, model_name: str) -> Optional[str]:
        """Get the provider for a given model name."""
        for provider, config in config.get_value("providers", {}).items():
            if isinstance(config, dict) and "models" in config:
                if model_name in config["models"]:
                    return provider
        return None

    async def execute_task(self, 
                          task: Union[str, BatchTask], 
                          batch_mode: bool = False,
                          **kwargs) -> Dict:
        """Execute a task using the appropriate plugins and models.
        
        Args:
            task: The task to execute
            batch_mode: Whether to use batch processing
            **kwargs: Additional execution parameters
            
        Returns:
            Dict: The execution result
        """
        try:
            # Initialize if needed
            if not self.initialized:
                await self.initialize()
            
            # Convert to BatchTask if string
            if isinstance(task, str):
                task = BatchTask(
                    id=str(uuid.uuid4()),
                    prompt=task,
                    model=kwargs.get("model")
                )
            
            # Use batch processing if enabled
            if batch_mode and config.get_value("batch.enabled", True):
                return await self._execute_batch_task(task, **kwargs)
            
            # Initialize result tracking
            start_time = asyncio.get_event_loop().time()
            result = {
                "success": False,
                "content": "",
                "tokens": 0,
                "cost": 0.0,
                "model": "",
                "execution_time": 0.0,
                "optimizations": {
                    "cache_hit": False,
                    "fallback_used": False,
                    "prompt_optimized": False,
                    "input_validated": False
                }
            }
            
            # Determine the task type for validation and optimization
            task_type = self.analyze_task(task.prompt)
            
            # Check if we should pre-select model based on task
            if not task.model:
                task.model = self.select_model(task.prompt)
                
            # Validate input if validator is available
            validated_prompt = task.prompt
            if self.input_validator:
                validation_result = await self.input_validator.execute(
                    input_text=task.prompt,
                    model=task.model,
                    context={"task_type": task_type}
                )
                
                if validation_result.get("success", False):
                    result["optimizations"]["input_validated"] = True
                    
                    # Check for critical violations
                    if not validation_result.get("is_valid", True):
                        violations = validation_result.get("violations", [])
                        logger.warning(f"Input validation failed: {violations}")
                        return {
                            "success": False,
                            "error": f"Input validation failed: {', '.join(violations)}",
                            "execution_time": asyncio.get_event_loop().time() - start_time,
                            "model": task.model,
                            "optimizations": result["optimizations"]
                        }
                    
                    # Use sanitized input
                    validated_prompt = validation_result["sanitized_input"]
                    
                    # Log any warnings
                    warnings = validation_result.get("warnings", [])
                    if warnings:
                        logger.info(f"Input warnings: {warnings}")
            
            # Check cache if enabled - use validated prompt for cache key
            if config.get_value("cache.enabled", True):
                cached = self.cache.get(validated_prompt, task.model or "default")
                if cached:
                    result.update(cached)
                    result["optimizations"]["cache_hit"] = True
                    result["execution_time"] = asyncio.get_event_loop().time() - start_time
                    return result
            
            # Get vector memory context
            similar_experiences = await self._get_similar_experiences(validated_prompt)
            enhanced_prompt = self._enhance_prompt_with_context(
                validated_prompt, 
                similar_experiences
            ) if similar_experiences else validated_prompt
            
            # Optimize request if enabled
            if config.get_value("performance.prompt_optimization", True):
                provider = self._get_provider_for_model(task.model) if task.model else self.default_provider
                try:
                    optimized = await connection_manager.optimize_request(
                        provider=provider,
                        prompt=enhanced_prompt,
                        **kwargs
                    )
                    enhanced_prompt = optimized["prompt"]
                    result["optimizations"]["prompt_optimized"] = True
                except Exception as e:
                    logger.warning(f"Prompt optimization failed: {e}")
            
            # Try execution with fallback
            if config.get_value("fallback.enabled", True):
                fallback_result = await self.fallback_chain.execute(
                    enhanced_prompt,
                    self,
                    task_type=task_type
                )
                if fallback_result["success"]:
                    result.update(fallback_result)
                    result["optimizations"]["fallback_used"] = True
                    
                    # Update cache
                    if config.get_value("cache.enabled", True):
                        self.cache.put(validated_prompt, result["model"], result)
                    
                    return result
            
            # Fall back to standard execution
            result["model"] = task.model
            
            # Get provider plugin
            provider = registry.get_active_plugin(PluginCategory.PROVIDER)
            if not provider:
                raise ValueError("No provider plugin active")
            
            # Prepare and execute
            tools = await self._prepare_tools(enhanced_prompt)
            response = await self._execute_with_provider(
                provider=provider,
                prompt=enhanced_prompt,
                tools=tools,
                model=task.model,
                **kwargs
            )
            
            # Update result with response
            result.update({
                "success": True,
                "content": response.get("content", ""),
                "tokens": response.get("usage", {}).get("total_tokens", 0),
                "cost": response.get("usage", {}).get("cost", 0.0),
                "execution_time": asyncio.get_event_loop().time() - start_time
            })
            
            # Update metrics
            self.total_tokens += result["tokens"]
            self.total_cost += result["cost"]
            self.model_usage[result["model"]] = self.model_usage.get(result["model"], 0) + 1
            
            # Store in cache - use validated prompt as key
            if config.get_value("cache.enabled", True):
                self.cache.put(validated_prompt, result["model"], deepcopy(result))
                
            # Store in vector memory - store original task for better retrieval
            await self._store_in_vector_memory(task.prompt, result)
            
            return result
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": asyncio.get_event_loop().time() - start_time,
                "model": task.model if isinstance(task, BatchTask) else kwargs.get("model", ""),
                "tokens": 0,
                "cost": 0.0,
                "optimizations": {
                    "cache_hit": False,
                    "fallback_used": False,
                    "prompt_optimized": False,
                    "input_validated": self.input_validator is not None
                }
            }
    
    async def _execute_batch_task(self, task: BatchTask, **kwargs) -> Dict:
        """Execute a task using batch processing."""
        # Add to batch processor
        task_id = await self.batch_processor.add_task(task)
        
        # Process queue if this is a high priority task
        if task.priority > 0:
            return (await self.batch_processor.process_queue(self))[0]
        
        # Otherwise just return queued status
        return {
            "success": True,
            "status": "queued",
            "task_id": task_id
        }

    async def _get_similar_experiences(self, prompt: str) -> List[Dict]:
        """Get similar past experiences from vector memory."""
        if not self.vector_memory:
            return []
            
        try:
            # Get task type for potential filtering
            task_type = self.analyze_task(prompt)
            
            search_result = await self.vector_memory.execute(
                operation="search",
                query=prompt,
                limit=3,
                filter_metadata={"task_type": task_type} if task_type != "default" else None
            )
            return search_result["results"] if search_result["success"] else []
        except Exception as e:
            logger.warning(f"Failed to retrieve similar experiences: {e}")
            return []

    async def _store_in_vector_memory(self, prompt: str, result: Dict):
        """Store successful result in vector memory."""
        if not self.vector_memory or not result["success"]:
            return
            
        try:
            task_type = self.analyze_task(prompt)
            
            await self.vector_memory.execute(
                operation="add",
                content=result["content"],
                metadata={
                    "task": prompt,
                    "task_type": task_type,
                    "model": result["model"],
                    "execution_time": result["execution_time"],
                    "tokens": result["tokens"],
                    "cost": result["cost"]
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store in vector memory: {e}")

    async def _prepare_tools(self, prompt: str) -> List[Dict]:
        """Prepare tool schemas based on prompt."""
        tools = []
        if any(word in prompt.lower() for word in ['tool', 'search', 'browse', 'crawl', 'scrape', 'website']):
            for category, plugin in registry.active_plugins.items():
                if category != PluginCategory.PROVIDER:
                    # Special handling for web crawler
                    if category == PluginCategory.WEB_CRAWLER:
                        tools.append({
                            "type": "function",
                            "function": {
                                "name": plugin.name,
                                "description": plugin.description,
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "url": {
                                            "type": "string",
                                            "description": "URL of the webpage to crawl"
                                        },
                                        "output_format": {
                                            "type": "string",
                                            "enum": ["html", "markdown", "json"],
                                            "default": "markdown",
                                            "description": "Output format for crawled content"
                                        },
                                        "schema": {
                                            "type": "object",
                                            "description": "Schema for structured data extraction",
                                            "optional": True
                                        }
                                    },
                                    "required": ["url"]
                                }
                            }
                        })
                    else:
                        tools.append({
                            "type": "function",
                            "function": {
                                "name": plugin.name,
                                "description": plugin.description,
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "args": {
                                            "type": "object",
                                            "description": f"Arguments for the {plugin.name} plugin"
                                        }
                                    },
                                    "required": ["args"]
                                }
                            }
                        })
        return tools

    async def _execute_with_provider(self,
                                   provider,
                                   prompt: str,
                                   tools: List[Dict],
                                   model: str,
                                   **kwargs) -> Dict:
        """Execute with provider and handle tool calls."""
        response = {}
        
        if tools:
            response = await provider.generate_with_tools(
                prompt=prompt,
                tools=tools,
                model=model,
                temperature=kwargs.get('temperature', 0.7),
                tool_choice=kwargs.get('tool_choice', 'auto')
            )
        else:
            response = await provider.generate(
                prompt=prompt,
                model=model,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', None)
            )
            
        content = response.get('content', '')
        tool_calls = response.get('tool_calls', [])
        
        if tool_calls:
            tool_results = []
            for tool_call in tool_calls:
                if tool_call.get('type') != 'function':
                    continue
                    
                function = tool_call.get('function', {})
                plugin_name = function.get('name')
                arguments = function.get('arguments', {})
                
                plugin = registry.get_plugin(plugin_name)
                if plugin:
                    try:
                        # Special handling for web crawler to avoid unnecessary LLM usage
                        if plugin_name == "crawl4ai":
                            # Only use LLM if explicitly requested or needed for extraction
                            use_llm = arguments.get("use_llm", False) or "extraction_prompt" in arguments
                            arguments["use_llm"] = use_llm
                            arguments["from_agent"] = use_llm
                        plugin_result = await plugin.execute(**arguments)
                        tool_results.append(f"Tool: {plugin_name}\nResult: {plugin_result}")
                    except Exception as e:
                        tool_results.append(f"Tool: {plugin_name}\nError: {str(e)}")
            
            if tool_results:
                tool_output = "\n\n".join(tool_results)
                follow_up = await provider.generate(
                    prompt=f"Task: {prompt}\n\nTool Results:\n{tool_output}\n\nBased on these results, please provide a complete answer.",
                    model=model,
                    temperature=kwargs.get('temperature', 0.7)
                )
                content = follow_up.get('content', '')
                
        response["content"] = content
        return response
    
    def _enhance_prompt_with_context(self, task: str, similar_experiences: List[Dict]) -> str:
        """Enhance a task prompt with context from similar past experiences.
        
        Args:
            task: The original task prompt
            similar_experiences: List of similar past experiences
            
        Returns:
            str: Enhanced prompt with context
        """
        if not similar_experiences:
            return task
            
        # Determine task type for better context structuring
        task_type = self.analyze_task(task)
        
        # Customize context format based on task type
        if task_type == "code":
            context = "Previous similar coding tasks and solutions:\n"
        elif task_type == "creative":
            context = "Previous similar creative tasks and samples:\n"
        else:
            context = "Previous similar tasks and solutions:\n"
            
        for exp in similar_experiences[:3]:  # Limit to top 3 experiences
            context += f"Task: {exp['metadata']['task']}\n"
            context += f"Solution: {exp['content']}\n\n"
        
        enhanced_prompt = f"{context}\nCurrent task: {task}\n"
        return enhanced_prompt
    
    async def cleanup(self):
        """Clean up resources used by the agent."""
        await registry.cleanup_all()
        self.initialized = False


# Helper function to quickly execute a task
async def execute_task(task: str, **kwargs) -> Dict:
    """Helper function to execute a task with ManusPrime.
    
    Args:
        task: The task to execute
        **kwargs: Additional execution parameters
        
    Returns:
        Dict: The execution result
    """
    agent = ManusPrime()
    await agent.initialize()
    result = await agent.execute_task(task, **kwargs)
    await agent.cleanup()
    return result
