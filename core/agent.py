# core/agent.py
import asyncio
import logging
import traceback
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
        logger.info("Initializing ManusPrime agent")
        try:
            self.default_provider = default_provider or config.get_value("providers.default")
            logger.debug(f"Using default provider: {self.default_provider}")
            self.budget_limit = budget_limit or config.get_value("budget.limit", 0.0)
            logger.debug(f"Budget limit set to: {self.budget_limit}")
            self.vector_memory = None
            self.input_validator = None
            
            # Initialize optimizations
            logger.debug("Setting up cache")
            self.cache = LRUCache(
                capacity=config.get_value("cache.max_entries", 1000),
                ttl=config.get_value("cache.ttl", 3600)
            )
            
            logger.debug("Setting up batch processor")
            self.batch_processor = BatchProcessor(
                max_batch_size=config.get_value("batch.max_batch_size", 10),
                max_concurrent=config.get_value("batch.max_concurrent", 3),
                cost_threshold=config.get_value("batch.cost_threshold", 1.0)
            )
            
            # Configure fallback chain
            logger.debug("Configuring fallback chain")
            fallback_models = []
            for model_name in config.get_value("fallback.chains.default", []):
                provider = self._get_provider_for_model(model_name)
                logger.debug(f"Adding fallback model: {model_name} with provider: {provider}")
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
            logger.info("ManusPrime agent initialization completed successfully")
        except Exception as e:
            logger.error(f"Error in ManusPrime.__init__: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def initialize(self) -> bool:
        """Initialize the agent and required plugins.
        
        Returns:
            bool: True if initialization was successful
        """
        logger.info("Starting ManusPrime agent initialization")
        if self.initialized:
            logger.info("Agent already initialized, skipping initialization")
            return True
            
        try:
            # Discover available plugins
            logger.debug("Discovering plugins")
            registry.discover_plugins()
            
            # Initialize vector memory if available
            logger.debug("Initializing vector memory")
            self.vector_memory = registry.get_plugin("vector_memory")
            if self.vector_memory:
                logger.debug("Vector memory plugin found, initializing")
                await self.vector_memory.initialize()
                logger.debug("Vector memory initialized successfully")
            else:
                logger.debug("Vector memory plugin not found")
            
            # Initialize input validator if available
            logger.debug("Initializing input validator")
            self.input_validator = registry.get_plugin("input_validator")
            if self.input_validator:
                logger.debug("Input validator plugin found, initializing")
                await self.input_validator.initialize()
                logger.info("Input validator plugin initialized")
            else:
                logger.info("Input validator plugin not found")
            
            # Activate required plugins based on configuration
            logger.debug("Activating plugins from configuration")
            for category, plugin_name in config.active_plugins.items():
                logger.debug(f"Processing plugin category: {category}, plugin: {plugin_name}")
                if not plugin_name:
                    logger.debug(f"No plugin specified for category {category}, skipping")
                    continue
                    
                logger.debug(f"Activating plugin {plugin_name} for category {category}")
                plugin_config = config.get_plugin_config(plugin_name)
                logger.debug(f"Plugin config for {plugin_name}: {plugin_config}")
                await registry.activate_plugin(plugin_name, plugin_config)
                logger.debug(f"Plugin {plugin_name} activated successfully")
            
            # Activate default provider if not already activated
            logger.debug("Checking for active provider plugin")
            provider_plugin = registry.get_active_plugin(PluginCategory.PROVIDER)
            if not provider_plugin:
                logger.debug(f"No active provider plugin found, activating default: {self.default_provider}")
                try:
                    provider_config = config.get_provider_config(self.default_provider)
                    logger.debug(f"Provider config for {self.default_provider}: {provider_config}")
                    provider_plugins = registry.get_plugin_classes_by_category(PluginCategory.PROVIDER)
                    logger.debug(f"Available provider plugins: {[p.name for p in provider_plugins]}")
                    
                    for plugin_class in provider_plugins:
                        if plugin_class.name == self.default_provider:
                            logger.debug(f"Found matching provider class for {self.default_provider}, activating")
                            await registry.activate_plugin(self.default_provider, provider_config)
                            logger.debug(f"Default provider {self.default_provider} activated successfully")
                            break
                    else:
                        logger.warning(f"Could not find provider plugin class for {self.default_provider}")
                except Exception as e:
                    logger.error(f"Error activating default provider: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.debug(f"Active provider plugin found: {provider_plugin.name}")
            
            self.initialized = True
            logger.info("ManusPrime agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing ManusPrime agent: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def analyze_task(self, task: str) -> str:
        """Analyze a task to determine the best model to use.
        
        Args:
            task: The task description
            
        Returns:
            str: The task category ('code', 'planning', 'tool_use', 'creative', or 'default')
        """
        logger.debug(f"Analyzing task: {task[:100]}...")
        try:
            task_lower = task.lower()
            
            for category, patterns in self.task_patterns.items():
                for pattern in patterns:
                    if pattern in task_lower:
                        logger.debug(f"Task categorized as: {category}")
                        return category
            
            logger.debug("Task categorized as: default")
            return 'default'
        except Exception as e:
            logger.error(f"Error in analyze_task: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 'default'
    
    def select_model(self, task: str) -> str:
        """Select the appropriate model for a given task.
        
        Args:
            task: The task description
            
        Returns:
            str: The selected model name
        """
        logger.debug(f"Selecting model for task: {task[:100]}...")
        try:
            category = self.analyze_task(task)
            selected_model = self.task_models.get(category, self.task_models['default'])
            logger.debug(f"Selected model: {selected_model} for category: {category}")
            return selected_model
        except Exception as e:
            logger.error(f"Error in select_model: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self.task_models['default']
    
    def _get_provider_for_model(self, model_name: str) -> Optional[str]:
        """Get the provider for a given model name."""
        logger.debug(f"Getting provider for model: {model_name}")
        try:
            # Check if we're using a local reference to config that might be shadowing the imported one
            config_value = config.get_value("providers", {})
            logger.debug(f"Checking providers config: {list(config_value.keys()) if isinstance(config_value, dict) else config_value}")
            
            for provider, provider_config in config_value.items():
                logger.debug(f"Checking provider: {provider}")
                if isinstance(provider_config, dict) and "models" in provider_config:
                    logger.debug(f"Provider {provider} has models: {provider_config['models']}")
                    if model_name in provider_config["models"]:
                        logger.debug(f"Provider for model {model_name} is {provider}")
                        return provider
            logger.warning(f"No provider found for model: {model_name}")
            return None
        except Exception as e:
            logger.error(f"Error in _get_provider_for_model for {model_name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        logger.info(f"Executing task: {task[:100] if isinstance(task, str) else task.prompt[:100]}...")
        try:
            # Initialize if needed
            if not self.initialized:
                logger.debug("Agent not initialized, initializing now")
                await self.initialize()
            
            # Convert to BatchTask if string
            if isinstance(task, str):
                task_id = str(uuid.uuid4())
                logger.debug(f"Converting string task to BatchTask with ID: {task_id}")
                task = BatchTask(
                    id=task_id,
                    prompt=task,
                    model=kwargs.get("model")
                )
            
            # Use batch processing if enabled
            if batch_mode and config.get_value("batch.enabled", True):
                logger.debug(f"Using batch processing for task: {task.id}")
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
            logger.debug("Analyzing task type")
            task_type = self.analyze_task(task.prompt)
            logger.debug(f"Task type: {task_type}")
            
            # Check if we should pre-select model based on task
            if not task.model:
                logger.debug("No model specified, selecting based on task")
                task.model = self.select_model(task.prompt)
                logger.debug(f"Selected model: {task.model}")
                
            # Validate input if validator is available
            validated_prompt = task.prompt
            if self.input_validator:
                logger.debug("Using input validator")
                try:
                    validation_result = await self.input_validator.execute(
                        input_text=task.prompt,
                        model=task.model,
                        context={"task_type": task_type}
                    )
                    logger.debug(f"Validation result: {validation_result}")
                    
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
                        logger.debug(f"Using sanitized prompt (length: {len(validated_prompt)})")
                        
                        # Log any warnings
                        warnings = validation_result.get("warnings", [])
                        if warnings:
                            logger.info(f"Input warnings: {warnings}")
                except Exception as e:
                    logger.error(f"Error in input validation: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Check cache if enabled - use validated prompt for cache key
            if config.get_value("cache.enabled", True):
                logger.debug("Checking cache")
                try:
                    cached = self.cache.get(validated_prompt, task.model or "default")
                    if cached:
                        logger.debug("Cache hit, returning cached result")
                        result.update(cached)
                        result["optimizations"]["cache_hit"] = True
                        result["execution_time"] = asyncio.get_event_loop().time() - start_time
                        return result
                except Exception as e:
                    logger.error(f"Error checking cache: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Get vector memory context
            logger.debug("Getting similar experiences from vector memory")
            try:
                similar_experiences = await self._get_similar_experiences(validated_prompt)
                logger.debug(f"Found {len(similar_experiences)} similar experiences")
                enhanced_prompt = self._enhance_prompt_with_context(
                    validated_prompt, 
                    similar_experiences
                ) if similar_experiences else validated_prompt
            except Exception as e:
                logger.error(f"Error getting similar experiences: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                enhanced_prompt = validated_prompt
            
            # Optimize request if enabled
            if config.get_value("performance.prompt_optimization", True):
                logger.debug("Optimizing prompt")
                try:
                    provider = self._get_provider_for_model(task.model) if task.model else self.default_provider
                    
                    # Add this check to handle None provider case
                    if not provider:
                        logger.warning(f"No provider found for model {task.model}, using default provider for optimization")
                        provider = self.default_provider
                        
                    logger.debug(f"Using provider for optimization: {provider}")
                    optimized = await connection_manager.optimize_request(
                        provider=provider,
                        prompt=enhanced_prompt,
                        **kwargs
                    )
                    enhanced_prompt = optimized["prompt"]
                    result["optimizations"]["prompt_optimized"] = True
                    logger.debug("Prompt optimization successful")
                except Exception as e:
                    logger.warning(f"Prompt optimization failed: {e}")
                    logger.warning(f"Traceback: {traceback.format_exc()}")
            
            # Try execution with fallback
            if config.get_value("fallback.enabled", True):
                logger.debug("Using fallback chain for execution")
                try:
                    fallback_result = await self.fallback_chain.execute(
                        enhanced_prompt,
                        self,
                        task_type=task_type
                    )
                    logger.debug(f"Fallback result success: {fallback_result.get('success', False)}")
                    if fallback_result["success"]:
                        result.update(fallback_result)
                        result["optimizations"]["fallback_used"] = True
                        
                        # Update cache
                        if config.get_value("cache.enabled", True):
                            logger.debug("Storing result in cache")
                            self.cache.put(validated_prompt, result["model"], result)
                        
                        return result
                except Exception as e:
                    logger.error(f"Error in fallback chain: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fall back to standard execution
            logger.debug("Fallback chain unsuccessful or disabled, using standard execution")
            result["model"] = task.model
            
            # Get provider plugin
            logger.debug("Getting provider plugin")
            provider = registry.get_active_plugin(PluginCategory.PROVIDER)
            if not provider:
                error_msg = "No provider plugin active"
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.debug(f"Using provider: {provider.name}")
            
            # Prepare and execute
            logger.debug("Preparing tools")
            try:
                tools = await self._prepare_tools(enhanced_prompt)
                logger.debug(f"Prepared {len(tools)} tools")
                response = await self._execute_with_provider(
                    provider=provider,
                    prompt=enhanced_prompt,
                    tools=tools,
                    model=task.model,
                    **kwargs
                )
                logger.debug("Provider execution successful")
            except Exception as e:
                logger.error(f"Error executing with provider: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            # Update result with response
            result.update({
                "success": True,
                "content": response.get("content", ""),
                "tokens": response.get("usage", {}).get("total_tokens", 0),
                "cost": response.get("usage", {}).get("cost", 0.0),
                "execution_time": asyncio.get_event_loop().time() - start_time
            })
            
            # Update metrics
            logger.debug(f"Updating metrics: tokens={result['tokens']}, cost={result['cost']}")
            self.total_tokens += result["tokens"]
            self.total_cost += result["cost"]
            self.model_usage[result["model"]] = self.model_usage.get(result["model"], 0) + 1
            
            # Store in cache - use validated prompt as key
            if config.get_value("cache.enabled", True):
                logger.debug("Storing result in cache")
                self.cache.put(validated_prompt, result["model"], deepcopy(result))
                
            # Store in vector memory - store original task for better retrieval
            logger.debug("Storing result in vector memory")
            try:
                await self._store_in_vector_memory(task.prompt, result)
            except Exception as e:
                logger.error(f"Error storing in vector memory: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            logger.info(f"Task execution completed successfully: tokens={result['tokens']}, cost={result['cost']}")
            return result
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            end_time = asyncio.get_event_loop().time() if 'start_time' in locals() else 0
            execution_time = end_time - start_time if 'start_time' in locals() else 0
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
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
        logger.debug(f"Executing batch task: {task.id}")
        try:
            # Add to batch processor
            task_id = await self.batch_processor.add_task(task)
            logger.debug(f"Added task to batch processor: {task_id}")
            
            # Process queue if this is a high priority task
            if task.priority > 0:
                logger.debug(f"High priority task {task_id}, processing queue immediately")
                return (await self.batch_processor.process_queue(self))[0]
            
            # Otherwise just return queued status
            logger.debug(f"Task {task_id} queued for batch processing")
            return {
                "success": True,
                "status": "queued",
                "task_id": task_id
            }
        except Exception as e:
            logger.error(f"Error in _execute_batch_task: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _get_similar_experiences(self, prompt: str) -> List[Dict]:
        """Get similar past experiences from vector memory."""
        logger.debug(f"Getting similar experiences for prompt: {prompt[:100]}...")
        if not self.vector_memory:
            logger.debug("No vector memory plugin available")
            return []
            
        try:
            # Get task type for potential filtering
            task_type = self.analyze_task(prompt)
            logger.debug(f"Task type for similarity search: {task_type}")
            
            search_result = await self.vector_memory.execute(
                operation="search",
                query=prompt,
                limit=3,
                filter_metadata={"task_type": task_type} if task_type != "default" else None
            )
            logger.debug(f"Vector memory search success: {search_result.get('success', False)}")
            return search_result["results"] if search_result["success"] else []
        except Exception as e:
            logger.warning(f"Failed to retrieve similar experiences: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
            return []

    async def _store_in_vector_memory(self, prompt: str, result: Dict):
        """Store successful result in vector memory."""
        logger.debug(f"Storing result in vector memory for prompt: {prompt[:100]}...")
        if not self.vector_memory or not result["success"]:
            if not self.vector_memory:
                logger.debug("No vector memory plugin available")
            else:
                logger.debug("Not storing unsuccessful result in vector memory")
            return
            
        try:
            task_type = self.analyze_task(prompt)
            logger.debug(f"Task type for vector memory storage: {task_type}")
            
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
            logger.debug("Successfully stored in vector memory")
        except Exception as e:
            logger.warning(f"Failed to store in vector memory: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")

    async def _prepare_tools(self, prompt: str) -> List[Dict]:
        """Prepare tool schemas based on prompt."""
        logger.debug(f"Preparing tools for prompt: {prompt[:100]}...")
        tools = []
        try:
            if any(word in prompt.lower() for word in ['tool', 'search', 'browse', 'crawl', 'scrape', 'website']):
                logger.debug("Tool-related keywords found in prompt, preparing tools")
                for category, plugin in registry.active_plugins.items():
                    if category != PluginCategory.PROVIDER:
                        logger.debug(f"Adding tool for plugin: {plugin.name} (category: {category})")
                        # Special handling for web crawler
                        if category == PluginCategory.WEB_CRAWLER:
                            logger.debug(f"Special handling for web crawler plugin: {plugin.name}")
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
            logger.debug(f"Prepared {len(tools)} tools")
        except Exception as e:
            logger.error(f"Error preparing tools: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return tools

    async def _execute_with_provider(self,
                                   provider,
                                   prompt: str,
                                   tools: List[Dict],
                                   model: str,
                                   **kwargs) -> Dict:
        """Execute with provider and handle tool calls."""
        logger.debug(f"Executing with provider: {provider.name}, model: {model}")
        response = {}
        
        try:
            if tools:
                logger.debug(f"Executing with {len(tools)} tools")
                response = await provider.generate_with_tools(
                    prompt=prompt,
                    tools=tools,
                    model=model,
                    temperature=kwargs.get('temperature', 0.7),
                    tool_choice=kwargs.get('tool_choice', 'auto')
                )
            else:
                logger.debug("Executing without tools")
                response = await provider.generate(
                    prompt=prompt,
                    model=model,
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', None)
                )
                
            content = response.get('content', '')
            tool_calls = response.get('tool_calls', [])
            
            if tool_calls:
                logger.debug(f"Provider returned {len(tool_calls)} tool calls")
                tool_results = []
                for tool_call in tool_calls:
                    if tool_call.get('type') != 'function':
                        logger.debug(f"Skipping non-function tool call: {tool_call.get('type')}")
                        continue
                        
                    function = tool_call.get('function', {})
                    plugin_name = function.get('name')
                    arguments = function.get('arguments', {})
                    
                    logger.debug(f"Processing tool call for plugin: {plugin_name}")
                    plugin = registry.get_plugin(plugin_name)
                    if plugin:
                        try:
                            # Special handling for web crawler to avoid unnecessary LLM usage
                            if plugin_name == "crawl4ai":
                                logger.debug("Special handling for crawl4ai plugin")
                                # Only use LLM if explicitly requested or needed for extraction
                                use_llm = arguments.get("use_llm", False) or "extraction_prompt" in arguments
                                arguments["use_llm"] = use_llm
                                arguments["from_agent"] = use_llm
                            logger.debug(f"Executing plugin {plugin_name} with arguments: {arguments}")
                            plugin_result = await plugin.execute(**arguments)
                            logger.debug(f"Plugin {plugin_name} execution successful")
                            tool_results.append(f"Tool: {plugin_name}\nResult: {plugin_result}")
                        except Exception as e:
                            logger.error(f"Error executing plugin {plugin_name}: {e}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            tool_results.append(f"Tool: {plugin_name}\nError: {str(e)}")
                
                if tool_results:
                    logger.debug(f"Generating follow-up with {len(tool_results)} tool results")
                    tool_output = "\n\n".join(tool_results)
                    follow_up = await provider.generate(
                        prompt=f"Task: {prompt}\n\nTool Results:\n{tool_output}\n\nBased on these results, please provide a complete answer.",
                        model=model,
                        temperature=kwargs.get('temperature', 0.7)
                    )
                    content = follow_up.get('content', '')
                    logger.debug("Follow-up generation successful")
                    
            response["content"] = content
            return response
        except Exception as e:
            logger.error(f"Error in _execute_with_provider: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _enhance_prompt_with_context(self, task: str, similar_experiences: List[Dict]) -> str:
        """Enhance a task prompt with context from similar past experiences.
        
        Args:
            task: The original task prompt
            similar_experiences: List of similar past experiences
            
        Returns:
            str: Enhanced prompt with context
        """
        logger.debug(f"Enhancing prompt with {len(similar_experiences)} similar experiences")
        try:
            if not similar_experiences:
                return task
                
            # Determine task type for better context structuring
            task_type = self.analyze_task(task)
            logger.debug(f"Task type for context enhancement: {task_type}")
            
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
            logger.debug(f"Enhanced prompt created (original length: {len(task)}, enhanced length: {len(enhanced_prompt)})")
            return enhanced_prompt
        except Exception as e:
            logger.error(f"Error enhancing prompt with context: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return task
    
    async def cleanup(self):
        """Clean up resources used by the agent."""
        logger.info("Cleaning up ManusPrime agent")
        try:
            await registry.cleanup_all()
            self.initialized = False
            logger.info("ManusPrime agent cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error cleaning up ManusPrime agent: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")


# Helper function to quickly execute a task
async def execute_task(task: str, **kwargs) -> Dict:
    """Helper function to execute a task with ManusPrime.
    
    Args:
        task: The task to execute
        **kwargs: Additional execution parameters
        
    Returns:
        Dict: The execution result
    """
    logger.info(f"Using helper function to execute task: {task[:100]}...")
    try:
        agent = ManusPrime()
        await agent.initialize()
        result = await agent.execute_task(task, **kwargs)
        await agent.cleanup()
        logger.info("Helper function execution completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in execute_task helper: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": 0.0
        }