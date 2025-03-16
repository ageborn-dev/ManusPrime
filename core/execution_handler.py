import logging
import asyncio
from typing import Dict, Any, Optional
from copy import deepcopy

from plugins.registry import registry

from config import config
from utils.performance import connection_manager
from core.memory_manager import MemoryManager
from core.task_analyzer import TaskAnalyzer
from core.tool_manager import ToolManager
from core.sandbox_manager import SandboxManager

logger = logging.getLogger("manusprime.core.execution_handler")

class ExecutionHandler:
    """Handles task execution flows."""
    
    def __init__(self, 
                memory_manager: MemoryManager,
                task_analyzer: TaskAnalyzer,
                tool_manager: ToolManager,
                sandbox_manager: SandboxManager):
        """Initialize ExecutionHandler.
        
        Args:
            memory_manager: Memory manager instance
            task_analyzer: Task analyzer instance
            tool_manager: Tool manager instance
            sandbox_manager: Sandbox manager instance
        """
        self.memory_manager = memory_manager
        self.task_analyzer = task_analyzer
        self.tool_manager = tool_manager
        self.sandbox_manager = sandbox_manager
        
    async def execute(self, 
                     prompt: str, 
                     provider: Any, 
                     model: Optional[str] = None,
                     cache=None,
                     **kwargs) -> Dict[str, Any]:
        """Execute a task using the appropriate components.
        
        Args:
            prompt: The task prompt
            provider: The provider plugin instance
            model: Optional model override
            cache: Optional cache instance
            **kwargs: Additional execution parameters
            
        Returns:
            Dict[str, Any]: Execution result
        """
        logger.info("Starting task execution")
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Initialize result tracking
            result = {
                "success": False,
                "content": "",
                "tokens": 0,
                "cost": 0.0,
                "model": model or "",
                "execution_time": 0.0,
                "optimizations": {
                    "cache_hit": False,
                    "prompt_optimized": False,
                    "fallback_used": False
                }
            }
            
            # Get task type
            task_type = self.task_analyzer.analyze_task(prompt)
            
            # Check cache
            if cache:
                logger.debug("Checking cache")
                cached = cache.get(prompt, model or "default")
                if cached:
                    logger.debug("Cache hit, returning cached result")
                    result.update(cached)
                    result["optimizations"]["cache_hit"] = True
                    result["execution_time"] = asyncio.get_event_loop().time() - start_time
                    return result
            
            # Get similar experiences
            similar_experiences = await self.memory_manager.get_similar_experiences(prompt, task_type)
            
            # Enhance prompt with context
            enhanced_prompt = self.memory_manager.enhance_prompt_with_context(prompt, similar_experiences, task_type)
            
            # Optimize request if enabled
            if config.get_value("performance.prompt_optimization", True):
                try:
                    optimized = await connection_manager.optimize_request(
                        provider=provider.name,
                        prompt=enhanced_prompt,
                        **kwargs
                    )
                    enhanced_prompt = optimized["prompt"]
                    result["optimizations"]["prompt_optimized"] = True
                    logger.debug("Prompt optimization successful")
                except Exception as e:
                    logger.warning(f"Prompt optimization failed: {e}")
            
            # Prepare tools
            tools = self.tool_manager.prepare_tools(enhanced_prompt)
            
            # Execute with provider
            response = await self._execute_with_provider(
                provider=provider,
                prompt=enhanced_prompt,
                tools=tools,
                model=model,
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
            
            # Handle sandbox execution if needed
            if task_type == 'sandbox' and result["success"]:
                sandbox_result = await self.sandbox_manager.execute(
                    result["content"],
                    task_id=kwargs.get("task_id"),
                    maintain_session=True
                )
                if sandbox_result["success"]:
                    result["content"] = sandbox_result["enhanced_content"]
                    result["sandbox_session_id"] = sandbox_result.get("session_id")
                else:
                    result["content"] += f"\n\n> **Note**: Sandbox execution failed: {sandbox_result.get('error')}"
            
            # Only cache and store result if not in continuation mode
            if not kwargs.get("continue_task"):
                # Cache result
                if cache and result["success"]:
                    cache.put(prompt, model or "default", deepcopy(result))
                
                # Store in memory
                await self.memory_manager.store_result(prompt, result, task_type)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in execute: {e}")
            logger.error("Traceback:", exc_info=True)
            execution_time = asyncio.get_event_loop().time() - start_time
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "model": model or "",
                "tokens": 0,
                "cost": 0.0,
                "optimizations": {
                    "cache_hit": False,
                    "prompt_optimized": False,
                    "fallback_used": False
                }
            }
    
    async def _execute_with_provider(self,
                                   provider: Any,
                                   prompt: str,
                                   tools: list,
                                   model: Optional[str] = None,
                                   **kwargs) -> Dict[str, Any]:
        """Execute with provider and handle tool calls.
        
        Args:
            provider: The provider plugin instance
            prompt: The prompt to execute
            tools: List of available tools
            model: Optional model override
            **kwargs: Additional execution parameters
            
        Returns:
            Dict[str, Any]: Execution result
        """
        try:
            # Add execution instructions
            execution_instruction = """
You are an autonomous task execution agent. Your job is to complete the entire requested task without asking for confirmation or waiting for user input between steps.

Important instructions:
1. Complete the ENTIRE task in one response
2. Don't ask for permission to continue or for clarification
3. Make reasonable assumptions when details aren't specified
4. Use any available tools to accomplish the task
5. If the task involves creating something, create the complete solution
6. Provide the final result, not just a plan or outline

When you finish the task, include a brief closing sentence like "The task has been completed. Let me know if you need any adjustments."
            """
            
            enhanced_prompt = f"{execution_instruction}\n\nTask: {prompt}"
            
            if tools:
                logger.debug(f"Executing with {len(tools)} tools")
                response = await provider.generate_with_tools(
                    prompt=enhanced_prompt,
                    tools=tools,
                    model=model,
                    temperature=kwargs.get('temperature', 0.7),
                    tool_choice=kwargs.get('tool_choice', 'auto')
                )
                
                # Handle tool calls
                if tool_calls := response.get('tool_calls', []):
                    tool_results = []
                    for tool_call in tool_calls:
                        result = await self.tool_manager.execute_tool(tool_call, registry)
                        tool_results.append(f"Tool: {tool_call.get('function', {}).get('name')}\nResult: {result}")
                    
                    # Generate completion with tool results
                    if tool_results:
                        completion_instruction = """
Based on the tool results above, complete the entire task. Provide a comprehensive solution that fully addresses the original request. Do not ask for further clarification or permission to proceed - deliver the complete final result.

When you finish the task, include a brief closing sentence like "The task has been completed. Let me know if you need any adjustments."
                        """
                        
                        follow_up = await provider.generate(
                            prompt=f"Original Task: {prompt}\n\nTool Results:\n{tool_results}\n\n{completion_instruction}",
                            model=model,
                            temperature=kwargs.get('temperature', 0.7)
                        )
                        response["content"] = follow_up.get('content', '')
                
            else:
                logger.debug("Executing without tools")
                response = await provider.generate(
                    prompt=enhanced_prompt,
                    model=model,
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', None)
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing with provider: {e}")
            logger.error("Traceback:", exc_info=True)
            raise
