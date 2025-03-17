import logging
import asyncio
from typing import Dict, Any, Optional, List
from copy import deepcopy

from plugins.registry import registry
from config import config
from utils.performance import connection_manager
from core.memory_manager import MemoryManager
from core.tool_manager import ToolManager
from core.sandbox_manager import SandboxManager
from core.ai_planner import AIPlanner

logger = logging.getLogger("manusprime.core.execution_handler")

class ExecutionHandler:
    """Handles task execution flows using AI planning."""
    
    def __init__(self, 
                memory_manager: MemoryManager,
                tool_manager: ToolManager,
                sandbox_manager: SandboxManager,
                ai_planner: AIPlanner):
        """Initialize ExecutionHandler.
        
        Args:
            memory_manager: Memory manager instance
            tool_manager: Tool manager instance
            sandbox_manager: Sandbox manager instance
            ai_planner: AI planner instance
        """
        self.memory_manager = memory_manager
        self.tool_manager = tool_manager
        self.sandbox_manager = sandbox_manager
        self.ai_planner = ai_planner
        
    async def execute(self, 
                     prompt: str, 
                     provider: Any, 
                     cache=None,
                     task_analysis: Dict[str, Any] = None,
                     **kwargs) -> Dict[str, Any]:
        """Execute a task using AI planning and appropriate models.
        
        Args:
            prompt: The task prompt
            provider: The provider plugin instance
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
                "execution_time": 0.0,
                "steps_executed": 0,
                "models_used": []
            }
            
            # Check cache
            if cache:
                logger.debug("Checking cache")
                cached = cache.get(prompt, "default")
                if cached:
                    logger.debug("Cache hit, returning cached result")
                    cached["execution_time"] = asyncio.get_event_loop().time() - start_time
                    return cached
            
            # Get similar experiences using task type from analysis
            task_type = task_analysis.get("analysis", {}).get("task_type", "default")
            similar_experiences = await self.memory_manager.get_similar_experiences(prompt, task_type)
            
            # Enhance prompt with context
            enhanced_prompt = self.memory_manager.enhance_prompt_with_context(prompt, similar_experiences, task_type)
            
            # Get or use provided execution plan
            if not task_analysis:
                task_analysis = await self.ai_planner.create_execution_plan(enhanced_prompt, provider, cache)
            
            execution_plan = task_analysis.get("execution_plan", {})
            steps = execution_plan.get("steps", [])
            logger.info(f"Processing execution plan with {len(steps)} steps")
            
            # Prepare step execution
            content_parts = {}
            pending_steps = {step["id"]: step for step in steps}
            completed_steps = set()
            
            # Process steps based on dependencies
            while pending_steps:
                # Find steps that can be executed (all dependencies met)
                executable_steps = [
                    step for step_id, step in pending_steps.items()
                    if all(dep in completed_steps for dep in step.get("dependencies", []))
                ]
                
                if not executable_steps:
                    raise ValueError("Circular dependency detected in execution plan")
                
                # Execute steps in parallel if allowed
                if execution_plan.get("parallel_execution", False):
                    tasks = [
                        self._execute_step(
                            step=step,
                            provider=provider,
                            context="\n".join([content_parts.get(dep, "") for dep in step.get("dependencies", [])]),
                            **kwargs
                        )
                        for step in executable_steps
                    ]
                    step_results = await asyncio.gather(*tasks)
                else:
                    step_results = []
                    for step in executable_steps:
                        result = await self._execute_step(
                            step=step,
                            provider=provider,
                            context="\n".join([content_parts.get(dep, "") for dep in step.get("dependencies", [])]),
                            **kwargs
                        )
                        step_results.append(result)
                
                # Process results
                for step, step_result in zip(executable_steps, step_results):
                    content_parts[step["id"]] = step_result["content"]
                    completed_steps.add(step["id"])
                    del pending_steps[step["id"]]
                    
                    # Update tracking
                    result["tokens"] += step_result["tokens"]
                    result["cost"] += step_result["cost"]
                    result["models_used"].append(step["model"])
                    result["steps_executed"] += 1
                logger.info(f"Executing step {i}: {step['description']}")
                
                # Get provider for chosen model
                step_provider_name = self.ai_planner.find_provider_for_model(step["model"])
                if not step_provider_name:
                    # Try fallback model
                    fallback = self.ai_planner.find_fallback_model(step["model"])
                    if fallback:
                        step_provider_name, step["model"] = fallback
                    else:
                        raise ValueError(f"No provider found for model {step['model']}")
                
                # Get provider plugin
                step_provider = registry.get_plugin(step_provider_name)
                if not step_provider:
                    raise ValueError(f"Provider plugin {step_provider_name} not found")
                
                # Execute step
                step_result = await self._execute_step(
                    step=step,
                    provider=step_provider,
                    context="\n".join(content_parts),  # Previous steps' output
                    **kwargs
                )
                
                # Update tracking
                content_parts.append(step_result["content"])
                result["tokens"] += step_result["tokens"]
                result["cost"] += step_result["cost"]
                result["models_used"].append(step["model"])
                result["steps_executed"] += 1
            
            # Combine results in dependency order
            ordered_content = []
            for step in steps:
                if step["id"] in content_parts:
                    ordered_content.append(content_parts[step["id"]])
            
            result.update({
                "success": True,
                "content": "\n\n".join(ordered_content),
                "execution_time": asyncio.get_event_loop().time() - start_time,
                "execution_pattern": "parallel" if execution_plan.get("parallel_execution", False) else "sequential",
                "performance_metrics": task_analysis.get("performance_estimates", {})
            })
            
            # Cache result if successful
            if cache and result["success"]:
                cache.put(prompt, "default", deepcopy(result))
            
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
                "tokens": 0,
                "cost": 0.0
            }
    
    async def _execute_step(self,
                         step: Dict[str, Any],
                         provider: Any,
                         context: str = "",
                         **kwargs) -> Dict[str, Any]:
        """Execute a single step of the plan.
        
        Args:
            step: Step definition from plan
            provider: Provider to use
            context: Output from previous steps
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Step execution result
        """
        try:
            # Prepare execution prompt
            prompt = f"""
Context from previous steps:
{context}

Current task: {step['description']}

Complete this specific task, ensuring the output matches the expected type: {step['expected_output']}
"""
            
            # Execute with provider
            response = await provider.generate(
                prompt=prompt,
                model=step["model"],
                temperature=kwargs.get("temperature", 0.7)
            )
            
            result = {
                "content": response.get("content", ""),
                "tokens": response.get("usage", {}).get("total_tokens", 0),
                "cost": response.get("usage", {}).get("cost", 0.0)
            }
            
            # Handle UI requirements
            if step.get("requires_ui", False):
                sandbox_result = await self.sandbox_manager.execute(
                    result["content"],
                    task_type="sandbox",  # Always use sandbox for UI
                    task_id=kwargs.get("task_id"),
                    maintain_session=True
                )
                
                if sandbox_result["success"]:
                    result["content"] = sandbox_result["enhanced_content"]
                else:
                    result["content"] += f"\n\n> **Note**: UI execution failed: {sandbox_result.get('error')}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing step: {e}")
            logger.error("Traceback:", exc_info=True)
            raise
