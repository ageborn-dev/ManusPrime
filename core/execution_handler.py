import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from copy import deepcopy

from plugins.registry import registry
from config import config
from utils.performance import connection_manager
from core.memory_manager import MemoryManager
from core.tool_manager import ToolManager
from core.sandbox_manager import SandboxManager
from core.ai_planner import AIPlanner
from utils.cache import Cache

logger = logging.getLogger("manusprime.core.execution_handler")

class ExecutionHandler:
    """Handles task execution flows using AI planning."""
    
    def __init__(self, 
                memory_manager: MemoryManager,
                tool_manager: ToolManager,
                sandbox_manager: SandboxManager,
                ai_planner: AIPlanner):
        """Initialize ExecutionHandler."""
        self.memory_manager = memory_manager
        self.tool_manager = tool_manager
        self.sandbox_manager = sandbox_manager
        self.ai_planner = ai_planner
        
    async def execute(self, 
                     prompt: str, 
                     provider: Any, 
                     cache: Optional[Union[Cache, Any]] = None,
                     execution_plan: Dict[str, Any] = None,
                     available_models: Dict[str, List[str]] = None,
                     **kwargs) -> Dict[str, Any]:
        """Execute a task using the orchestration plan."""
        logger.info("\n" + "="*50)
        logger.info("STARTING TASK EXECUTION")
        logger.info("="*50)
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
            
            # Check cache if available (use provider name for domain separation)
            if cache:
                logger.debug("Checking cache")
                cached = cache.get(prompt, provider.__class__.__name__)
                if cached:
                    logger.debug("Cache hit, returning cached result")
                    cached["execution_time"] = asyncio.get_event_loop().time() - start_time
                    return cached
            
            # Validate execution plan
            if not isinstance(execution_plan, dict):
                logger.error("Invalid execution plan: not a dictionary")
                raise ValueError("Invalid execution plan structure")
            
            steps = execution_plan.get("steps", [])
            if not steps:
                logger.error("Execution plan contains no steps")
                raise ValueError("Execution plan must contain at least one step")
            
            # Log execution plan
            logger.info("\nEXECUTION PLAN VALIDATION:")
            logger.info(f"Total Steps: {len(steps)}")
            logger.info(f"Execution Mode: {'Parallel' if execution_plan.get('parallel_execution', False) else 'Sequential'}")
            
            # Prepare execution
            content_parts = {}
            pending_steps = {step["id"]: step for step in steps}
            completed_steps = set()
            
            logger.info("\n" + "-"*50)
            logger.info("STARTING STEP EXECUTION")
            logger.info("-"*50)
            
            # Process steps based on dependencies
            step_number = 1
            while pending_steps:
                # Find executable steps (dependencies met)
                executable_steps = [
                    step for step in pending_steps.values()
                    if all(dep in completed_steps for dep in step.get("dependencies", []))
                ]
                
                if not executable_steps:
                    raise ValueError("Circular dependency detected in execution plan")
                
                # Execute steps (parallel if allowed)
                if execution_plan.get("parallel_execution", False):
                    tasks = [
                        self._execute_step(
                            step=step,
                            provider=provider,
                            context="\n".join([content_parts.get(dep, "") for dep in step.get("dependencies", [])]),
                            available_models=available_models,
                            **kwargs
                        )
                        for step in executable_steps
                    ]
                    step_results = await asyncio.gather(*tasks)
                else:
                    step_results = []
                    for step in executable_steps:
                        step_result = await self._execute_step(
                            step=step,
                            provider=provider,
                            context="\n".join([content_parts.get(dep, "") for dep in step.get("dependencies", [])]),
                            available_models=available_models,
                            **kwargs
                        )
                        step_results.append(step_result)
                
                # Process results
                for step, step_result in zip(executable_steps, step_results):
                    logger.info(f"\nExecuting Step {step_number} ({step['id']}):")
                    logger.info(f"Description: {step['description']}")
                    logger.info(f"Model Used: {step.get('model', 'default')}")
                    
                    if 'content' in step_result:
                        logger.info("\nStep Output:")
                        logger.info("-" * 30)
                        logger.info(step_result['content'])
                        logger.info("-" * 30)
                    
                    logger.info("\nStep Metrics:")
                    logger.info(f"Tokens Used: {step_result.get('tokens', 0)}")
                    logger.info(f"Step Cost: ${step_result.get('cost', 0.0):.4f}")
                    
                    step_number += 1
                    
                    content_parts[step["id"]] = step_result["content"]
                    completed_steps.add(step["id"])
                    del pending_steps[step["id"]]
                    
                    # Update tracking
                    try:
                        result["tokens"] += step_result.get("tokens", 0)
                        result["cost"] += step_result.get("cost", 0.0)
                        if "models_used" not in result:
                            result["models_used"] = []
                        if step.get("model"):
                            result["models_used"].append(step["model"])
                        result["steps_executed"] = result.get("steps_executed", 0) + 1
                        
                        if "step_results" not in result:
                            result["step_results"] = {}
                        result["step_results"][step["id"]] = {
                            "model": step.get("model", "unknown"),
                            "tokens": step_result.get("tokens", 0),
                            "cost": step_result.get("cost", 0.0),
                            "success": True
                        }
                    except Exception as e:
                        logger.error(f"Error updating step result: {e}")
                        result["step_results"][step["id"]] = {
                            "error": str(e),
                            "success": False
                        }
            
            # Combine results in dependency order
            ordered_content = []
            for step in steps:
                if step["id"] in content_parts:
                    ordered_content.append(content_parts[step["id"]])
            
            result.update({
                "success": True,
                "content": "\n\n".join(ordered_content),
                "model": result["models_used"][0] if result["models_used"] else provider.get_default_model(),
                "execution_time": asyncio.get_event_loop().time() - start_time,
                "execution_pattern": "parallel" if execution_plan.get("parallel_execution", False) else "sequential",
                "performance_metrics": execution_plan.get("performance_estimates", {}),
                "all_models": result["models_used"]
            })
            
            # Cache successful result with provider name
            if cache and result["success"]:
                cache.put(prompt, result, provider.__class__.__name__)
            
            # Store in memory if needed
            task_type = execution_plan.get("analysis", {}).get("task_type", "default")
            await self.memory_manager.store_result(prompt, result, task_type)
            
            # Log final results
            logger.info("\n" + "="*50)
            logger.info("TASK EXECUTION COMPLETED")
            logger.info("="*50)
            logger.info(f"Total Steps Executed: {result['steps_executed']}")
            logger.info(f"Total Tokens: {result['tokens']}")
            logger.info(f"Total Cost: ${result['cost']:.4f}")
            logger.info(f"Execution Time: {result['execution_time']:.2f}s")
            logger.info(f"Models Used: {', '.join(result['models_used'])}")
            logger.info("="*50 + "\n")
            
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
                "cost": 0.0,
                "model": provider.get_default_model(),
                "models_used": [],
                "steps_executed": 0,
                "step_results": {}
            }
    
    async def _execute_step(self,
                          step: Dict[str, Any],
                          provider: Any,
                          context: str = "",
                          available_models: Dict[str, List[str]] = None,
                          **kwargs) -> Dict[str, Any]:
        """Execute a single step of the plan."""
        try:
            # Get correct provider and model for step
            step_provider = provider  # Default to orchestration provider
            step_model = provider.get_default_model()
            
            if step.get("model") and available_models:
                provider_model = step["model"].split("/")
                if len(provider_model) == 2:
                    provider_name, model_name = provider_model
                    if (provider_name in available_models and 
                        model_name in available_models[provider_name]):
                        # Get provider instance
                        step_provider = await registry.get_plugin(provider_name)
                        if step_provider:
                            step_model = model_name
            
            # Prepare execution prompt
            prompt = f"""
Context from previous steps:
{context}

Current task: {step['description']}

Complete this specific task, ensuring the output matches the expected type: {step['expected_output']}
"""
            
            # Execute with selected provider/model
            response = await step_provider.generate(
                prompt=prompt,
                model=step_model,
                temperature=kwargs.get("temperature", 0.7)
            )
            
            content = response.get("content", "")
            
            result = {
                "content": content,
                "tokens": response.get("usage", {}).get("total_tokens", 0),
                "cost": response.get("usage", {}).get("cost", 0.0)
            }
            
            # Handle UI requirements
            if any(p in ["browser", "sandbox"] for p in step.get("plugins", [])):
                sandbox_result = await self.sandbox_manager.execute(
                    result["content"],
                    task_type="sandbox",
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
            
            return {
                "success": False,
                "content": f"Error executing step: {str(e)}",
                "tokens": 0,
                "cost": 0.0
            }
