import logging
import json
import os
from typing import Dict, List, Optional, Any

from config import config

logger = logging.getLogger("manusprime.core.ai_planner")

class AIPlannerException(Exception):
    """Exception raised for AI planner errors."""
    pass

class AIPlanner:
    """Handles AI-driven task analysis and planning."""
    
    def __init__(self):
        """Initialize the AI planner."""
        self.model_capabilities = config.get_value("model_capabilities", {})
        
    def _create_orchestration_prompt(self, task: str, available_models: Dict[str, List[str]], available_plugins: List[str]) -> str:
        """Create the prompt for task orchestration and planning."""
        # Format available models by provider
        model_list = []
        for provider, models in available_models.items():
            model_list.append(f"{provider}: {', '.join(models)}")
            
        # Create a clear model specification format
        model_format = "provider_name/model_name (e.g. " + next(iter(available_models.items()))[0] + "/" + next(iter(available_models.values()))[0] + ")"
        
        return f'''
You will create a master plan to complete this task: "{task}"

Available Models by Provider:
{os.linesep.join(model_list)}

Available Plugins:
{', '.join(available_plugins)}

Create a complete execution plan that:
1. Uses appropriate models for each subtask
2. Leverages available plugins effectively
3. Defines clear dependencies between subtasks

Important Model Selection:
- Always specify models as: {model_format}
- Only use models from the available list above
- Each step must have a valid model assigned

Return in this exact format:

TASK TYPE: [primary category of task]

PLUGINS NEEDED: [comma-separated list of required plugins]

EXECUTION MODE: [sequential or parallel]

STEP 1:
Description: [clear description of subtask]
Model: [provider_name/model_name]
Plugins: [plugins needed for this step]
Dependencies: none
Output: [what this step produces]

STEP 2:
Description: [clear description of subtask]
Model: [provider_name/model_name]
Plugins: [plugins needed for this step]
Dependencies: [step numbers this depends on, or "none"]
Output: [what this step produces]

Important:
1. Only use models actually available above
2. Break complex tasks into clear steps
3. Use appropriate plugins for each step
4. Define clear dependencies
5. Keep steps focused and specific
6. Don't include sections outside this format
'''

    def _validate_model(self, model_str: str, available_models: Dict[str, List[str]], provider: Any) -> str:
        """Validate and normalize model specification.
        
        Args:
            model_str: Model specification (provider/model or just model)
            available_models: Dictionary of available models
            provider: Default provider instance
            
        Returns:
            str: Normalized model specification (provider/model)
        """
        try:
            if '/' in model_str:
                provider_name, model_name = model_str.split('/')
                if (provider_name in available_models and 
                    model_name in available_models[provider_name]):
                    return f"{provider_name}/{model_name}"
            else:
                # Try to find model in default provider
                default_name = provider.__class__.__name__
                if model_str in available_models.get(default_name, []):
                    return f"{default_name}/{model_str}"
                
                # Try to find model in any provider
                for p_name, models in available_models.items():
                    if model_str in models:
                        return f"{p_name}/{model_str}"
            
            # If no valid model found, return default
            return f"{provider.__class__.__name__}/{provider.get_default_model()}"
            
        except Exception as e:
            logger.warning(f"Error validating model {model_str}: {e}")
            return f"{provider.__class__.__name__}/{provider.get_default_model()}"

    async def create_execution_plan(self, task: str, provider: Any, available_models: Dict[str, List[str]], cache: Optional[Any] = None) -> Dict[str, Any]:
        """Create an execution plan for a task using available models."""
        try:
            if not available_models:
                logger.warning("No available models provided, using default provider models only")
                try:
                    models = await provider.get_available_models()
                    available_models = {provider.__class__.__name__: models}
                except Exception as e:
                    logger.error(f"Error getting default provider models: {e}")
                    raise AIPlannerException("No models available for planning")

            # Get available plugins
            available_plugins = list(config.get_value("plugins.active", {}).keys())
            
            # Create orchestration prompt
            prompt = self._create_orchestration_prompt(task, available_models, available_plugins)
            
            # Get plan from default provider
            response = await provider.generate(
                prompt=prompt,
                model=provider.get_default_model(),
                temperature=0.7
            )
            
            logger.info("\n" + "="*50)
            logger.info("CREATING EXECUTION PLAN")
            logger.info("="*50)
            logger.info(f"Task: {task[:200]}...")
            
            # Initialize plan
            plan = {
                "analysis": {
                    "task_type": "default"
                },
                "execution_plan": {
                    "parallel_execution": False,
                    "steps": []
                }
            }
            
            # Parse the response
            content = response.get("content", "")
            if isinstance(content, dict):
                content = str(content)
            
            try:
                # Split into sections
                sections = content.strip().split('\n\n')
                
                for section in sections:
                    section = section.strip()
                    if not section:
                        continue
                        
                    if section.startswith('TASK TYPE:'):
                        task_type = section.split(':', 1)[1].strip()
                        if task_type:
                            plan["analysis"]["task_type"] = task_type
                            
                    elif section.startswith('PLUGINS NEEDED:'):
                        plugins = section.split(':', 1)[1].strip()
                        if plugins.lower() != 'none':
                            plan["execution_plan"]["required_plugins"] = [p.strip() for p in plugins.split(',')]
                            
                    elif section.startswith('EXECUTION MODE:'):
                        mode = section.split(':', 1)[1].strip().lower()
                        plan["execution_plan"]["parallel_execution"] = "parallel" in mode
                        
                    elif section.startswith('STEP '):
                        try:
                            # Parse step information
                            lines = section.split('\n')
                            step_num = lines[0].split()[1].rstrip(':')
                            
                            # Initialize step
                            step = {
                                "id": f"step-{step_num}",
                                "description": f"Step {step_num}",
                                "model": None,  # Will be set after validation
                                "plugins": [],
                                "dependencies": [],
                                "expected_output": "completion"
                            }
                            
                            # Parse step details
                            for line in lines[1:]:
                                if ':' not in line:
                                    continue
                                    
                                key, value = line.split(':', 1)
                                key = key.strip().lower()
                                value = value.strip()
                                
                                if key == 'description' and value:
                                    step["description"] = value
                                elif key == 'model' and value:
                                    # Validate and normalize model specification
                                    step["model"] = self._validate_model(value, available_models, provider)
                                elif key == 'plugins' and value and value.lower() != 'none':
                                    plugins = [p.strip() for p in value.split(',')]
                                    step["plugins"] = [p for p in plugins if p in available_plugins]
                                elif key == 'dependencies' and value and value.lower() != 'none':
                                    step["dependencies"] = [f"step-{d.strip()}" for d in value.split(',')]
                                elif key == 'output' and value:
                                    step["expected_output"] = value
                            
                            # Ensure step has a valid model
                            if not step["model"]:
                                step["model"] = f"{provider.__class__.__name__}/{provider.get_default_model()}"
                                
                            plan["execution_plan"]["steps"].append(step)
                            
                        except Exception as step_error:
                            logger.warning(f"Error parsing step in section: {section}. Error: {step_error}")
                            continue
                
                # Validate plan has steps
                if not plan["execution_plan"]["steps"]:
                    raise AIPlannerException("No valid steps found in execution plan")
                
                # Add performance estimates
                step_count = len(plan["execution_plan"]["steps"])
                plan["performance_estimates"] = {
                    "expected_duration": "short" if step_count <= 3 else "medium" if step_count <= 6 else "long",
                    "resource_intensity": "low" if step_count <= 3 else "medium" if step_count <= 6 else "high",
                    "parallelization_potential": plan["execution_plan"]["parallel_execution"]
                }
                
                # Log execution plan
                logger.info("\nEXECUTION PLAN CREATED:")
                logger.info(f"Steps: {step_count}")
                logger.info("Model Assignments:")
                for step in plan["execution_plan"]["steps"]:
                    logger.info(f"  Step {step['id']}: {step['model']}")
                logger.info(f"Mode: {plan['execution_plan'].get('parallel_execution', False)}")
                logger.info("="*50 + "\n")
                
                return plan
                
            except Exception as parse_error:
                logger.error(f"Error parsing plan: {parse_error}")
                raise AIPlannerException(f"Failed to parse execution plan: {parse_error}")
            
        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            raise AIPlannerException(f"Planning failed: {e}")
