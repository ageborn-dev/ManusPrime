import asyncio
from typing import Dict, List, Optional

from pydantic import Field

from app.agent.manus import Manus
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import Message
from app.utils.monitor import resource_monitor


class ManusPrime(Manus):
    """
    An enhanced version of Manus that uses multiple AI models.
    
    This agent extends the standard Manus agent with the ability to select
    different models based on the task type and complexity, and integrates
    resource monitoring and caching for better performance.
    """

    name: str = "ManusPrime"
    description: str = "A multi-model agent that can solve various tasks efficiently"

    # Use the same prompts as Manus
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Track last few messages to analyze tasks
    message_history_size: int = Field(default=3)
    
    # Budget controls
    budget_limit: Optional[float] = Field(default=None)
    
    def __init__(self, **data):
        """Initialize the ManusPrime agent with monitoring."""
        super().__init__(**data)
        
        # Set up resource monitoring
        if self.budget_limit:
            resource_monitor.start_session(budget_limit=self.budget_limit)
            resource_monitor.add_budget_listener(self._handle_budget_exceeded)
        else:
            resource_monitor.start_session()
            
        logger.info("ManusPrime agent initialized with resource monitoring")
    
    async def think(self) -> bool:
        """Process current state and decide next action using the appropriate model."""
        try:
            # Start task tracking
            resource_monitor.start_task(f"think_step_{self.current_step}")
            
            # Select the most appropriate model based on the task
            model = self._select_model_for_task()
            
            if self.next_step_prompt:
                user_msg = Message.user_message(self.next_step_prompt)
                self.messages += [user_msg]

            # Get response using the selected model
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=[Message.system_message(self.system_prompt)]
                if self.system_prompt
                else None,
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
                model=model,  # Use the selected model
            )
            
            logger.info(f"Used model '{model}' for current task")
            
            self.tool_calls = response.tool_calls
            
            # Log response info
            logger.info(f"✨ {self.name}'s thoughts: {response.content}")
            logger.info(
                f"🛠️ {self.name} selected {len(response.tool_calls) if response.tool_calls else 0} tools to use"
            )
            if response.tool_calls:
                logger.info(
                    f"🧰 Tools being prepared: {[call.function.name for call in response.tool_calls]}"
                )

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(
                    content=response.content, tool_calls=self.tool_calls
                )
                if self.tool_calls
                else Message.assistant_message(response.content)
            )
            self.memory.add_message(assistant_msg)

            # Process result same as parent class
            if self.tool_choices == "none":
                if response.tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if response.content:
                    return True
                return False

            if self.tool_choices == "required" and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == "auto" and not self.tool_calls:
                return bool(response.content)

            return bool(self.tool_calls)
            
        except Exception as e:
            logger.error(f"Error in ManusPrime thinking process: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False
        finally:
            # End task tracking
            resource_monitor.end_task()
    
    async def act(self) -> str:
        """Execute tool calls with monitoring."""
        try:
            # Start task tracking
            resource_monitor.start_task(f"act_step_{self.current_step}")
            
            if not self.tool_calls:
                if self.tool_choices == "required":
                    raise ValueError("Tool calls required but none provided")
                return self.messages[-1].content or "No content or commands to execute"

            results = []
            for command in self.tool_calls:
                # Track specific tool execution
                resource_monitor.start_task(f"tool_{command.function.name}")
                
                result = await self.execute_tool(command)
                logger.info(
                    f"🎯 Tool '{command.function.name}' completed with result: {result}"
                )
                
                # Track tool usage
                resource_monitor.track_tool_usage(command.function.name)
                
                # End tool tracking
                resource_monitor.end_task()

                # Add tool response to memory
                tool_msg = Message.tool_message(
                    content=result, tool_call_id=command.id, name=command.function.name
                )
                self.memory.add_message(tool_msg)
                results.append(result)

            return "\n\n".join(results)
        finally:
            # End task tracking
            resource_monitor.end_task()
    
    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop with monitoring."""
        # Start task tracking for entire run
        resource_monitor.start_task("complete_run")
        
        try:
            if request:
                # Add request to memory
                self.update_memory("user", request)
            
            return await super().run(request)
        finally:
            # End task tracking
            resource_monitor.end_task()
            
            # Log resource usage summary
            usage = resource_monitor.get_summary()
            logger.info(f"Run completed. Token usage: {usage['tokens']['total']}, Cost: ${usage['cost']:.4f}")
            if usage['models']:
                logger.info(f"Models used: {usage['models']}")
    
    def _select_model_for_task(self) -> str:
        """
        Select the most appropriate model based on the current task.
        
        Returns:
            str: The name of the selected model
        """
        # Get recent messages for analysis
        recent_msgs = self.memory.messages[-self.message_history_size:] if len(self.memory.messages) >= self.message_history_size else self.memory.messages
        
        # Combine text for analysis
        combined_text = " ".join([msg.content or "" for msg in recent_msgs if hasattr(msg, 'content') and msg.content])
        combined_text = combined_text.lower()
        
        # Determine task type
        if any(word in combined_text for word in ["code", "program", "function", "script", "python"]):
            # Use DeepSeek model for coding tasks
            logger.info("Identified coding task - using deepseek-chat")
            return "deepseek-chat"
            
        elif any(word in combined_text for word in ["plan", "organize", "strategy", "workflow", "complex"]):
            # Use GPT-4o for complex reasoning or planning
            logger.info("Identified complex reasoning/planning task - using gpt-4o")
            return "gpt-4o"
            
        # Default to more efficient model for general tasks
        logger.info("Using default efficient model gpt-4o-mini for general task")
        return "gpt-4o-mini"
    
    def is_stuck(self) -> bool:
        """Enhanced stuck detection with model switching."""
        stuck = super().is_stuck()
        
        if stuck:
            # If we're stuck, try using a more powerful model
            logger.warning("Detected stuck state - will try more powerful model")
            
        return stuck
    
    def handle_stuck_state(self):
        """Enhanced handling of stuck states with model switching."""
        # Add the original stuck handling
        super().handle_stuck_state()
        
        # Add note that we'll try a more powerful model
        logger.info("Switching to more powerful model (gpt-4o) to get unstuck")
    
    async def _handle_budget_exceeded(self, current_cost: float, budget_limit: float):
        """
        Handle budget limit exceeded event.
        
        Args:
            current_cost: Current estimated cost
            budget_limit: Budget limit that was exceeded
        """
        warning_message = f"⚠️ Budget alert: ${current_cost:.4f} exceeds limit of ${budget_limit:.4f}"
        logger.warning(warning_message)
        
        # Add a message to the agent's memory
        self.update_memory("system", f"{warning_message} Switching to more cost-effective models.")
        
        # Force use of the most cost-effective model
        logger.info("Switching to most cost-effective model (gpt-4o-mini)")
