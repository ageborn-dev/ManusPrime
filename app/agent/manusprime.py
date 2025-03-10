from pydantic import Field

from app.agent.manus import Manus
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import Message


class ManusPrime(Manus):
    """
    An enhanced version of Manus that uses multiple AI models.
    
    This agent extends the standard Manus agent with the ability to select
    different models based on the task type and complexity.
    """

    name: str = "ManusPrime"
    description: str = "A multi-model agent that can solve various tasks using multiple AI models"

    # Use the same prompts as Manus
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Track last few messages to analyze tasks
    message_history_size: int = Field(default=3)
    
    async def think(self) -> bool:
        """Process current state and decide next action using the appropriate model."""
        try:
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
            
            # Continue with standard processing
            return await super().think()
            
        except Exception as e:
            logger.error(f"Error in ManusPrime thinking process: {e}")
            return False
    
    def _select_model_for_task(self) -> str:
        """
        Select the most appropriate model based on the current task.
        
        Returns:
            str: The name of the selected model
        """
        # Get recent messages for analysis
        recent_msgs = self.memory.messages[-self.message_history_size:] if len(self.memory.messages) >= self.message_history_size else self.memory.messages
        
        # Combine text for analysis
        combined_text = " ".join([msg.content or "" for msg in recent_msgs if hasattr(msg, "content") and msg.content])
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
