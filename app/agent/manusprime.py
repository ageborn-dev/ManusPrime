import asyncio
from typing import Dict, List, Optional

from pydantic import Field

from app.agent.manus import Manus
from app.logger import logger
from app.memory.manager import MemoryManager
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import Message


class ManusPrime(Manus):
    """
    An enhanced version of Manus that uses multiple AI models and
    an advanced memory system with vector storage.
    
    This agent extends the standard Manus agent with the ability to select
    different models based on the task type and complexity, and leverages
    a semantic memory system for better context recall.
    """

    name: str = "ManusPrime"
    description: str = "A multi-model agent with semantic memory that can solve various tasks"

    # Use the same prompts as Manus
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Track last few messages to analyze tasks
    message_history_size: int = Field(default=3)
    
    # Enhanced memory system
    memory_manager: Optional[MemoryManager] = Field(default=None)
    use_semantic_memory: bool = Field(default=True)
    
    def __init__(self, **data):
        """Initialize the ManusPrime agent with enhanced memory."""
        super().__init__(**data)
        
        # Initialize memory manager if not provided
        if not self.memory_manager:
            self.memory_manager = MemoryManager(base_memory=self.memory)
            logger.info("Initialized MemoryManager for ManusPrime agent")
    
    async def think(self) -> bool:
        """Process current state and decide next action using the appropriate model."""
        try:
            # Select the most appropriate model based on the task
            model = self._select_model_for_task()
            
            # If using semantic memory, enhance context with relevant memories
            if self.use_semantic_memory and self.memory_manager:
                await self._enhance_context_with_memory()
            
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
            
            # Add to regular memory and semantic memory
            self.memory.add_message(assistant_msg)
            if self.memory_manager and self.use_semantic_memory:
                await self.memory_manager.add_message(assistant_msg)

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
    
    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop with enhanced memory."""
        if request:
            # Add request to regular memory
            user_msg = Message.user_message(request)
            self.memory.add_message(user_msg)
            
            # Add to semantic memory if enabled
            if self.memory_manager and self.use_semantic_memory:
                await self.memory_manager.add_message(user_msg)
        
        # Continue with regular execution
        return await super().run(request)
    
    def update_memory(
        self,
        role: str,
        content: str,
        **kwargs,
    ) -> None:
        """Override update_memory to also update semantic memory."""
        # Update regular memory first
        super().update_memory(role, content, **kwargs)
        
        # Also update semantic memory if enabled
        if self.memory_manager and self.use_semantic_memory:
            # Get the last message added to regular memory
            if self.memory.messages:
                last_message = self.memory.messages[-1]
                asyncio.create_task(self.memory_manager.add_message(last_message))
    
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
    
    async def _enhance_context_with_memory(self) -> None:
        """
        Enhance the conversation context with relevant memories.
        
        This method finds semantically relevant past information and adds it to 
        the conversation context to improve the agent's responses.
        """
        if not self.memory_manager or not self.memory.messages:
            return
            
        # Get the last user message to use as a query
        user_messages = [msg for msg in self.memory.messages[-5:] if hasattr(msg, 'role') and msg.role == "user"]
        if not user_messages:
            return
            
        # Use the most recent user message as query
        query = user_messages[-1].content
        if not query:
            return
            
        # Get relevant context from memory
        relevant_context = await self.memory_manager.get_context(
            query=query,
            include_recent=False,  # We already have recent messages in the context
            include_relevant=True,
            relevant_count=3  # Retrieve 3 most relevant messages
        )
        
        if not relevant_context:
            return
            
        # Format the relevant context
        context_message = "Here's some relevant information from previous interactions:\n\n"
        for i, msg in enumerate(relevant_context):
            role = getattr(msg, 'role', 'memory')
            content = getattr(msg, 'content', '')
            if content:
                context_message += f"{i+1}. {role}: {content}\n\n"
        
        # Add to memory as a system message
        system_msg = Message.system_message(context_message)
        self.memory.messages = [system_msg] + self.memory.messages[-5:]  # Keep system message and last 5 messages
        
        logger.info(f"Enhanced context with {len(relevant_context)} relevant memories")
    
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
        
        # If we have semantic memory, add a prompt to use it
        if self.memory_manager and self.use_semantic_memory:
            memory_prompt = "Use your semantic memory to recall relevant information from earlier in our conversation that might help solve this problem."
            self.next_step_prompt = f"{memory_prompt}\n{self.next_step_prompt}"
