# plugins/providers/deepseek.py
import logging
from typing import Dict, List, Optional, Any, ClassVar
from datetime import datetime, timezone

from openai import AsyncOpenAI
from plugins.base import Plugin, PluginCategory, ProviderPlugin
from config import config

logger = logging.getLogger("manusprime.plugins.deepseek")

class DeepSeekProvider(ProviderPlugin):
    """Provider plugin for DeepSeek models."""
    
    name: ClassVar[str] = "deepseek"
    description: ClassVar[str] = "DeepSeek provider for natural language and reasoning tasks"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.PROVIDER
    supported_models: ClassVar[List[str]] = [
        "deepseek-chat",
        "deepseek-reasoner"
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the DeepSeek provider.
        
        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self.client = None
        self.api_key = None
        self.base_url = "https://api.deepseek.com/v1"
        self.model_costs = {
            "deepseek-chat": 0.003,     # $0.003 per 1K tokens (estimation)
            "deepseek-reasoner": 0.006  # $0.006 per 1K tokens (estimation)
        }
        self.default_model = "deepseek-chat"
        
        # Time-based pricing windows
        self.discount_window = {
            "start_hour": 16,
            "start_minute": 30,
            "end_hour": 0,
            "end_minute": 30
        }
        self.discount_factor = 0.5  # 50% discount during discount window
    
    async def initialize(self) -> bool:
        """Initialize the DeepSeek client.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Extract config
            self.api_key = self.config.get("api_key", "")
            base_url = self.config.get("base_url", self.base_url)
            
            if not self.api_key:
                logger.error("DeepSeek API key not provided")
                return False
            
            # Initialize client with OpenAI compatibility
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)
            logger.info("DeepSeek provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing DeepSeek provider: {e}")
            return False
    
    def _is_discount_period(self) -> bool:
        """Check if current time is in the discount period (UTC 16:30-00:30)."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        minute = now.minute
        current_minutes = hour * 60 + minute
        
        # Calculate start and end minutes of discount window in minutes since midnight
        start_minutes = (self.discount_window["start_hour"] * 60 + 
                        self.discount_window["start_minute"])
        end_minutes = (self.discount_window["end_hour"] * 60 + 
                      self.discount_window["end_minute"])
        
        # If end time is earlier than start time, it wraps around midnight
        if end_minutes < start_minutes:
            end_minutes += 24 * 60  # Add a day in minutes
            if current_minutes < start_minutes:
                current_minutes += 24 * 60  # Add a day if we're after midnight
                
        return start_minutes <= current_minutes < end_minutes
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """Generate a response from DeepSeek.
        
        Args:
            prompt: The prompt to send to the model
            model: The specific model to use
            temperature: The sampling temperature
            max_tokens: The maximum number of tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dict: The response data
        """
        if not self.client:
            raise ValueError("DeepSeek provider not initialized")
        
        # Validate and select model
        model_name = model if model in self.supported_models else self.default_model
        
        try:
            # Prepare messages format
            messages = [{"role": "user", "content": prompt}]
            
            # Special parameters for deepseek-reasoner model
            if model_name == "deepseek-reasoner":
                # Add reasoning tokens parameter if not provided
                if "reasoning_tokens" not in kwargs:
                    kwargs["reasoning_tokens"] = 1000
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract content
            content = response.choices[0].message.content if response.choices else ""
            
            # Calculate cost
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # Apply time-based pricing if applicable
            discount = self._is_discount_period()
            discount_multiplier = self.discount_factor if discount else 1.0
            
            # Calculate cost (per 1K tokens)
            model_cost = self.get_model_cost(model_name)
            cost = (total_tokens / 1000) * model_cost * discount_multiplier
            
            # Return standardized response
            return {
                "content": content,
                "model": model_name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response from DeepSeek: {e}")
            raise
    
    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        tool_choice: str = "auto",
        **kwargs
    ) -> Dict:
        """Generate a response from DeepSeek that may include tool calls.
        
        Args:
            prompt: The prompt to send to the model
            tools: The tools available to the model
            model: The specific model to use
            temperature: The sampling temperature
            tool_choice: How to choose tools ("none", "auto", "required")
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dict: The response data
        """
        if not self.client:
            raise ValueError("DeepSeek provider not initialized")
        
        # Validate and select model
        model_name = model if model in self.supported_models else self.default_model
        
        try:
            # Prepare messages format
            messages = [{"role": "user", "content": prompt}]
            
            # Map tool_choice to format
            ds_tool_choice = tool_choice
            if tool_choice == "required" and tools:
                ds_tool_choice = {"type": "function", "function": {"name": tools[0]["function"]["name"]}}
            
            # Special parameters for deepseek-reasoner model
            if model_name == "deepseek-reasoner":
                # Add reasoning tokens parameter if not provided
                if "reasoning_tokens" not in kwargs:
                    kwargs["reasoning_tokens"] = 1000
                    
            # Skip tools if tool_choice is "none"
            if tool_choice == "none":
                tools = None
                
            # Generate response
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice=ds_tool_choice,
                **kwargs
            )
            
            # Extract content and tool calls
            message = response.choices[0].message if response.choices else None
            content = message.content if message else ""
            
            # Extract tool calls
            tool_calls = []
            if message and hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    # Convert to standard format
                    tool_calls.append({
                        'id': tool_call.id,
                        'type': 'function',
                        'function': {
                            'name': tool_call.function.name,
                            'arguments': tool_call.function.arguments
                        }
                    })
            
            # Calculate cost
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # Apply time-based pricing if applicable
            discount = self._is_discount_period()
            discount_multiplier = self.discount_factor if discount else 1.0
            
            # Calculate cost (per 1K tokens)
            model_cost = self.get_model_cost(model_name)
            cost = (total_tokens / 1000) * model_cost * discount_multiplier
            
            # Return standardized response
            return {
                "content": content if not tool_calls else None,
                "tool_calls": tool_calls,
                "model": model_name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response with tools from DeepSeek: {e}")
            raise
    
    async def execute(self, **kwargs) -> Dict:
        """Execute the provider's primary function (generate).
        
        Args:
            **kwargs: Arguments for the generate method
            
        Returns:
            Dict: The generation result
        """
        if "tools" in kwargs:
            return await self.generate_with_tools(**kwargs)
        else:
            return await self.generate(**kwargs)
    
    def get_default_model(self) -> str:
        """Get the default model for this provider.
        
        Returns:
            str: The name of the default model
        """
        return self.default_model
    
    def get_model_cost(self, model: str) -> float:
        """Get the cost per 1K tokens for a specific model.
        
        Args:
            model: The model name
            
        Returns:
            float: The cost per 1K tokens
        """
        return self.model_costs.get(model, 0.003)
    
    async def cleanup(self) -> None:
        """Clean up resources used by the provider."""
        self.client = None
