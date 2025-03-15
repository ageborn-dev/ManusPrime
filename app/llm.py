from typing import Dict, List, Literal, Optional, Union, Any, AsyncGenerator
from openai import AsyncOpenAI

# Optional provider imports with fallbacks
try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

try:
    from mistralai.client import MistralAsyncClient
except ImportError:
    MistralAsyncClient = None

try:
    from google import generativeai as genai
except ImportError:
    genai = None

from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.config import config
from app.logger import logger
from app.schema import Message
from app.utils.monitor import resource_monitor
from app.providers import provider_registry


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class ProviderNotAvailableError(ProviderError):
    """Exception raised when a provider is not available."""
    pass


class ModelNotAvailableError(ProviderError):
    """Exception raised when a model is not available."""
    pass


class LLM:
    """Main LLM interface that handles multiple providers."""
    
    _instances: Dict[str, "LLM"] = {}

    def __new__(cls, config_name: str = "default", llm_config: Optional[Dict[str, Any]] = None) -> "LLM":
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(self, config_name: str = "default", llm_config: Optional[Dict[str, Any]] = None) -> None:
        if not hasattr(self, "initialized"):
            # Get provider from config or registry
            provider_name = config_name
            
            # If provider isn't registered yet, initialize it
            if not provider_registry.get_provider(provider_name, raise_error=False):
                provider_config = config.providers.providers.get(
                    config_name, 
                    config.providers.providers[config.providers.default_provider]
                )
                
                # Check if this is a local model configuration
                if hasattr(provider_config, 'type') and provider_config.type == "local":
                    # Use Ollama for local models
                    provider_registry.initialize_provider("ollama", provider_config)
                    self.provider_name = "ollama"
                else:
                    # Use the specified provider for cloud models
                    provider_registry.initialize_provider(provider_name, provider_config)
                    self.provider_name = provider_name
            else:
                self.provider_name = provider_name
                
            self.temperature = 0.7
            self.model_usage = {}
            self.initialized = True

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """Format messages for LLM by converting them to standardized format."""
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                formatted_messages.append(message.to_dict())
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages
        for msg in formatted_messages:
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"Invalid role: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError("Message must contain either 'content' or 'tool_calls'")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        extended_thinking: bool = False,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Send a prompt to the LLM and get the response."""
        try:
            resource_monitor.start_timer("llm_call")
            
            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Get provider - handle local models
            provider_name = self.provider_name
            if model and model in config.providers.providers.get("ollama", {}).get("models", []):
                provider_name = "ollama"  # Override provider for local models
                
            provider = provider_registry.get_provider(provider_name)
            
            # Track usage for specific model
            if model:
                self.model_usage[model] = self.model_usage.get(model, 0) + 1
                logger.info(f"Using model: {model} with provider: {provider_name}")

            # Generate response
            response = await provider.generate(
                prompt=messages,
                model=model,
                temperature=temperature or self.temperature,
                max_tokens=None,  # Let provider choose based on model
                stream=stream,
                extended_thinking=extended_thinking
            )

            # Handle usage tracking for non-streaming responses
            if not stream:
                if hasattr(response, 'usage'):
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                    cost = response.usage.cost
                else:
                    # Fallback if usage not provided
                    prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
                    completion_tokens = len(str(response.content).split())
                    total_tokens = prompt_tokens + completion_tokens
                    cost = 0.0

                resource_monitor.track_api_call(
                    model=model or provider.get_default_model(),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=cost,
                    success=True
                )

                elapsed = resource_monitor.end_timer("llm_call")
                if elapsed:
                    logger.debug(f"LLM call completed in {elapsed:.2f} seconds")

                return response.content

            return response 

        except Exception as e:
            logger.error(f"Error in ask: {str(e)}")
            resource_monitor.end_timer("llm_call")
            resource_monitor.track_api_call(
                model=model or provider_registry.get_provider(self.provider_name).get_default_model(),
                success=False
            )
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        tools: List[dict],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        extended_thinking: bool = False,
    ) -> Any:
        """Send a prompt to the LLM and get a response that may include tool calls."""
        try:
            resource_monitor.start_timer("llm_tool_call")
            
            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Get provider - handle local models
            provider_name = self.provider_name
            if model and model in config.providers.providers.get("ollama", {}).get("models", []):
                provider_name = "ollama"
                
            provider = provider_registry.get_provider(provider_name)
            
            # Track usage for specific model
            if model:
                self.model_usage[model] = self.model_usage.get(model, 0) + 1
                logger.info(f"Using model with tools: {model} with provider: {provider_name}")

            # Generate response with tools
            response = await provider.generate_with_tools(
                prompt=messages,
                tools=tools,
                model=model,
                temperature=temperature or self.temperature,
                tool_choice=tool_choice,
                max_tokens=None,
                extended_thinking=extended_thinking
            )

            # Handle usage tracking
            if "usage" in response:
                usage = response["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                cost = usage.get("cost", 0.0)
            else:
                # Fallback if usage not provided
                prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
                completion_tokens = len(str(response.get("content", "")).split())
                total_tokens = prompt_tokens + completion_tokens
                cost = 0.0

            resource_monitor.track_api_call(
                model=model or provider.get_default_model(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                success=True
            )

            elapsed = resource_monitor.end_timer("llm_tool_call")
            if elapsed:
                logger.debug(f"LLM tool call completed in {elapsed:.2f} seconds")

            # Create standardized response object
            result = type('ToolResponse', (), {
                'content': response.get("content"),
                'tool_calls': response.get("tool_calls", []),
                'model': response.get("model", model),
                'choices': [],
                'usage': response.get("usage", {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost
                })
            })

            return result

        except Exception as e:
            logger.error(f"Error in ask_tool: {str(e)}")
            resource_monitor.end_timer("llm_tool_call")
            resource_monitor.track_api_call(
                model=model or provider_registry.get_provider(self.provider_name).get_default_model(),
                success=False
            )
            raise