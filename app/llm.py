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
from app.llm.providers import (
    _ask_openai,
    _ask_anthropic,
    _ask_mistral,
    _ask_gemini,
    _ask_deepseek
)

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
        if not hasattr(self, "clients"):  # Only initialize if not already initialized
            self.provider = config.providers.providers.get(
                config_name, 
                config.providers.providers[config.providers.default_provider]
            )
            self.model = None  # Will be set per request
            self.temperature = 0.7
            
            # Initialize provider-specific clients
            self.clients = {}
            self._initialize_clients()
            
            # Track usage
            self.model_usage = {}

    def _initialize_clients(self) -> None:
        """Initialize provider-specific clients."""
        if self.provider.type != "cloud":
            return

        base_url = self.provider.base_url

        # Initialize OpenAI-compatible client (OpenAI, DeepSeek)
        if "openai" in base_url or "deepseek" in base_url:
            self.clients["openai"] = AsyncOpenAI(
                api_key=self.provider.api_key,
                base_url=self.provider.base_url,
                timeout=self.provider.timeout
            )
        
        # Initialize Anthropic client
        if "anthropic" in base_url:
            if not AsyncAnthropic:
                raise ProviderNotAvailableError("Anthropic client not available. Install with 'pip install anthropic'")
            
            headers = {}
            if self.provider.capabilities.beta_features:
                headers["output-128k-2025-02-19"] = "true"
            
            self.clients["anthropic"] = AsyncAnthropic(
                api_key=self.provider.api_key,
                base_url=self.provider.base_url,
                timeout=self.provider.timeout,
                default_headers=headers
            )
        
        # Initialize Mistral client
        if "mistral" in base_url:
            if not MistralAsyncClient:
                raise ProviderNotAvailableError("Mistral client not available. Install with 'pip install mistralai'")
            
            self.clients["mistral"] = MistralAsyncClient(
                api_key=self.provider.api_key,
                endpoint=self.provider.base_url
            )
        
        # Initialize Gemini client
        if "googleapis" in base_url:
            if not genai:
                raise ProviderNotAvailableError("Gemini client not available. Install with 'pip install google-generativeai'")
            
            genai.configure(api_key=self.provider.api_key)
            self.clients["gemini"] = genai

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """Format messages for LLM by converting them to OpenAI message format."""
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

    def _get_token_limits(self, model: str) -> Dict[str, int]:
        """Get input/output token limits for a model."""
        if "claude" in model:
            if model == "claude-3.7-sonnet" and self.provider.capabilities.extended_thinking:
                return {
                    "input": 200000,  # 200K context
                    "output": 128000 if "output-128k-2025-02-19" in self.provider.capabilities.beta_features else 64000
                }
            return {"input": 200000, "output": 8192}  # Standard limits
        elif "gemini" in model:
            return {
                "input": 1048576,  # 1M tokens
                "output": 128000 if self.provider.capabilities.beta_features else 8192
            }
        elif any(x in model for x in ["mistral", "pixtral", "codestral"]):
            if "codestral" in model:
                return {"input": 262144, "output": 8192}  # 256K context
            return {"input": 131072, "output": 8192}  # 131K context
        elif "deepseek" in model:
            return {"input": 64000, "output": 8000}  # 64K context
        
        return {"input": 4096, "output": 4096}  # Default fallback

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int, cache_hit: bool = False) -> float:
        """Calculate cost based on token usage and model pricing."""
        costs = config.monitoring.costs
        
        input_cost = costs.get_cost(model, "input", cache_hit) * (prompt_tokens / 1000)
        output_cost = costs.get_cost(model, "output") * (completion_tokens / 1000)
        
        if self.provider.time_based_pricing and self.provider.is_discount_period():
            if "deepseek" in model:
                discount = 0.5 if "chat" in model else 0.25
                input_cost *= discount
                output_cost *= discount
        
        return input_cost + output_cost

    def validate_model(self, model: Optional[str] = None) -> str:
        """Validate and return the model name."""
        model_name = model or self.provider.models[0]
        if model_name not in self.provider.models:
            raise ModelNotAvailableError(
                f"Model {model_name} not available for provider {self.provider.__class__.__name__}"
            )
        return model_name

    def get_client(self, provider_type: str) -> Any:
        """Get provider client with validation."""
        client = self.clients.get(provider_type)
        if not client:
            raise ProviderNotAvailableError(f"{provider_type} client not initialized")
        return client

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

            # Validate model and get limits
            model_name = self.validate_model(model)
            limits = self._get_token_limits(model_name)
            max_tokens = limits["output"]
            
            # Track usage
            self.model_usage[model_name] = self.model_usage.get(model_name, 0) + 1
            logger.info(f"Using model: {model_name}")

            # Get appropriate response using implementations
            if "anthropic" in self.provider.base_url:
                response = await _ask_anthropic(
                    self, messages, model_name, temperature, max_tokens, stream, extended_thinking
                )
            elif "mistral" in self.provider.base_url:
                response = await _ask_mistral(
                    self, messages, model_name, temperature, max_tokens, stream
                )
            elif "googleapis" in self.provider.base_url:
                response = await _ask_gemini(
                    self, messages, model_name, temperature, max_tokens, stream
                )
            elif "deepseek" in self.provider.base_url:
                response = await _ask_deepseek(
                    self, messages, model_name, temperature, max_tokens, stream
                )
            else:  # OpenAI-compatible API
                response = await _ask_openai(
                    self, messages, model_name, temperature, max_tokens, stream
                )

            # Handle usage tracking
            if not stream:
                if hasattr(response, 'usage'):
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                else:
                    prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
                    completion_tokens = len(str(response.choices[0].message.content).split())

                cost = self._calculate_cost(
                    model_name,
                    prompt_tokens,
                    completion_tokens,
                    cache_hit=getattr(response, 'cache_hit', False)
                )

                resource_monitor.track_api_call(
                    model=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=cost,
                    success=True
                )

                elapsed = resource_monitor.end_timer("llm_call")
                if elapsed:
                    logger.debug(f"LLM call completed in {elapsed:.2f} seconds")

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty response from LLM")
                return response.choices[0].message.content

            return response  # Return AsyncGenerator for streaming

        except Exception as e:
            logger.error(f"Error in ask: {str(e)}")
            resource_monitor.end_timer("llm_call")
            resource_monitor.track_api_call(model=model or self.provider.models[0], success=False)
            raise

    # Import tool calling implementation
    ask_tool = ask_tool
