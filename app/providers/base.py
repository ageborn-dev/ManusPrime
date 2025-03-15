from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

class ProviderConfig(BaseModel):
    """Configuration for an AI provider."""
    type: str = Field(..., description="Provider type (cloud/local)")
    base_url: str = Field(..., description="API base URL")
    models: List[str] = Field(..., description="Available models")
    api_key: Optional[str] = Field(None, description="API key if required")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum retry attempts")

class ModelUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float = 0.0

class ProviderResponse(BaseModel):
    """Standardized response from any provider."""
    content: str
    model: str
    usage: ModelUsage
    finish_reason: Optional[str]
    raw_response: Optional[Dict[str, Any]] = None

class BaseProvider(ABC):
    """Base class for all AI providers."""
    
    def __init__(self, config: Union[Dict[str, Any], ProviderConfig]):
        if isinstance(config, dict):
            self.config = ProviderConfig(**config)
        else:
            self.config = config
        self.models = self.config.models
        self._model_costs: Dict[str, float] = {}

    def set_model_cost(self, model: str, cost_per_1k: float) -> None:
        """Set the cost per 1K tokens for a model."""
        self._model_costs[model] = cost_per_1k

    def get_model_cost(self, model: str) -> float:
        """Get the cost per 1K tokens for a model."""
        return self._model_costs.get(model, 0.001)  # Default to low cost if not set

    def calculate_cost(self, usage: ModelUsage, model: str) -> float:
        """Calculate the cost for token usage."""
        cost_per_1k = self.get_model_cost(model)
        return (usage.total_tokens / 1000) * cost_per_1k

    @abstractmethod
    async def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ProviderResponse, AsyncGenerator[str, None]]:
        """Generate a response from the AI model."""
        pass

    @abstractmethod
    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response that may include tool calls."""
        pass

    def validate_model(self, model: Optional[str] = None) -> str:
        """Validate and return appropriate model name."""
        if not model:
            return self.models[0]  # Use first model as default
        if model not in self.models:
            raise ValueError(f"Model {model} not available for this provider. Available models: {self.models}")
        return model

    @staticmethod
    def format_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for the provider."""
        formatted = []
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("Messages must be dictionaries")
            if "role" not in msg or "content" not in msg:
                raise ValueError("Messages must contain 'role' and 'content'")
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"Invalid role: {msg['role']}")
            formatted.append(msg)
        return formatted

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming responses."""
        return True

    def supports_tools(self) -> bool:
        """Check if provider supports tool/function calling."""
        return False

    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return self.models[0]

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities and limitations."""
        return {
            "streaming": self.supports_streaming(),
            "tools": self.supports_tools(),
            "models": self.models,
            "max_tokens": 8192,  # Default, override in specific providers
            "supports_vision": False,  # Default, override if supported
            "supports_embeddings": False,  # Default, override if supported
        }
