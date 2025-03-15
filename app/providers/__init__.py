from .base import BaseProvider, ProviderConfig, ProviderResponse, ModelUsage
from .registry import ProviderRegistry, provider_registry
from .openai import OpenAIProvider
from .gemini import GeminiProvider
from .anthropic import AnthropicProvider
from .mistral import MistralProvider
from .deepseek import DeepSeekProvider
from .ollama import OllamaProvider

# Register all available provider classes
provider_registry.register_provider_class("openai", OpenAIProvider)
provider_registry.register_provider_class("gemini", GeminiProvider)
provider_registry.register_provider_class("anthropic", AnthropicProvider)
provider_registry.register_provider_class("mistral", MistralProvider)
provider_registry.register_provider_class("deepseek", DeepSeekProvider)
provider_registry.register_provider_class("ollama", OllamaProvider)

__all__ = [
    # Base classes
    "BaseProvider",
    "ProviderConfig",
    "ProviderResponse",
    "ModelUsage",
    "ProviderRegistry",
    
    # Provider implementations
    "OpenAIProvider",
    "GeminiProvider",
    "AnthropicProvider",
    "MistralProvider",
    "DeepSeekProvider",
    "OllamaProvider",
    
    # Global registry instance
    "provider_registry",
]

# Model mappings for easy reference
MODEL_TO_PROVIDER = {
    # OpenAI models
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    
    # Gemini models
    "gemini-2.0-flash": "gemini",
    "gemini-2.0-flash-lite": "gemini",
    "gemini-1.5-pro": "gemini",
    
    # Anthropic models
    "claude-3.7-sonnet": "anthropic",
    "claude-3.5-haiku": "anthropic",
    
    # Mistral models
    "mistral-large-latest": "mistral",
    "pixtral-large-latest": "mistral",
    "mistral-small-latest": "mistral",
    
    # DeepSeek models
    "deepseek-chat": "deepseek",
    "deepseek-reasoner": "deepseek",
    
    # Ollama models (these are examples, actual models depend on local installation)
    "llama2": "ollama",
    "codellama": "ollama",
    "mistral-local": "ollama"
}

# Define task-specific model defaults
TASK_MODEL_DEFAULTS = {
    "code_generation": "gpt-4o",
    "planning": "claude-3.7-sonnet",
    "reasoning": "mistral-large-latest",
    "chat": "gemini-2.0-flash-lite",
    "default": "gpt-4o-mini"
}

def get_provider_for_model(model_name: str) -> str:
    """Get the provider name for a given model."""
    return MODEL_TO_PROVIDER.get(model_name)

def get_model_for_task(task_name: str) -> str:
    """Get the recommended model for a specific task."""
    return TASK_MODEL_DEFAULTS.get(task_name, TASK_MODEL_DEFAULTS["default"])