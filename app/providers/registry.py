from typing import Dict, Optional, Type, Union, Any
from .base import BaseProvider, ProviderConfig

class ProviderRegistry:
    """Registry for managing multiple AI providers."""
    
    _instance = None  # Singleton instance
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self._providers: Dict[str, BaseProvider] = {}
            self._provider_classes: Dict[str, Type[BaseProvider]] = {}
            self._default_provider = None
            self._model_to_provider = {}
            self._initialized = True

    def register_provider_class(self, name: str, provider_class: Type[BaseProvider]):
        """Register a provider class for later instantiation."""
        self._provider_classes[name] = provider_class

    def initialize_provider(
        self,
        name: str,
        config: Union[Dict, ProviderConfig],
        set_as_default: bool = False
    ) -> BaseProvider:
        """Initialize a provider with configuration."""
        # If the provider is already initialized, return it
        if name in self._providers:
            return self._providers[name]
            
        # Determine provider type from config or name
        provider_type = name
        if hasattr(config, 'type') and config.type == "local":
            provider_type = "ollama"  # Use Ollama for local models
            
        # Get the provider class
        if provider_type not in self._provider_classes:
            raise ValueError(f"Provider class {provider_type} not registered")

        # Create the provider instance
        provider = self._provider_classes[provider_type](config)
        self._providers[name] = provider

        # Map models to this provider
        for model in provider.models:
            self._model_to_provider[model] = name

        # Set as default if requested or if no default is set
        if set_as_default or self._default_provider is None:
            self._default_provider = name

        return provider

    def get_provider(self, name: Optional[str] = None, raise_error: bool = True) -> Optional[BaseProvider]:
        """Get a provider instance by name or default provider."""
        if name is None:
            if self._default_provider is None:
                if raise_error:
                    raise ValueError("No default provider set")
                return None
            name = self._default_provider
            
        if name not in self._providers:
            if raise_error:
                raise ValueError(f"Provider {name} not initialized")
            return None
            
        return self._providers[name]

    def get_provider_for_model(self, model: str) -> Optional[BaseProvider]:
        """Get the appropriate provider for a specific model."""
        provider_name = self._model_to_provider.get(model)
        if not provider_name:
            return None
        return self.get_provider(provider_name)

    def list_providers(self) -> Dict[str, list]:
        """List all initialized providers and their available models."""
        return {
            name: provider.models 
            for name, provider in self._providers.items()
        }

    def list_all_models(self) -> Dict[str, str]:
        """List all available models and their providers."""
        return self._model_to_provider.copy()

    def get_default_provider(self) -> Optional[BaseProvider]:
        """Get the default provider instance."""
        return self.get_provider(self._default_provider, raise_error=False)

    def set_default_provider(self, name: str):
        """Set the default provider."""
        if name not in self._providers:
            raise ValueError(f"Provider {name} not initialized")
        self._default_provider = name

# Create a singleton instance
provider_registry = ProviderRegistry()