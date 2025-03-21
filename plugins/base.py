# plugins/base.py
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, ClassVar, Type


class PluginCategory(str, Enum):
    """Categories of plugins to organize capabilities and prevent conflicts."""
    PROVIDER = "provider"
    BROWSER = "browser"
    FILE_SYSTEM = "file_system"
    CODE_EXECUTION = "code_execution"
    SEARCH = "search"
    AUTOMATION = "automation"
    VECTOR_STORE = "vector_store"
    WEB_CRAWLER = "web_crawler"
    UTILITY = "utility"


def capability(*capabilities):
    """Decorator to mark plugin methods with capabilities."""
    def decorator(func):
        if not hasattr(func, '_capabilities'):
            func._capabilities = set()
        func._capabilities.update(capabilities)
        return func
    return decorator

def requires(*requirements):
    """Decorator to mark plugin methods with requirements."""
    def decorator(func):
        if not hasattr(func, '_requires'):
            func._requires = set()
        func._requires.update(requirements)
        return func
    return decorator

class Plugin(ABC):
    """Base class for all plugins in ManusPrime."""
    
    # Class attributes that should be overridden by subclasses
    name: ClassVar[str] = "base_plugin"
    description: ClassVar[str] = "Base plugin class"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = None
    
    # Plugin capabilities and requirements
    capabilities: ClassVar[set] = set()
    requirements: ClassVar[set] = set()
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the plugin with optional configuration."""
        self.config = config or {}
        self.initialized = False
        self._dependencies = {}
        self._performance_metrics = {
            "calls": 0,
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "last_error": None,
            "last_success_time": None
        }
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin with necessary setup.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the plugin's primary function.
        
        Args:
            **kwargs: Keyword arguments specific to the plugin
            
        Returns:
            Any: Result of the plugin execution
        """
        pass
    
    async def cleanup(self) -> None:
        """Clean up resources used by the plugin."""
        pass
    
    def inject_dependency(self, name: str, instance: 'Plugin') -> None:
        """Inject a plugin dependency.
        
        Args:
            name: Name of the dependency
            instance: Plugin instance to inject
        """
        self._dependencies[name] = instance
        
        # Try to call a setter if it exists
        setter_name = f"set_{name}"
        if hasattr(self, setter_name):
            getattr(self, setter_name)(instance)
    
    def get_dependency(self, name: str) -> Optional['Plugin']:
        """Get an injected dependency.
        
        Args:
            name: Name of the dependency
            
        Returns:
            Optional[Plugin]: The dependency plugin instance if found
        """
        return self._dependencies.get(name)
    
    def update_metrics(self, success: bool, response_time: float) -> None:
        """Update plugin performance metrics.
        
        Args:
            success: Whether the operation was successful
            response_time: Time taken for the operation
        """
        self._performance_metrics["calls"] += 1
        
        # Update success rate
        total_success = self._performance_metrics["success_rate"] * (self._performance_metrics["calls"] - 1)
        total_success += 1 if success else 0
        self._performance_metrics["success_rate"] = total_success / self._performance_metrics["calls"]
        
        # Update average response time
        old_avg = self._performance_metrics["avg_response_time"]
        self._performance_metrics["avg_response_time"] = (
            (old_avg * (self._performance_metrics["calls"] - 1) + response_time) / 
            self._performance_metrics["calls"]
        )
    
    @property
    def info(self) -> Dict:
        """Get information about the plugin."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category.value if self.category else None,
            "initialized": self.initialized,
            "capabilities": list(self.capabilities),
            "requirements": list(self.requirements),
            "dependencies": list(self._dependencies.keys()),
            "performance": self._performance_metrics
        }


class ProviderPlugin(Plugin):
    """Base class for AI provider plugins with enhanced capabilities."""
    
    capabilities = {
        "text_generation",
        "model_selection",
        "token_counting",
        "cost_estimation"
    }
    """Base class for AI provider plugins."""
    
    category: ClassVar[PluginCategory] = PluginCategory.PROVIDER
    supported_models: ClassVar[List[str]] = []
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """Generate a response from the AI model.
        
        Args:
            prompt: The prompt to send to the model
            model: The specific model to use (optional)
            temperature: The sampling temperature
            max_tokens: The maximum number of tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dict: The response data including 'content' and usage information
        """
        pass
    
    @abstractmethod
    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        tool_choice: str = "auto",
        **kwargs
    ) -> Dict:
        """Generate a response that may include tool calls.
        
        Args:
            prompt: The prompt to send to the model
            tools: The tools available to the model
            model: The specific model to use (optional)
            temperature: The sampling temperature
            tool_choice: How to choose tools ("none", "auto", "required")
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dict: The response data including 'content', 'tool_calls' and usage information
        """
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider.
        
        Returns:
            str: The name of the default model
        """
        pass
    
    @abstractmethod
    def get_model_cost(self, model: str) -> float:
        """Get the cost per 1K tokens for a specific model.
        
        Args:
            model: The model name
            
        Returns:
            float: The cost per 1K tokens
        """
        pass
        
    async def get_available_models(self) -> List[str]:
        """Get list of available models for this provider.
        
        Returns:
            List[str]: List of supported model names
        """
        return self.supported_models

# Add the missing BaseProvider class
class BaseProvider(ProviderPlugin):
    """Base class for provider plugins with common functionality."""
    
    name: ClassVar[str] = "base_provider"
    description: ClassVar[str] = "Base provider plugin class"
    
    async def initialize(self) -> bool:
        """Initialize the provider plugin.
        
        Returns:
            bool: True if initialization was successful
        """
        return True
    
    async def execute(self, **kwargs) -> Any:
        """Execute the provider plugin.
        
        Args:
            **kwargs: Keyword arguments for execution
            
        Returns:
            Any: Result of the execution
        """
        prompt = kwargs.get("prompt", "")
        if "tools" in kwargs:
            return await self.generate_with_tools(prompt, kwargs.get("tools"), **kwargs)
        else:
            return await self.generate(prompt, **kwargs)
    
    async def generate(self, prompt: str, model: Optional[str] = None, temperature: float = 0.7, 
                      max_tokens: Optional[int] = None, **kwargs) -> Dict:
        """Default implementation to be overridden."""
        raise NotImplementedError("Provider must implement generate method")
    
    async def generate_with_tools(self, prompt: str, tools: List[Dict], model: Optional[str] = None,
                                 temperature: float = 0.7, tool_choice: str = "auto", **kwargs) -> Dict:
        """Default implementation to be overridden."""
        raise NotImplementedError("Provider must implement generate_with_tools method")
    
    def get_default_model(self) -> str:
        """Default implementation to be overridden."""
        if self.supported_models:
            return self.supported_models[0]
        raise NotImplementedError("Provider must implement get_default_model method")
    
    def get_model_cost(self, model: str) -> float:
        """Default implementation to be overridden."""
        raise NotImplementedError("Provider must implement get_model_cost method")
