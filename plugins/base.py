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


class Plugin(ABC):
    """Base class for all plugins in ManusPrime."""
    
    # Class attributes that should be overridden by subclasses
    name: ClassVar[str] = "base_plugin"
    description: ClassVar[str] = "Base plugin class"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = None
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the plugin with optional configuration."""
        self.config = config or {}
        self.initialized = False
    
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
    
    @property
    def info(self) -> Dict:
        """Get information about the plugin."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category.value if self.category else None,
            "initialized": self.initialized
        }


class ProviderPlugin(Plugin):
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