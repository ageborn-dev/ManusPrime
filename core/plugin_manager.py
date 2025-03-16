# core/plugin_manager.py
import asyncio
import logging
from typing import Dict, List, Optional, Any, Type

from plugins.base import Plugin, PluginCategory
from plugins.registry import registry
from config import config

logger = logging.getLogger("manusprime.plugin_manager")

class PluginManager:
    """Manager for plugin discovery, activation, and execution."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the plugin manager."""
        if not self._initialized:
            self._initialized = False
            self._lock = asyncio.Lock()
    
    async def initialize(self) -> bool:
        """Initialize the plugin manager and activate plugins defined in config.
        
        Returns:
            bool: True if initialization was successful
        """
        if self._initialized:
            return True
            
        async with self._lock:
            if self._initialized:  # Double-check inside lock
                return True
                
            try:
                # Discover available plugins
                registry.discover_plugins()
                
                # Activate plugins based on configuration
                activated_count = 0
                for category, plugin_name in config.active_plugins.items():
                    if not plugin_name:
                        continue
                        
                    plugin_config = config.get_plugin_config(plugin_name)
                    plugin = await registry.activate_plugin(plugin_name, plugin_config)
                    if plugin:
                        activated_count += 1
                
                logger.info(f"Activated {activated_count} plugins from configuration")
                self._initialized = True
                return True
                
            except Exception as e:
                logger.error(f"Error initializing plugin manager: {e}")
                return False
    
    async def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin instance by name, initializing it if needed.
        
        Args:
            name: The name of the plugin
            
        Returns:
            Optional[Plugin]: The plugin instance or None if not found
        """
        if not self._initialized:
            await self.initialize()
            
        plugin = registry.get_plugin(name)
        if not plugin:
            # Try to activate the plugin
            plugin_config = config.get_plugin_config(name)
            plugin = await registry.activate_plugin(name, plugin_config)
            
        return plugin
    
    async def get_active_plugin(self, category: PluginCategory) -> Optional[Plugin]:
        """Get the active plugin for a category, initializing it if needed.
        
        Args:
            category: The plugin category
            
        Returns:
            Optional[Plugin]: The active plugin for the category or None
        """
        if not self._initialized:
            await self.initialize()
            
        plugin = registry.get_active_plugin(category)
        if not plugin:
            # Try to activate the plugin defined in config
            plugin_name = config.get_active_plugin(category.value)
            if plugin_name:
                plugin_config = config.get_plugin_config(plugin_name)
                plugin = await registry.activate_plugin(plugin_name, plugin_config)
                
        return plugin
    
    async def execute_plugin(self, 
                           name: str, 
                           **kwargs) -> Any:
        """Execute a plugin by name.
        
        Args:
            name: The name of the plugin to execute
            **kwargs: Arguments to pass to the plugin
            
        Returns:
            Any: The result of plugin execution
            
        Raises:
            ValueError: If the plugin is not found or not initialized
        """
        plugin = await self.get_plugin(name)
        if not plugin:
            raise ValueError(f"Plugin '{name}' not found")
            
        if not plugin.initialized:
            raise ValueError(f"Plugin '{name}' is not initialized")
            
        return await plugin.execute(**kwargs)
    
    async def execute_category(self, 
                             category: PluginCategory, 
                             **kwargs) -> Any:
        """Execute the active plugin for a category.
        
        Args:
            category: The plugin category
            **kwargs: Arguments to pass to the plugin
            
        Returns:
            Any: The result of plugin execution
            
        Raises:
            ValueError: If no active plugin exists for the category
        """
        plugin = await self.get_active_plugin(category)
        if not plugin:
            raise ValueError(f"No active plugin for category '{category.value}'")
            
        return await plugin.execute(**kwargs)
    
    def get_available_plugins(self, category: Optional[PluginCategory] = None) -> List[str]:
        """Get a list of available plugin names by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List[str]: List of plugin names
        """
        if category:
            return [plugin.name for plugin in registry.get_plugin_classes_by_category(category)]
        else:
            return list(registry.plugin_classes.keys())
    
    async def cleanup(self) -> None:
        """Clean up all active plugins."""
        await registry.cleanup_all()
        self._initialized = False


# Create a global instance
plugin_manager = PluginManager()