# plugins/sandbox/__init__.py
from plugins.base import Plugin, PluginCategory
from .selenium_sandbox import SeleniumSandboxPlugin

__all__ = ['Plugin', 'PluginCategory', 'SeleniumSandboxPlugin']