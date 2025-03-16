# plugins/browser/__init__.py
from plugins.base import Plugin, PluginCategory
from .browser_user import BrowserUserPlugin

__all__ = ['Plugin', 'PluginCategory', 'BrowserUserPlugin']