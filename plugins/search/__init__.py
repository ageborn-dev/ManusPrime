# plugins/search/__init__.py
from plugins.base import Plugin, PluginCategory
from .google_search import GoogleSearchPlugin

__all__ = ['Plugin', 'PluginCategory', 'GoogleSearchPlugin']