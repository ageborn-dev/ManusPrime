# plugins/web_crawler/__init__.py
from plugins.base import Plugin, PluginCategory
from .crawl4ai_plugin import Crawl4AIPlugin

__all__ = ['Plugin', 'PluginCategory', 'Crawl4AIPlugin']