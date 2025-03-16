# plugins/vector_store/__init__.py
from plugins.base import Plugin, PluginCategory
from .vector_memory import VectorMemoryPlugin

__all__ = ['Plugin', 'PluginCategory', 'VectorMemoryPlugin']