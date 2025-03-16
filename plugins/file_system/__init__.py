# plugins/file_system/__init__.py
from plugins.base import Plugin, PluginCategory
from .file_manager import FileManagerPlugin

__all__ = ['Plugin', 'PluginCategory', 'FileManagerPlugin']