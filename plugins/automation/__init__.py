# plugins/automation/__init__.py
from plugins.base import Plugin, PluginCategory
from .zapier import ZapierPlugin

__all__ = ['Plugin', 'PluginCategory', 'ZapierPlugin']