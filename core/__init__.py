# core/__init__.py
from core.agent import ManusPrime, execute_task
from core.plugin_manager import PluginManager, plugin_manager

__all__ = [
    'ManusPrime',
    'execute_task',
    'PluginManager',
    'plugin_manager'
]
