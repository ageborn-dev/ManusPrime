# core/__init__.py
from core.agent import ManusPrime, execute_task
from core.model_selector import ModelSelector, model_selector
from core.plugin_manager import PluginManager, plugin_manager

__all__ = [
    'ManusPrime',
    'execute_task',
    'ModelSelector',
    'model_selector',
    'PluginManager',
    'plugin_manager'
]