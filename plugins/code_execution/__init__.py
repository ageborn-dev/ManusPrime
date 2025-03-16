# plugins/code_execution/__init__.py
from plugins.base import Plugin, PluginCategory
from .python_execute import PythonExecutePlugin

__all__ = ['Plugin', 'PluginCategory', 'PythonExecutePlugin']