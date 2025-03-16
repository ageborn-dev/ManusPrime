# plugins/__init__.py
from plugins.base import Plugin, PluginCategory

# Import all plugin types
from plugins.automation import ZapierPlugin
from plugins.browser import BrowserUserPlugin
from plugins.code_execution import PythonExecutePlugin
from plugins.file_system import FileManagerPlugin
from plugins.search import GoogleSearchPlugin
from plugins.utility import InputValidatorPlugin
from plugins.vector_store import VectorMemoryPlugin
from plugins.web_crawler import Crawl4AIPlugin

__all__ = [
    'Plugin', 
    'PluginCategory',
    'ZapierPlugin',
    'BrowserUserPlugin',
    'PythonExecutePlugin',
    'FileManagerPlugin',
    'GoogleSearchPlugin',
    'InputValidatorPlugin',
    'VectorMemoryPlugin',
    'Crawl4AIPlugin'
]