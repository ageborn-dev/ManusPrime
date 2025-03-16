"""Web crawler plugins for ManusPrime."""

from .base import WebCrawlerPlugin
from .crawl4ai_plugin import Crawl4AIPlugin

__all__ = ["WebCrawlerPlugin", "Crawl4AIPlugin"]
