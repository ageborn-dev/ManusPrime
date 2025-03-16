"""Crawl4AI plugin implementation."""

import logging
import asyncio
from typing import Dict, Optional, Any, ClassVar

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logging.warning("crawl4ai not installed. Install with 'pip install crawl4ai'")

from .base import WebCrawlerPlugin, CrawlStrategy

logger = logging.getLogger("manusprime.plugins.web_crawler.crawl4ai")

class Crawl4AIPlugin(WebCrawlerPlugin):
    """Plugin for web crawling using Crawl4AI library."""
    
    name: ClassVar[str] = "crawl4ai"
    description: ClassVar[str] = "Web crawler and scraper using Crawl4AI"
    version: ClassVar[str] = "0.1.0"
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Crawl4AI plugin."""
        super().__init__(config)
        self.crawler = None
        self.lock = asyncio.Lock()
            
    async def initialize(self) -> bool:
        """Initialize the crawler."""
        if not CRAWL4AI_AVAILABLE:
            logger.error("crawl4ai not installed")
            return False

        try:
            # Get configuration
            headless = self.config.get("headless", True)
            user_data_dir = self.config.get("user_data_dir")
            proxy = self.config.get("proxy", {})

            # Configure browser
            browser_config = BrowserConfig(
                headless=headless,
                user_data_dir=str(user_data_dir) if user_data_dir else None,
                proxy=proxy if proxy.get("url") else None,
                verbose=True
            )

            # Initialize crawler
            self.crawler = AsyncWebCrawler(config=browser_config)
            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            await self.cleanup()
            return False
    
    async def arun(self, url: str, **kwargs) -> Dict[str, Any]:
        """Execute web crawling.
        
        Args:
            url: The URL to crawl
            **kwargs: Supported options:
                - output_format: "html", "markdown", or "json"
                - schema: Dict for structured data extraction
                - strategy: CrawlStrategy for deep crawling
                - max_pages: int for deep crawling limit
                - use_llm: bool, whether to use LLM for content extraction
        """
        if not self.crawler:
            return {"success": False, "error": "Not initialized"}

        async with self.lock:
            try:
                use_llm = kwargs.pop("use_llm", False)
                # Let the agent override LLM usage
                if "from_agent" in kwargs:
                    use_llm = kwargs.pop("from_agent")

                return await (self._llm_crawl(url, **kwargs) if use_llm 
                            else self._direct_crawl(url, **kwargs))
            except Exception as e:
                logger.error(f"Crawling error: {e}")
                return {"success": False, "error": str(e)}

    async def _direct_crawl(self, url: str, **kwargs) -> Dict[str, Any]:
        """Execute web crawling without LLM."""
        try:
            # Configure crawler
            output_format = kwargs.get("output_format", "html")
            schema = kwargs.get("schema", None)
            strategy = kwargs.get("strategy", CrawlStrategy.BASIC)
            max_pages = kwargs.get("max_pages", None)

            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.ENABLED if self.config.get("cache_enabled", True) else CacheMode.BYPASS,
                max_pages=max_pages or self.config.get("max_pages", 10)
            )

            # Add extraction strategy if needed
            if schema and output_format == "json":
                run_config.extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

            # Execute crawl
            if strategy != CrawlStrategy.BASIC:
                result = await self.crawler.deep_crawl(url=url, strategy=strategy, config=run_config)
            else:
                result = await self.crawler.arun(url=url, config=run_config)

            # Format response based on output type
            if output_format == "markdown":
                content = result.markdown.raw_markdown if result.markdown else ""
            elif output_format == "json":
                content = result.extracted_content
            else:  # html
                content = result.html

            return {
                "success": True,
                "content": content
            }
        except Exception as e:
            logger.error(f"Direct crawling error: {e}")
            return {"success": False, "error": str(e)}

    async def _llm_crawl(self, url: str, **kwargs) -> Dict[str, Any]:
        """Execute web crawling with LLM-based extraction."""
        try:
            output_format = kwargs.get("output_format", "markdown")
            extraction_prompt = kwargs.get("extraction_prompt", None)
            
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.ENABLED if self.config.get("cache_enabled", True) else CacheMode.BYPASS,
                extraction_strategy=LLMExtractionStrategy(
                    instruction=extraction_prompt
                ) if extraction_prompt else None
            )

            result = await self.crawler.arun(url=url, config=run_config)

            return {
                "success": True,
                "content": result.extracted_content if extraction_prompt else result.markdown.raw_markdown
            }
        except Exception as e:
            logger.error(f"LLM crawling error: {e}")
            return {"success": False, "error": str(e)}
    
    
    async def cleanup(self) -> None:
        """Clean up crawler resources."""
        if self.crawler:
            try:
                await self.crawler.close()
            except:
                pass
            finally:
                self.crawler = None
                self.initialized = False
