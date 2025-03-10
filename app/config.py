import threading
import tomllib
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


class LLMSettings(BaseModel):
    model: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    temperature: float = Field(1.0, description="Sampling temperature")
    api_type: str = Field("openai", description="API type (always openai for our providers)")


class CacheSettings(BaseModel):
    enable: bool = Field(True, description="Enable or disable content caching")
    expiration_time: int = Field(86400, description="Cache expiration time in seconds")
    max_cache_size: int = Field(100, description="Maximum number of items in cache")
    cache_dir: str = Field("cache", description="Cache directory")


class ModelCosts(BaseModel):
    """Model costs per 1K tokens."""
    __root__: Dict[str, float] = Field(default_factory=dict)

    def __getitem__(self, key):
        return self.__root__.get(key, 0.001)  # Default to low cost if not found


class MonitoringSettings(BaseModel):
    budget_limit: float = Field(0.0, description="Budget limit in dollars (0 for no limit)")
    enable_budget_alerts: bool = Field(True, description="Enable budget alerts")
    costs: ModelCosts = Field(default_factory=ModelCosts)


class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]
    cache: CacheSettings = Field(default_factory=CacheSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)


class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._load_initial_config()
                    self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("No configuration file found in config directory")

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        raw_config = self._load_config()
        
        # Get global defaults from base llm section
        base_llm = raw_config.get("llm", {})
        
        # Extract nested configs vs simple key-values
        llm_subconfigs = {}
        default_settings = {}
        
        for key, value in base_llm.items():
            if isinstance(value, dict):
                # This is a nested config (like llm.openai, llm.deepseek)
                llm_subconfigs[key] = value
            else:
                # This is a direct setting (like llm.model, llm.api_key)
                default_settings[key] = value
        
        # Create default settings dictionary
        default_settings = {
            "model": default_settings.get("model", "gpt-4o-mini"),
            "base_url": default_settings.get("base_url", "https://api.openai.com/v1"),
            "api_key": default_settings.get("api_key", ""),
            "max_tokens": default_settings.get("max_tokens", 4096),
            "temperature": default_settings.get("temperature", 0.7),
            "api_type": default_settings.get("api_type", "openai"),
        }
        
        # Setup default config
        providers = {"default": default_settings}
        
        # Add nested configs (override with their specific settings)
        for provider, provider_config in llm_subconfigs.items():
            providers[provider] = {**default_settings, **provider_config}
        
        # Load cache settings
        cache_config = raw_config.get("cache", {})
        cache_settings = CacheSettings(
            enable=cache_config.get("enable", True),
            expiration_time=cache_config.get("expiration_time", 86400),
            max_cache_size=cache_config.get("max_cache_size", 100),
            cache_dir=cache_config.get("cache_dir", "cache")
        )
        
        # Load monitoring settings
        monitoring_config = raw_config.get("monitoring", {})
        costs_dict = monitoring_config.get("costs", {})
        
        monitoring_settings = MonitoringSettings(
            budget_limit=monitoring_config.get("budget_limit", 0.0),
            enable_budget_alerts=monitoring_config.get("enable_budget_alerts", True),
            costs=ModelCosts(__root__=costs_dict)
        )
        
        self._config = AppConfig(
            llm=providers,
            cache=cache_settings,
            monitoring=monitoring_settings
        )

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        return self._config.llm
    
    @property
    def cache(self) -> CacheSettings:
        return self._config.cache
    
    @property
    def monitoring(self) -> MonitoringSettings:
        return self._config.monitoring


config = Config()
