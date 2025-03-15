import os
import threading
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"

class ModelCapabilities(BaseModel):
    """Model capabilities configuration"""
    vision: bool = Field(False, description="Support for vision/image input")
    extended_thinking: bool = Field(False, description="Support for extended thinking mode")
    multimodal: bool = Field(False, description="Support for multimodal inputs")
    extended_context: bool = Field(False, description="Support for extended context windows")
    multilingual: bool = Field(False, description="Native multilingual support")
    context_caching: bool = Field(False, description="Support for context caching")
    chain_of_thought: bool = Field(False, description="Support for chain of thought reasoning")
    beta_features: List[str] = Field(default_factory=list, description="Beta features enabled")

class TimePricing(BaseModel):
    """Time-based pricing configuration"""
    standard_time: str = Field(..., description="Standard pricing time window")
    discount_time: str = Field(..., description="Discount pricing time window")

class TokenPricing(BaseModel):
    """Token pricing configuration"""
    input: Optional[float] = Field(None, description="Cost per 1K input tokens")
    output: Optional[float] = Field(None, description="Cost per 1K output tokens")
    input_cache_hit: Optional[float] = Field(None, description="Cost per 1K tokens with cache hit")
    input_cache_miss: Optional[float] = Field(None, description="Cost per 1K tokens with cache miss")

class ProviderSettings(BaseModel):
    """Settings for an AI provider"""
    type: str = Field(..., description="Provider type (cloud/local)")
    base_url: str = Field(..., description="API base URL")
    models: List[str] = Field(..., description="Available models")
    api_key: Optional[str] = Field(None, description="API key (if required)")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities, description="Model capabilities")
    time_based_pricing: Optional[TimePricing] = Field(None, description="Time-based pricing config")

    def is_discount_period(self) -> bool:
        """Check if current time is in discount period."""
        if not self.time_based_pricing:
            return False

        now = datetime.now(timezone.utc)
        hour = now.hour
        minute = now.minute
        current_minutes = hour * 60 + minute

        # Parse time windows (assuming format "UTC HH:MM-HH:MM")
        std_start, std_end = self.time_based_pricing.standard_time.split(" ")[1].split("-")
        std_start_h, std_start_m = map(int, std_start.split(":"))
        std_end_h, std_end_m = map(int, std_end.split(":"))
        
        std_start_mins = std_start_h * 60 + std_start_m
        std_end_mins = std_end_h * 60 + std_end_m

        # If current time is outside standard time, it's discount time
        return not (std_start_mins <= current_minutes < std_end_mins)

class ProvidersConfig(BaseModel):
    """Configuration for all providers"""
    default_provider: str = Field(..., description="Default provider to use")
    enable_local_models: bool = Field(True, description="Enable local model support")
    providers: Dict[str, ProviderSettings] = Field(..., description="Provider settings")

class TaskModels(BaseModel):
    """Task-specific model configuration"""
    code_generation: str = Field(..., description="Model for code generation")
    planning: str = Field(..., description="Model for planning")
    tool_use: str = Field(..., description="Model for tool usage")
    default: str = Field(..., description="Default model for other tasks")

class ZapierSettings(BaseModel):
    """Zapier integration settings"""
    enabled: bool = Field(False, description="Enable Zapier integration")
    webhook_secret: str = Field("", description="Webhook secret for validation")
    api_key: str = Field("", description="Zapier API key")
    allowed_actions: List[str] = Field(default_factory=list, description="Allowed Zapier actions")
    max_retries: int = Field(3, description="Maximum retry attempts")
    timeout: int = Field(30, description="Request timeout in seconds")

class CacheSettings(BaseModel):
    """Cache configuration"""
    enable: bool = Field(True, description="Enable content caching")
    expiration_time: int = Field(86400, description="Cache expiration time in seconds")
    max_cache_size: int = Field(100, description="Maximum cache items")
    cache_dir: str = Field("cache", description="Cache directory")
    context_caching: bool = Field(False, description="Enable context caching for supported providers")

class ModelCosts(BaseModel):
    """Model costs configuration"""
    __root__: Dict[str, TokenPricing] = Field(default_factory=dict)

    def get_cost(self, model: str, token_type: str = "input", cache_hit: bool = False) -> float:
        """Get cost per 1K tokens for a specific model and token type."""
        if model not in self.__root__:
            return 0.001  # Default low cost if not found

        pricing = self.__root__[model]
        
        if token_type == "output" and pricing.output is not None:
            return pricing.output
        elif token_type == "input":
            if cache_hit and pricing.input_cache_hit is not None:
                return pricing.input_cache_hit
            elif not cache_hit and pricing.input_cache_miss is not None:
                return pricing.input_cache_miss
            elif pricing.input is not None:
                return pricing.input
            
        return 0.001  # Default fallback

class MonitoringSettings(BaseModel):
    """Resource monitoring settings"""
    budget_limit: float = Field(0.0, description="Budget limit in dollars")
    enable_budget_alerts: bool = Field(True, description="Enable budget alerts")
    costs: ModelCosts = Field(default_factory=ModelCosts, description="Model costs configuration")

class AppConfig(BaseModel):
    """Complete application configuration"""
    providers: ProvidersConfig
    task_models: TaskModels
    zapier: ZapierSettings = Field(default_factory=ZapierSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

class Config:
    """Configuration singleton"""
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
        """Get the configuration file path"""
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("No configuration file found")

    def _load_config(self) -> dict:
        """Load the raw configuration file"""
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        """Load and parse the configuration"""
        raw_config = self._load_config()
        
        # Load provider configurations
        provider_configs = {}
        for provider_name, provider_data in raw_config.get("providers", {}).items():
            if provider_name != "default_provider" and provider_name != "enable_local_models":
                # Get API key from environment if specified
                if provider_data.get("api_key", "").startswith("sk-"):
                    env_var = f"{provider_name.upper()}_API_KEY"
                    provider_data["api_key"] = os.getenv(env_var, provider_data["api_key"])
                
                # Add capabilities if present
                if "capabilities" in provider_data:
                    capabilities = ModelCapabilities(**provider_data["capabilities"])
                    provider_data["capabilities"] = capabilities
                
                # Add time-based pricing if present
                if "time_based_pricing" in provider_data.get("capabilities", {}):
                    time_pricing = TimePricing(**provider_data["capabilities"]["time_based_pricing"])
                    provider_data["time_based_pricing"] = time_pricing
                
                provider_configs[provider_name] = ProviderSettings(**provider_data)

        providers_config = ProvidersConfig(
            default_provider=raw_config["providers"].get("default_provider", "openai"),
            enable_local_models=raw_config["providers"].get("enable_local_models", True),
            providers=provider_configs
        )

        # Load task-specific model configuration
        task_models = TaskModels(**raw_config.get("task_models", {
            "code_generation": "codestral-latest",
            "planning": "claude-3.7-sonnet",
            "tool_use": "claude-3.7-sonnet",
            "default": "mistral-small-latest"
        }))

        # Load Zapier configuration
        zapier_config = raw_config.get("zapier", {})
        if zapier_config.get("enabled", False):
            zapier_config["webhook_secret"] = os.getenv("ZAPIER_WEBHOOK_SECRET", 
                                                      zapier_config.get("webhook_secret", ""))
            zapier_config["api_key"] = os.getenv("ZAPIER_API_KEY", 
                                                zapier_config.get("api_key", ""))
        
        zapier_settings = ZapierSettings(**zapier_config)

        # Load other configurations
        cache_settings = CacheSettings(**raw_config.get("cache", {}))
        
        # Convert model costs to new format
        costs_data = raw_config.get("monitoring", {}).get("costs", {})
        model_costs = {}
        for model, cost_data in costs_data.items():
            if isinstance(cost_data, dict):
                model_costs[model] = TokenPricing(**cost_data)
            else:
                model_costs[model] = TokenPricing(input=float(cost_data))
        
        monitoring_settings = MonitoringSettings(
            budget_limit=raw_config.get("monitoring", {}).get("budget_limit", 0.0),
            enable_budget_alerts=raw_config.get("monitoring", {}).get("enable_budget_alerts", True),
            costs=ModelCosts(__root__=model_costs)
        )

        # Create complete configuration
        self._config = AppConfig(
            providers=providers_config,
            task_models=task_models,
            zapier=zapier_settings,
            cache=cache_settings,
            monitoring=monitoring_settings
        )

    @property
    def providers(self) -> ProvidersConfig:
        """Get provider configuration"""
        return self._config.providers

    @property
    def task_models(self) -> TaskModels:
        """Get task-specific model configuration"""
        return self._config.task_models

    @property
    def zapier(self) -> ZapierSettings:
        """Get Zapier configuration"""
        return self._config.zapier

    @property
    def cache(self) -> CacheSettings:
        """Get cache configuration"""
        return self._config.cache

    @property
    def monitoring(self) -> MonitoringSettings:
        """Get monitoring configuration"""
        return self._config.monitoring

config = Config()
