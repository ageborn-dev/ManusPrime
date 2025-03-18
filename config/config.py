# config.py
import os
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("manusprime")

class Config:
    """Configuration manager for ManusPrime."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.config_path = self._find_config_file()
            self.config_data = self._load_config()
            self._validate_config()
            
            # Extract key configurations
            self.default_provider = self.config_data.get("providers", {}).get("default", "anthropic")
            self.providers = self.config_data.get("providers", {})
            self.costs = self.config_data.get("costs", {})
            self.budget_limit = self.config_data.get("budget", {}).get("limit", 0.0)
            self.active_plugins = self.config_data.get("plugins", {}).get("active", {})
            
            self._initialized = True
    
    def _find_config_file(self) -> Path:
        """Find the configuration file in standard locations."""
        # Get the directory of the current module
        current_dir = Path(__file__).parent.absolute()
        project_root = current_dir.parent.parent
        
        possible_locations = [

            current_dir / "default.toml",
            project_root / "default.toml",
            project_root / "config" / "default.toml",
            project_root / "manusprime" / "config" / "default.toml",
            
            # System-wide locations
            Path.home() / ".manusprime/config/default.toml",
            Path("/etc/manusprime/config/default.toml")
        ]
        
        for location in possible_locations:
            if location.exists():
                logger.info(f"Using configuration file: {location}")
                return location

           
        # Build a better error message showing where we looked
        locations_str = "\n - ".join([str(loc) for loc in possible_locations])
        raise FileNotFoundError(f"No configuration file (default.toml) found. Looked in:\n - {locations_str}")
    
    def _load_config(self) -> Dict:
        """Load the configuration from the TOML file."""
        try:
            with open(self.config_path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _validate_config(self) -> None:
        """Validate the required configuration settings."""
        if "providers" not in self.config_data:
            raise ValueError("Configuration must include 'providers' section")
            
        if "default" not in self.config_data.get("providers", {}):
            raise ValueError("Configuration must specify a default provider")
    
    def get_provider_config(self, provider_name: Optional[str] = None) -> Dict:
        """Get configuration for a specific provider or the default provider."""
        provider = provider_name or self.default_provider
        provider_config = self.providers.get(provider, {})
        
        # Inject API key from environment variable if needed
        if isinstance(provider_config, dict) and "api_key" in provider_config:
            api_key = provider_config["api_key"]
            if isinstance(api_key, str) and api_key.startswith("$"):
                env_var = api_key[1:]
                provider_config["api_key"] = os.environ.get(env_var, "")
        
        return provider_config
    
    def get_model_cost(self, model_name: str) -> float:
        """Get the cost per 1K tokens for a specific model."""
        return float(self.costs.get(model_name, 0.001))
    
    def get_active_plugin(self, category: str) -> Optional[str]:
        """Get the active plugin for a specific category."""
        return self.active_plugins.get(category)
    
    def get_plugin_config(self, plugin_name: str) -> Dict:
        """Get configuration for a specific plugin."""
        return self.config_data.get("plugins", {}).get(plugin_name, {})
    
    def get_value(self, path: str, default: Any = None) -> Any:
        """Get a value from the configuration using dot notation path."""
        parts = path.split('.')
        config = self.config_data
        
        for part in parts:
            if isinstance(config, dict) and part in config:
                config = config[part]
            else:
                return default
                
        return config

# Create a global instance
config = Config()