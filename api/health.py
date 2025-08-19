# api/health.py
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from db.session import get_db
from plugins.registry import registry
from plugins.base import PluginCategory
from config import config
from utils.monitor import resource_monitor

logger = logging.getLogger("manusprime.api.health")

router = APIRouter()

class HealthChecker:
    """Centralized health checking for all system components."""
    
    def __init__(self):
        self.last_check = {}
        self.check_interval = 60  # seconds
        self.cached_results = {}
    
    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and basic operations."""
        try:
            db = next(get_db())
            start_time = time.time()
            
            # Test basic query
            result = db.execute(text("SELECT 1"))
            result.fetchone()
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "message": "Database connection successful"
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Database connection failed"
            }
    
    async def check_providers(self) -> Dict[str, Any]:
        """Check health of all configured AI providers."""
        provider_results = {}
        overall_status = "healthy"
        
        # Get all provider configurations
        provider_configs = config.get_value("providers", {})
        
        for provider_name, provider_config in provider_configs.items():
            if provider_name == "default":
                continue
                
            try:
                # Get provider plugin
                provider = registry.get_plugin(provider_name)
                if not provider:
                    provider_results[provider_name] = {
                        "status": "unavailable",
                        "message": "Provider plugin not loaded"
                    }
                    overall_status = "degraded"
                    continue
                
                # Check if provider has API key validation
                if hasattr(provider, 'has_valid_api_key'):
                    start_time = time.time()
                    is_valid = await provider.has_valid_api_key()
                    response_time = time.time() - start_time
                    
                    if is_valid:
                        provider_results[provider_name] = {
                            "status": "healthy",
                            "response_time_ms": round(response_time * 1000, 2),
                            "message": "API key valid and responsive"
                        }
                    else:
                        provider_results[provider_name] = {
                            "status": "unhealthy",
                            "message": "Invalid API key or authentication failed"
                        }
                        overall_status = "degraded"
                else:
                    provider_results[provider_name] = {
                        "status": "unknown",
                        "message": "Provider does not support health checks"
                    }
                    
            except Exception as e:
                logger.error(f"Provider {provider_name} health check failed: {e}")
                provider_results[provider_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "message": "Health check failed"
                }
                overall_status = "degraded"
        
        return {
            "status": overall_status,
            "providers": provider_results,
            "total_providers": len(provider_results),
            "healthy_providers": len([p for p in provider_results.values() if p["status"] == "healthy"])
        }
    
    async def check_plugins(self) -> Dict[str, Any]:
        """Check health of active plugins."""
        plugin_results = {}
        overall_status = "healthy"
        
        # Check active plugins by category
        for category in PluginCategory:
            plugin = registry.get_active_plugin(category)
            if plugin:
                try:
                    # Basic health check - verify plugin is initialized
                    if hasattr(plugin, 'initialized') and plugin.initialized:
                        plugin_results[plugin.name] = {
                            "status": "healthy",
                            "category": category.value,
                            "message": "Plugin initialized and ready"
                        }
                    else:
                        plugin_results[plugin.name] = {
                            "status": "unhealthy",
                            "category": category.value,
                            "message": "Plugin not properly initialized"
                        }
                        overall_status = "degraded"
                        
                except Exception as e:
                    logger.error(f"Plugin {plugin.name} health check failed: {e}")
                    plugin_results[plugin.name] = {
                        "status": "unhealthy",
                        "category": category.value,
                        "error": str(e),
                        "message": "Health check failed"
                    }
                    overall_status = "degraded"
        
        return {
            "status": overall_status,
            "plugins": plugin_results,
            "total_plugins": len(plugin_results),
            "healthy_plugins": len([p for p in plugin_results.values() if p["status"] == "healthy"])
        }
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage and limits."""
        try:
            import psutil
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Determine overall status
            status = "healthy"
            warnings = []
            
            if cpu_percent > 80:
                status = "degraded"
                warnings.append(f"High CPU usage: {cpu_percent}%")
            
            if memory_percent > 80:
                status = "degraded"
                warnings.append(f"High memory usage: {memory_percent}%")
            
            if disk_percent > 90:
                status = "degraded"
                warnings.append(f"High disk usage: {disk_percent}%")
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "warnings": warnings,
                "message": "System resources within normal limits" if status == "healthy" else "Resource usage concerns detected"
            }
            
        except ImportError:
            return {
                "status": "unknown",
                "message": "psutil not available for system monitoring"
            }
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return {
                "status": "unknown",
                "error": str(e),
                "message": "System resource check failed"
            }

    async def get_usage_metrics(self) -> Dict[str, Any]:
        """Get current usage metrics from resource monitor."""
        try:
            if resource_monitor.active_session:
                summary = resource_monitor.get_summary()
                return {
                    "status": "active",
                    "current_session": {
                        "task_id": summary["task_id"],
                        "tokens_used": summary["tokens"]["total"],
                        "cost_incurred": summary["cost"],
                        "api_calls": summary["api_calls"]["total"],
                        "api_errors": summary["api_calls"]["errors"],
                        "models_used": summary["models"],
                        "session_duration": summary["session_duration"]
                    }
                }
            else:
                return {
                    "status": "idle",
                    "message": "No active monitoring session"
                }
        except Exception as e:
            logger.error(f"Usage metrics check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to retrieve usage metrics"
            }

# Create global health checker instance
health_checker = HealthChecker()

@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ManusPrime",
        "version": "0.1.0"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check of all system components."""
    start_time = time.time()
    
    # Run all health checks
    checks = await asyncio.gather(
        health_checker.check_database(),
        health_checker.check_providers(),
        health_checker.check_plugins(),
        health_checker.check_system_resources(),
        health_checker.get_usage_metrics(),
        return_exceptions=True
    )
    
    database_health, provider_health, plugin_health, system_health, usage_metrics = checks
    
    # Handle any exceptions
    for i, check in enumerate(checks):
        if isinstance(check, Exception):
            logger.error(f"Health check {i} failed with exception: {check}")
            checks[i] = {"status": "error", "error": str(check)}
    
    # Determine overall system status
    statuses = [
        database_health.get("status", "unknown"),
        provider_health.get("status", "unknown"),
        plugin_health.get("status", "unknown"),
        system_health.get("status", "unknown")
    ]
    
    if "unhealthy" in statuses:
        overall_status = "unhealthy"
    elif "degraded" in statuses:
        overall_status = "degraded"
    elif "unknown" in statuses:
        overall_status = "unknown"
    else:
        overall_status = "healthy"
    
    execution_time = time.time() - start_time
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "execution_time_ms": round(execution_time * 1000, 2),
        "checks": {
            "database": database_health,
            "providers": provider_health,
            "plugins": plugin_health,
            "system": system_health,
            "usage": usage_metrics
        }
    }

@router.get("/health/providers")
async def provider_health():
    """Check health of AI providers only."""
    result = await health_checker.check_providers()
    return result

@router.get("/health/database")
async def database_health():
    """Check database health only."""
    result = await health_checker.check_database()
    return result

@router.get("/metrics")
async def get_metrics():
    """Get basic system metrics."""
    try:
        # Get current usage from resource monitor
        usage_metrics = await health_checker.get_usage_metrics()
        
        # Get plugin metrics
        plugin_metrics = {}
        for category in PluginCategory:
            plugin = registry.get_active_plugin(category)
            if plugin and hasattr(plugin, 'info'):
                info = plugin.info
                if 'performance' in info:
                    plugin_metrics[plugin.name] = info['performance']
        
        # Get provider status summary
        provider_health = await health_checker.check_providers()
        provider_summary = {
            "total": provider_health.get("total_providers", 0),
            "healthy": provider_health.get("healthy_providers", 0),
            "status": provider_health.get("status", "unknown")
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "usage": usage_metrics,
            "plugins": plugin_metrics,
            "providers": provider_summary,
            "uptime_seconds": time.time() - getattr(get_metrics, 'start_time', time.time())
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Store start time for uptime calculation
get_metrics.start_time = time.time()
