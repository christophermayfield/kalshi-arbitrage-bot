"""Optimized startup script for production deployment."""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.utils.config import Config
from src.utils.logging_utils import setup_logging, get_logger
from src.config.enhanced_config import create_enhanced_config, validate_config, get_cost_estimate
from src.utils.secrets import validate_secrets

logger = get_logger("startup")


class BotOptimizer:
    """Bot optimization and startup manager."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize optimizer."""
        self.config = config or {}
        
        # Optimization settings
        self.max_budget = self.config.get("budget.max_monthly", 300)
        self.performance_target = self.config.get("performance.target_latency_ms", 100)
        self.min_profit_threshold = self.config.get("trading.min_profit_cents", 10)
        
        logger.info("Bot optimizer initialized")
    
    def optimize_for_environment(self, environment: str = "development") -> Dict[str, Any]:
        """Optimize configuration for specific environment."""
        logger.info(f"Optimizing for {environment} environment")
        
        if environment == "production":
            # Production optimizations
            optimizations = {
                "trading": {
                    "high_frequency": {
                        "enabled": True,
                        "max_concurrent_orders": 15,
                        "order_timeout_ms": 2000,
                        "max_retries": 1,
                        "retry_delay_ms": 25
                    },
                    "min_profit_cents": max(5, self.min_profit_threshold // 2),
                    "auto_mode": True
                },
                "scanning": {
                    "scan_interval_ms": 50,
                    "max_concurrent_scans": 20,
                    "cache_ttl_seconds": 2
                },
                "monitoring": {
                    "performance_tracking": {
                        "enabled": True,
                        "latency_threshold_ms": 50,
                        "success_rate_threshold": 0.8
                    }
                },
                "statistical": {
                    "enabled": True,
                    "strategies": ["mean_reversion"],
                    "min_confidence": 0.6
                }
            }
            
        elif environment == "staging":
            # Staging optimizations
            optimizations = {
                "trading": {
                    "high_frequency": {
                        "enabled": False,
                        "max_concurrent_orders": 8,
                        "order_timeout_seconds": 20,
                        "max_retries": 3
                    },
                    "auto_mode": False
                },
                "scanning": {
                    "scan_interval_seconds": 1.0,
                    "max_concurrent_scans": 5,
                    "cache_ttl_seconds": 30
                },
                "statistical": {
                    "enabled": False
                }
            }
            
        else:
            # Development optimizations
            optimizations = {
                "trading": {
                    "paper_mode": True,
                    "scan_interval_seconds": 2.0,
                    "min_profit_cents": 20
                },
                "scanning": {
                    "scan_interval_seconds": 5.0
                },
                "statistical": {
                    "enabled": False
                }
            }
        
        return optimizations
    
    def optimize_for_budget(self) -> str:
        """Select best strategy within budget constraints."""
        best_strategy = optimize_for_budget(self.max_budget)
        
        logger.info(f"Optimized strategy: {best_strategy}")
        return best_strategy
    
    def apply_optimizations(self, base_config: Dict[str, Any], optimizations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimizations to base configuration."""
        config = base_config.copy()
        
        # Apply trading optimizations
        if "trading" in optimizations:
            config["trading"].update(optimizations["trading"])
        
        # Apply scanning optimizations
        if "scanning" in optimizations:
            config["scanning"].update(optimizations["scanning"])
        
        # Apply monitoring optimizations
        if "monitoring" in optimizations:
            config["monitoring"].update(optimizations["monitoring"])
        
        # Apply statistical arbitrage
        if "statistical" in optimizations:
            config["statistical"].update(optimizations["statistical"])
        
        return config
    
    def calculate_performance_impact(self, optimizations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate expected performance improvements."""
        base_latency = 500  # Base latency in ms
        base_concurrency = 3
        base_scan_interval = 1.0  # Base scan in seconds
        
        improvements = {}
        
        # Trading improvements
        if "trading" in optimizations:
            trading = optimizations["trading"]
            
            hf = trading.get("high_frequency", {})
            latency_improvement = (base_latency - hf.get("order_timeout_ms", base_latency)) / base_latency * 100
            concurrency_improvement = (hf.get("max_concurrent_orders", base_concurrency) - base_concurrency) / base_concurrency * 100
            
            improvements["trading"] = {
                "latency_improvement_percent": latency_improvement,
                "concurrency_improvement_percent": concurrency_improvement,
                "expected_latency_ms": hf.get("order_timeout_ms", base_latency)
            }
        
        # Scanning improvements
        if "scanning" in optimizations:
            scanning = optimizations["scanning"]
            
            base_scan_time = base_scan_interval * 1000  # Convert to ms
            new_scan_time = scanning.get("scan_interval_ms", base_scan_time)
            scan_improvement = (base_scan_time - new_scan_time) / base_scan_time * 100
            scan_concurrency = scanning.get("max_concurrent_scans", 2) - base_concurrency
            scan_improvement = scan_concurrency / base_concurrency * 100
            
            improvements["scanning"] = {
                "scan_speed_improvement_percent": scan_improvement,
                "concurrency_improvement_percent": scan_improvement
            }
        
        # Calculate total improvement
        total_improvement = 0
        for category in improvements.values():
            if "improvement_percent" in category:
                total_improvement += category["improvement_percent"]
        
        improvements["total_performance_improvement"] = total_improvement / len(improvements)
        
        return improvements
    
    def generate_startup_commands(self, config: Dict[str, Any]) -> list:
        """Generate optimized startup commands."""
        commands = []
        
        # Environment variables
        env_vars = [
            f"PYTHONPATH=/opt/python/bin:{os.environ.get('PATH', '')}",
            f"CACHE_SIZE={config.get('cache.connection_pool_size', 15)}",
            f"MAX_WORKERS={config.get('cache.max_workers', 10)}",
            "ULIMIT_NPROC=cpu",
            "PYTHONHASHSEED=0",
            "PYTHONUNBUFFERED=1"
        ]
        
        # Redis configuration
        cache_config = config.get("cache", {})
        if cache_config:
            commands.append(f"REDIS_HOST={cache_config.get('redis_host', 'localhost')}")
            commands.append(f"REDIS_PORT={cache_config.get('redis_port', 6379)}")
            commands.append(f"REDIS_DB={cache_config.get('redis_db', 0)}")
        
        # Performance settings
        if config.get("monitoring.performance_tracking", {}).get("enabled", False):
            commands.append("PYTHONASYNCIODEBUG=1")
            commands.append("PYTHONTRACEMALLOC=0")
            commands.append("PYTHONMALLOCSTATS=0")
        
        # Trading configuration
        if config.get("trading.high_frequency", {}).get("enabled", False):
            commands.extend([
                "UVLOOP_WORKERS=15",
                "UVLOOP_WORKER_CONNECTIONS=100",
                "UVLOOP_WORKER_MAX_REQUESTS=1000",
                "UVLOOP_WORKER_TIMEOUT_KEEP_ALIVE=15"
                "UVLOOP_WORKER_LIMIT_MAX_CONCURRENCY=15"
            ])
        
        return commands
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resources and make recommendations."""
        import psutil
        
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent()
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_available_gb = memory.available / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Disk info
        disk = psutil.disk_usage('/')
        disk_available_gb = disk.free / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        
        recommendations = []
        
        # CPU recommendations
        if cpu_count < 4:
            recommendations.append("Consider upgrading CPU for better performance")
        
        if cpu_count >= 8:
            recommendations.append("High CPU count available - optimize for parallel processing")
        
        # Memory recommendations
        if memory_available_gb < 2:
            recommendations.append("Low memory available - consider adding more RAM")
        elif memory_available_gb > 8:
            recommendations.append(f"High memory available ({memory_available_gb:.1f}GB) - can support very high-frequency operations")
        
        # Disk recommendations
        if disk_available_gb < 10:
            recommendations.append(f"Limited disk space available ({disk_available_gb:.1f}GB) - monitor disk usage")
        
        return {
            "cpu_count": cpu_count,
            "cpu_freq_mhz": cpu_freq[0] if cpu_freq else 0,
            "cpu_usage_percent": cpu_percent,
            "memory_available_gb": memory_available_gb,
            "memory_total_gb": memory_total_gb,
            "disk_available_gb": disk_available_gb,
            "disk_total_gb": disk_total_gb,
            "recommendations": recommendations
        }


class ProductionDeployer:
    """Production deployment manager."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize production deployer."""
        self.config = config or {}
        
        # Deployment settings
        self.deployment_type = self.config.get("deployment.type", "docker")
        self.target_servers = self.config.get("deployment.servers", ["localhost"])
        self.use_ssl = self.config.get("deployment.ssl", False)
        self.backup_enabled = self.config.get("deployment.backup", True)
        
        logger.info("Production deployer initialized")
    
    def generate_deployment_config(self) -> Dict[str, Any]:
        """Generate production deployment configuration."""
        config = self.config.copy()
        
        # Docker settings
        if self.deployment_type == "docker":
            config.update({
                "container_name": "arbitrage_bot",
                "replicas": 3,  # Load balanced
                "restart_policy": "always",
                "health_check_interval": 30,
                "resources": {
                    "cpu_limit": "0.5",
                    "memory_limit": "1g",
                    "ephemeral": False
                }
            })
        
        # Kubernetes settings
        elif self.deployment_type == "kubernetes":
            config.update({
                "replicas": 3,
                "resources": {
                    "requests": {
                        "cpu": "200m",
                        "memory": "1Gi"
                    },
                    "limits": {
                        "cpu": "500m",
                        "memory": "2Gi"
                    }
                },
                "health_check": {
                    "enabled": True,
                    "path": "/health",
                    "port": 8000
                }
            })
        
        return config
    
    def deploy_command(self) -> str:
        """Generate deployment command."""
        if self.deployment_type == "docker":
            return "docker-compose up -d"
        elif self.deployment_type == "kubernetes":
            return "kubectl apply -f deployment.yaml"
        else:
            return "echo 'No deployment configured'"


def validate_production_readiness(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate system is ready for production."""
    issues = []
    
    # Check required fields
    required_fields = [
        "kalshi.api_key_id",
        "kalshi.private_key_path",
        "trading.max_position_contracts",
        "risk.max_daily_loss_cents",
        "risk.max_open_positions"
    ]
    
    for field in required_fields:
        if not config.get(f"kalshi.{field}"):
            issues.append(f"Missing required field: {field}")
    
    # Validate values
    if config.get("trading.min_profit_cents", 0) <= 0:
        issues.append("min_profit_cents must be > 0")
    
    if config.get("risk.max_daily_loss_cents", 0) <= 0:
        issues.append("max_daily_loss_cents must be > 0")
    
    if config.get("risk.max_open_positions", 0) <= 0:
        issues.append("max_open_positions must be > 0")
    
    # Check paper mode for production
    if config.get("kalshi.demo_mode", True):
        issues.append("Cannot use demo_mode in production")
    
    # Check reasonable values
    if config.get("trading.max_position_contracts", 0) > 10000:
        issues.append("max_position_contracts seems very high for risk management")
    
    return {
        "ready": len(issues) == 0,
        "issues": issues,
        "warnings": []
    }


def setup_environment(config: Dict[str, Any]) -> bool:
    """Setup production environment."""
    logger.info("Setting up production environment...")
    
    try:
        # Create necessary directories
        directories = [
            "/app/data",
            "/app/logs",
            "/app/config",
            "/app/cache"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Set appropriate permissions
        os.chmod("/app/data", 0o755)
        os.chmod("/app/logs", 0o755)
        
        # Create log directories with rotation
        log_dirs = [
            "/app/logs/trading",
            "/app/logs/scanning",
            "/app/logs/errors",
            "/app/logs/performance"
        ]
        
        for directory in log_dirs:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("Environment setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return False


def main():
    """Main startup function."""
    logger = setup_logging(level="INFO")
    
    # Load configuration
    config = Config()
    
    # Validate configuration
    if not validate_config(config)["valid"]:
        logger.error("Configuration validation failed")
        return 1
    
    # Setup environment
    if not setup_environment(config):
        logger.error("Environment setup failed")
        return 1
    
    # Optimize for current environment
    environment = os.getenv("ENVIRONMENT", "development")
    optimizer = BotOptimizer(config)
    optimizations = optimizer.optimize_for_environment(environment)
    
    # Apply optimizations
    enhanced_config = optimizer.apply_optimizations(config, optimizations)
    
    # Validate enhanced configuration
    if not validate_config(enhanced_config)["valid"]:
        logger.error("Enhanced configuration validation failed")
        return 1
    
    # Production readiness check
    if environment == "production":
        deployer = ProductionDeployer(config)
        production_config = deployer.generate_deployment_config()
        
        readiness = validate_production_readiness(enhanced_config)
        
        if not readiness["ready"]:
            logger.error("Not ready for production")
            return 1
    
        logger.info("Deploying to production...")
        # Generate deployment command
        deploy_command = deployer.deploy_command()
        logger.info(f"Deployment command: {deploy_command}")
        
        # In production, would run deployment command here
        if environment != "production":
            # Start the bot with optimizations
            from src.main_enhanced import EnhancedArbitrageBot
            bot = EnhancedArbitrageBot(enhanced_config)
            
            try:
                asyncio.run(bot.start())
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                await bot.stop()
            except Exception as e:
                logger.error(f"Bot error: {e}")
                await bot.stop()
                raise
    
    else:
        # Development mode - start with basic config
        from src.main_enhanced import EnhancedArbitrageBot
        bot = EnhancedArbitrageBot(config)
        
        try:
            asyncio.run(bot.start())
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            await bot.stop()
        except Exception as e:
            logger.error(f"Bot error: {e}")
                await bot.stop()
                raise


if __name__ == "__main__":
    main()