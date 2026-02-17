"""Strategy versioning and rollback system for arbitrage bot."""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
import shutil
import pathlib
import logging
from pathlib import Path

from src.utils.config import Config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VersionStatus(Enum):
    """Version status enumeration."""

    DRAFT = "draft"
    TESTING = "testing"
    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"
    DEPRECATED = "deprecated"


class VersionType(Enum):
    """Version type enumeration."""

    MAJOR = "major"  # Breaking changes
    MINOR = "minor"  # New features
    PATCH = "patch"  # Bug fixes
    HOTFIX = "hotfix"  # Emergency fixes


@dataclass
class StrategyVersion:
    """Strategy version metadata."""

    version: str
    strategy_name: str
    version_type: VersionType
    status: VersionStatus
    created_at: datetime
    created_by: str
    description: str
    config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    backtest_results: Dict[str, Any]
    dependencies: Dict[str, str]
    rollback_version: Optional[str] = None
    parent_version: Optional[str] = None
    tags: List[str] = None
    changelog: str = ""

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class StrategyVersionManager:
    """Manages strategy versioning and rollback capabilities."""

    def __init__(self, base_path: str = "strategy_versions"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.config = Config()
        self.versions_db_path = self.base_path / "versions.json"
        self.active_versions_path = self.base_path / "active_versions.json"

        # Load existing version database
        self._load_versions()
        self._load_active_versions()

    def _load_versions(self):
        """Load version database from disk."""
        if self.versions_db_path.exists():
            try:
                with open(self.versions_db_path, "r") as f:
                    data = json.load(f)
                    self.versions = {
                        version_id: self._dict_to_version(version_data)
                        for version_id, version_data in data.items()
                    }
            except Exception as e:
                logger.error(f"Failed to load versions database: {e}")
                self.versions = {}
        else:
            self.versions = {}

    def _load_active_versions(self):
        """Load active versions mapping."""
        if self.active_versions_path.exists():
            try:
                with open(self.active_versions_path, "r") as f:
                    self.active_versions = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load active versions: {e}")
                self.active_versions = {}
        else:
            self.active_versions = {}

    def _save_versions(self):
        """Save version database to disk."""
        try:
            data = {
                version_id: self._version_to_dict(version)
                for version_id, version in self.versions.items()
            }
            with open(self.versions_db_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save versions database: {e}")

    def _save_active_versions(self):
        """Save active versions mapping to disk."""
        try:
            with open(self.active_versions_path, "w") as f:
                json.dump(self.active_versions, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save active versions: {e}")

    def _version_to_dict(self, version: StrategyVersion) -> Dict[str, Any]:
        """Convert version object to dictionary."""
        data = asdict(version)
        data["version_type"] = version.version_type.value
        data["status"] = version.status.value
        return data

    def _dict_to_version(self, data: Dict[str, Any]) -> StrategyVersion:
        """Convert dictionary to version object."""
        data["version_type"] = VersionType(data["version_type"])
        data["status"] = VersionStatus(data["status"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return StrategyVersion(**data)

    def create_version(
        self,
        strategy_name: str,
        config: Dict[str, Any],
        version_type: VersionType,
        description: str,
        created_by: str,
        parent_version: Optional[str] = None,
        tags: List[str] = None,
        changelog: str = "",
    ) -> str:
        """Create a new strategy version."""
        try:
            # Generate version number
            version = self._generate_version_number(strategy_name, version_type)
            version_id = f"{strategy_name}:{version}"

            # Get dependencies from config
            dependencies = self._extract_dependencies(config)

            # Create version object
            strategy_version = StrategyVersion(
                version=version,
                strategy_name=strategy_name,
                version_type=version_type,
                status=VersionStatus.DRAFT,
                created_at=datetime.utcnow(),
                created_by=created_by,
                description=description,
                config=config,
                performance_metrics={},
                backtest_results={},
                dependencies=dependencies,
                parent_version=parent_version,
                tags=tags or [],
                changelog=changelog,
            )

            # Store version
            self.versions[version_id] = strategy_version

            # Save version files
            self._save_version_files(strategy_version)
            self._save_versions()

            logger.info(f"Created new version: {version_id}")
            return version_id

        except Exception as e:
            logger.error(f"Failed to create version: {e}")
            raise

    def _generate_version_number(
        self, strategy_name: str, version_type: VersionType
    ) -> str:
        """Generate next version number for strategy."""
        # Get existing versions for this strategy
        existing_versions = [
            v for v in self.versions.values() if v.strategy_name == strategy_name
        ]

        if not existing_versions:
            return "1.0.0"

        # Sort versions by semantic version
        latest_version = max(
            existing_versions, key=lambda v: tuple(map(int, v.version.split(".")))
        )
        major, minor, patch = map(int, latest_version.version.split("."))

        # Increment based on version type
        if version_type == VersionType.MAJOR:
            major += 1
            minor = 0
            patch = 0
        elif version_type == VersionType.MINOR:
            minor += 1
            patch = 0
        else:  # PATCH or HOTFIX
            patch += 1

        return f"{major}.{minor}.{patch}"

    def _extract_dependencies(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Extract dependencies from strategy configuration."""
        dependencies = {}

        # Extract model dependencies
        if "ml_models" in config:
            for model_name, model_config in config["ml_models"].items():
                if "model_id" in model_config:
                    dependencies[f"ml_model:{model_name}"] = model_config["model_id"]

        # Extract data source dependencies
        if "data_sources" in config:
            for source in config["data_sources"]:
                dependencies[f"data_source:{source}"] = "latest"

        # Extract exchange dependencies
        if "exchanges" in config:
            for exchange in config["exchanges"]:
                dependencies[f"exchange:{exchange}"] = "latest"

        return dependencies

    def _save_version_files(self, version: StrategyVersion):
        """Save version-specific files."""
        version_dir = self.base_path / version.strategy_name / version.version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_path = version_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(version.config, f, indent=2, default=str)

        # Save metadata
        metadata_path = version_dir / "metadata.json"
        metadata = self._version_to_dict(version)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Calculate and save file hash
        file_hash = self._calculate_version_hash(version_dir)
        hash_path = version_dir / "hash.txt"
        with open(hash_path, "w") as f:
            f.write(file_hash)

    def _calculate_version_hash(self, version_dir: Path) -> str:
        """Calculate hash of version files."""
        hash_md5 = hashlib.md5()

        for file_path in version_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "hash.txt":
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def promote_version(self, version_id: str, target_status: VersionStatus) -> bool:
        """Promote version to new status."""
        try:
            if version_id not in self.versions:
                logger.error(f"Version {version_id} not found")
                return False

            version = self.versions[version_id]

            # Validate promotion path
            if not self._validate_promotion(version.status, target_status):
                logger.error(
                    f"Invalid promotion from {version.status} to {target_status}"
                )
                return False

            # Update status
            old_status = version.status
            version.status = target_status

            # If promoting to ACTIVE, update active versions
            if target_status == VersionStatus.ACTIVE:
                self.active_versions[version.strategy_name] = version_id

            # Save changes
            self._save_versions()
            self._save_active_versions()

            logger.info(f"Promoted {version_id} from {old_status} to {target_status}")
            return True

        except Exception as e:
            logger.error(f"Failed to promote version {version_id}: {e}")
            return False

    def _validate_promotion(
        self, current_status: VersionStatus, target_status: VersionStatus
    ) -> bool:
        """Validate if promotion is allowed."""
        valid_transitions = {
            VersionStatus.DRAFT: [VersionStatus.TESTING, VersionStatus.DEPRECATED],
            VersionStatus.TESTING: [
                VersionStatus.ACTIVE,
                VersionStatus.DRAFT,
                VersionStatus.DEPRECATED,
            ],
            VersionStatus.ACTIVE: [VersionStatus.ROLLED_BACK, VersionStatus.DEPRECATED],
            VersionStatus.ROLLED_BACK: [
                VersionStatus.DRAFT,
                VersionStatus.TESTING,
                VersionStatus.DEPRECATED,
            ],
            VersionStatus.DEPRECATED: [],
        }

        return target_status in valid_transitions.get(current_status, [])

    def rollback_strategy(
        self, strategy_name: str, target_version: Optional[str] = None
    ) -> bool:
        """Rollback strategy to previous version."""
        try:
            # Get current active version
            current_version_id = self.active_versions.get(strategy_name)
            if not current_version_id:
                logger.error(f"No active version found for strategy {strategy_name}")
                return False

            if target_version:
                # Rollback to specific version
                if target_version not in self.versions:
                    logger.error(f"Target version {target_version} not found")
                    return False
            else:
                # Find previous version
                target_version = self._find_previous_version(
                    strategy_name, current_version_id
                )
                if not target_version:
                    logger.error(f"No previous version found for rollback")
                    return False

            # Update versions
            current_version = self.versions[current_version_id]
            target_version_obj = self.versions[target_version]

            current_version.status = VersionStatus.ROLLED_BACK
            current_version.rollback_version = target_version

            # Activate target version
            target_version_obj.status = VersionStatus.ACTIVE
            self.active_versions[strategy_name] = target_version

            # Save changes
            self._save_versions()
            self._save_active_versions()

            logger.info(
                f"Rolled back {strategy_name} from {current_version_id} to {target_version}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to rollback strategy {strategy_name}: {e}")
            return False

    def _find_previous_version(
        self, strategy_name: str, current_version_id: str
    ) -> Optional[str]:
        """Find previous version for rollback."""
        strategy_versions = [
            (v_id, v)
            for v_id, v in self.versions.items()
            if v.strategy_name == strategy_name and v_id != current_version_id
        ]

        if not strategy_versions:
            return None

        # Sort by version number and return the previous one
        strategy_versions.sort(
            key=lambda x: tuple(map(int, x[1].version.split("."))), reverse=True
        )
        return strategy_versions[0][0]

    def get_version_history(self, strategy_name: str) -> List[StrategyVersion]:
        """Get version history for a strategy."""
        return [
            version
            for version in self.versions.values()
            if version.strategy_name == strategy_name
        ]

    def get_active_version(self, strategy_name: str) -> Optional[StrategyVersion]:
        """Get currently active version for a strategy."""
        version_id = self.active_versions.get(strategy_name)
        if version_id and version_id in self.versions:
            return self.versions[version_id]
        return None

    def list_versions(
        self, status: Optional[VersionStatus] = None
    ) -> List[StrategyVersion]:
        """List all versions, optionally filtered by status."""
        versions = list(self.versions.values())
        if status:
            versions = [v for v in versions if v.status == status]
        return versions

    def compare_versions(self, version_id_1: str, version_id_2: str) -> Dict[str, Any]:
        """Compare two versions and return differences."""
        try:
            if version_id_1 not in self.versions or version_id_2 not in self.versions:
                raise ValueError("One or both versions not found")

            v1 = self.versions[version_id_1]
            v2 = self.versions[version_id_2]

            comparison = {
                "version_1": {
                    "id": version_id_1,
                    "version": v1.version,
                    "status": v1.status.value,
                    "created_at": v1.created_at.isoformat(),
                },
                "version_2": {
                    "id": version_id_2,
                    "version": v2.version,
                    "status": v2.status.value,
                    "created_at": v2.created_at.isoformat(),
                },
                "differences": {
                    "config_changes": self._compare_configs(v1.config, v2.config),
                    "performance_diff": self._compare_performance(
                        v1.performance_metrics, v2.performance_metrics
                    ),
                    "dependency_changes": self._compare_dependencies(
                        v1.dependencies, v2.dependencies
                    ),
                },
            }

            return comparison

        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            raise

    def _compare_configs(
        self, config1: Dict[str, Any], config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare configuration differences."""
        changes = {}

        all_keys = set(config1.keys()) | set(config2.keys())

        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)

            if val1 != val2:
                changes[key] = {"from": val1, "to": val2}

        return changes

    def _compare_performance(
        self, perf1: Dict[str, float], perf2: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Compare performance metrics."""
        differences = {}

        all_metrics = set(perf1.keys()) | set(perf2.keys())

        for metric in all_metrics:
            val1 = perf1.get(metric, 0.0)
            val2 = perf2.get(metric, 0.0)

            if val1 != val2:
                differences[metric] = {
                    "version_1": val1,
                    "version_2": val2,
                    "difference": val2 - val1,
                    "percent_change": ((val2 - val1) / val1 * 100) if val1 != 0 else 0,
                }

        return differences

    def _compare_dependencies(
        self, deps1: Dict[str, str], deps2: Dict[str, str]
    ) -> Dict[str, Dict[str, str]]:
        """Compare dependencies."""
        changes = {}

        all_deps = set(deps1.keys()) | set(deps2.keys())

        for dep in all_deps:
            ver1 = deps1.get(dep)
            ver2 = deps2.get(dep)

            if ver1 != ver2:
                changes[dep] = {"version_1": ver1, "version_2": ver2}

        return changes

    def delete_version(self, version_id: str, force: bool = False) -> bool:
        """Delete a version (only if not active or deprecated)."""
        try:
            if version_id not in self.versions:
                logger.error(f"Version {version_id} not found")
                return False

            version = self.versions[version_id]

            # Check if version can be deleted
            if not force:
                if version.status in [VersionStatus.ACTIVE, VersionStatus.ROLLED_BACK]:
                    logger.error(
                        f"Cannot delete active/rolled back version {version_id}"
                    )
                    return False

                # Check if it's parent of any active version
                for active_version_id in self.active_versions.values():
                    if active_version_id in self.versions:
                        active_version = self.versions[active_version_id]
                        if active_version.parent_version == version_id:
                            logger.error(
                                f"Cannot delete version {version_id} as it's parent of active version {active_version_id}"
                            )
                            return False

            # Remove from active versions if present
            if version.strategy_name in self.active_versions:
                if self.active_versions[version.strategy_name] == version_id:
                    del self.active_versions[version.strategy_name]

            # Remove version files
            version_dir = self.base_path / version.strategy_name / version.version
            if version_dir.exists():
                shutil.rmtree(version_dir)

            # Remove from versions database
            del self.versions[version_id]

            # Save changes
            self._save_versions()
            self._save_active_versions()

            logger.info(f"Deleted version {version_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {e}")
            return False

    def cleanup_old_versions(self, strategy_name: str, keep_count: int = 10) -> int:
        """Clean up old versions, keeping only the most recent ones."""
        try:
            versions = [
                v for v in self.versions.values() if v.strategy_name == strategy_name
            ]

            # Sort by creation date (newest first)
            versions.sort(key=lambda v: v.created_at, reverse=True)

            # Keep active versions and recent ones
            to_keep = []
            to_delete = []

            for version in versions:
                if version.status in [VersionStatus.ACTIVE, VersionStatus.ROLLED_BACK]:
                    to_keep.append(version)
                elif len(to_keep) < keep_count:
                    to_keep.append(version)
                else:
                    to_delete.append(version)

            # Delete old versions
            deleted_count = 0
            for version in to_delete:
                version_id = f"{version.strategy_name}:{version.version}"
                if self.delete_version(version_id):
                    deleted_count += 1

            logger.info(f"Cleaned up {deleted_count} old versions for {strategy_name}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old versions: {e}")
            return 0


# Strategy versioning decorator for automatic versioning
def auto_version(
    strategy_name: str,
    version_type: VersionType = VersionType.PATCH,
    description: str = "",
    tags: List[str] = None,
):
    """Decorator to automatically version strategy configurations."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Execute the function to get config
            config = func(*args, **kwargs)

            # Create version
            version_manager = StrategyVersionManager()
            version_manager.create_version(
                strategy_name=strategy_name,
                config=config,
                version_type=version_type,
                description=description,
                created_by="system",
                tags=tags,
            )

            return config

        return wrapper

    return decorator
