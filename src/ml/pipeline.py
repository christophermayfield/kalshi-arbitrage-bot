"""
Production Machine Learning Pipeline
Professional ML pipeline with automated training, deployment, and monitoring
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import pickle
import json
import asyncio
import threading
from pathlib import Path
import uuid
import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils.logging_utils import get_logger
from src.utils.config import Config
from src.core.enhanced_ml_features import EnhancedFeatureExtractor, FeatureScaler

logger = get_logger("ml_pipeline")


class ModelType(Enum):
    """Model types"""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ENSEMBLE = "ensemble"


class ModelStatus(Enum):
    """Model lifecycle status"""

    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    RETIRED = "retired"
    FAILED = "failed"


class TrainingMode(Enum):
    """Training modes"""

    BATCH = "batch"
    ONLINE = "online"
    INCREMENTAL = "incremental"


@dataclass
class ModelConfig:
    """Model configuration"""

    model_id: str
    model_type: ModelType
    model_class: str  # sklearn class name
    target_column: str

    # Training parameters
    training_mode: TrainingMode = TrainingMode.BATCH
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42

    # Model hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Feature configuration
    feature_columns: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)

    # Training schedule
    retrain_interval_hours: int = 24
    min_samples_for_retrain: int = 1000
    performance_threshold: float = 0.7

    # Validation criteria
    min_accuracy: float = 0.6
    min_r2_score: float = 0.5
    max_overfitting_ratio: float = 2.0

    # Deployment settings
    auto_deploy: bool = True
    rollback_on_degradation: bool = True
    performance_monitoring: bool = True


@dataclass
class TrainingResult:
    """Result of model training"""

    model_id: str
    training_timestamp: datetime
    training_duration_seconds: float

    # Performance metrics
    train_score: float = 0.0
    validation_score: float = 0.0
    test_score: float = 0.0
    cv_mean_score: float = 0.0
    cv_std_score: float = 0.0

    # Model statistics
    training_samples: int = 0
    validation_samples: int = 0
    feature_count: int = 0

    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # Validation results
    classification_report: Dict[str, Any] = field(default_factory=dict)
    regression_metrics: Dict[str, float] = field(default_factory=dict)

    # Status
    status: ModelStatus = ModelStatus.TRAINED
    error_message: Optional[str] = None

    # Model artifacts
    model_path: Optional[str] = None
    scaler_path: Optional[str] = None
    metadata_path: Optional[str] = None


@dataclass
class ModelMetadata:
    """Model metadata and versioning"""

    model_id: str
    version: int
    created_at: datetime
    updated_at: datetime

    # Model info
    model_type: ModelType
    model_class: str
    target_column: str

    # Performance
    latest_performance: Dict[str, float] = field(default_factory=dict)
    best_performance: Dict[str, float] = field(default_factory=dict)

    # Training history
    training_history: List[TrainingResult] = field(default_factory=list)

    # Deployment info
    is_deployed: bool = False
    deployed_version: Optional[int] = None
    deployment_timestamp: Optional[datetime] = None

    # Data info
    training_data_hash: Optional[str] = None
    feature_schema_hash: Optional[str] = None

    # Monitoring
    prediction_count: int = 0
    last_prediction_timestamp: Optional[datetime] = None
    performance_degradation: float = 0.0


class ModelRegistry:
    """Model registry for versioning and lifecycle management"""

    def __init__(self, registry_path: str = "data/models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, ModelMetadata] = {}
        self.deployed_models: Dict[str, int] = {}  # model_id -> version

        self._load_registry()

    def _load_registry(self) -> None:
        """Load model registry from disk"""
        try:
            registry_file = self.registry_path / "registry.json"
            if registry_file.exists():
                with open(registry_file, "r") as f:
                    registry_data = json.load(f)

                for model_id, model_data in registry_data.items():
                    self.models[model_id] = ModelMetadata(
                        model_id=model_data["model_id"],
                        version=model_data["version"],
                        created_at=datetime.fromisoformat(model_data["created_at"]),
                        updated_at=datetime.fromisoformat(model_data["updated_at"]),
                        model_type=ModelType(model_data["model_type"]),
                        model_class=model_data["model_class"],
                        target_column=model_data["target_column"],
                        latest_performance=model_data.get("latest_performance", {}),
                        best_performance=model_data.get("best_performance", {}),
                        training_history=[],
                        is_deployed=model_data.get("is_deployed", False),
                        deployed_version=model_data.get("deployed_version"),
                        deployment_timestamp=datetime.fromisoformat(
                            model_data["deployment_timestamp"]
                        )
                        if model_data.get("deployment_timestamp")
                        else None,
                        training_data_hash=model_data.get("training_data_hash"),
                        feature_schema_hash=model_data.get("feature_schema_hash"),
                        prediction_count=model_data.get("prediction_count", 0),
                        last_prediction_timestamp=datetime.fromisoformat(
                            model_data["last_prediction_timestamp"]
                        )
                        if model_data.get("last_prediction_timestamp")
                        else None,
                        performance_degradation=model_data.get(
                            "performance_degradation", 0.0
                        ),
                    )

                # Load deployed models
                self.deployed_models = registry_data.get("deployed_models", {})

                logger.info(f"Loaded {len(self.models)} models from registry")

        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")

    def _save_registry(self) -> None:
        """Save model registry to disk"""
        try:
            registry_data = {"models": {}, "deployed_models": self.deployed_models}

            for model_id, metadata in self.models.items():
                registry_data["models"][model_id] = {
                    "model_id": metadata.model_id,
                    "version": metadata.version,
                    "created_at": metadata.created_at.isoformat(),
                    "updated_at": metadata.updated_at.isoformat(),
                    "model_type": metadata.model_type.value,
                    "model_class": metadata.model_class,
                    "target_column": metadata.target_column,
                    "latest_performance": metadata.latest_performance,
                    "best_performance": metadata.best_performance,
                    "is_deployed": metadata.is_deployed,
                    "deployed_version": metadata.deployed_version,
                    "deployment_timestamp": metadata.deployment_timestamp.isoformat()
                    if metadata.deployment_timestamp
                    else None,
                    "training_data_hash": metadata.training_data_hash,
                    "feature_schema_hash": metadata.feature_schema_hash,
                    "prediction_count": metadata.prediction_count,
                    "last_prediction_timestamp": metadata.last_prediction_timestamp.isoformat()
                    if metadata.last_prediction_timestamp
                    else None,
                    "performance_degradation": metadata.performance_degradation,
                }

            registry_file = self.registry_path / "registry.json"
            with open(registry_file, "w") as f:
                json.dump(registry_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")

    def register_model(self, config: ModelConfig) -> ModelMetadata:
        """Register a new model"""
        try:
            if config.model_id in self.models:
                # Update existing model
                metadata = self.models[config.model_id]
                metadata.version += 1
                metadata.updated_at = datetime.now()
            else:
                # Create new model metadata
                metadata = ModelMetadata(
                    model_id=config.model_id,
                    version=1,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    model_type=config.model_type,
                    model_class=config.model_class,
                    target_column=config.target_column,
                )
                self.models[config.model_id] = metadata

            self._save_registry()
            logger.info(
                f"Registered model: {config.model_id} (version {metadata.version})"
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata"""
        return self.models.get(model_id)

    def get_deployed_model(self, model_id: str) -> Optional[Tuple[ModelMetadata, int]]:
        """Get deployed model with version"""
        if model_id not in self.models:
            return None

        metadata = self.models[model_id]
        if not metadata.is_deployed or metadata.deployed_version is None:
            return None

        return metadata, metadata.deployed_version

    def deploy_model(self, model_id: str, version: Optional[int] = None) -> bool:
        """Deploy a model version"""
        try:
            metadata = self.models.get(model_id)
            if not metadata:
                logger.error(f"Model not found: {model_id}")
                return False

            deploy_version = version if version is not None else metadata.version

            # Update deployment info
            metadata.is_deployed = True
            metadata.deployed_version = deploy_version
            metadata.deployment_timestamp = datetime.now()

            self.deployed_models[model_id] = deploy_version
            self._save_registry()

            logger.info(f"Deployed model: {model_id} (version {deploy_version})")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False

    def rollback_model(self, model_id: str) -> bool:
        """Rollback to previous version"""
        try:
            metadata = self.models.get(model_id)
            if not metadata or metadata.version <= 1:
                logger.error(f"Cannot rollback model: {model_id}")
                return False

            # Rollback to previous version
            previous_version = metadata.version - 1
            return self.deploy_model(model_id, previous_version)

        except Exception as e:
            logger.error(f"Failed to rollback model: {e}")
            return False


class ModelTrainer:
    """Model training with automated hyperparameter optimization"""

    def __init__(self, config: Config, registry: ModelRegistry):
        self.config = config
        self.registry = registry
        self.training_config = config.get("ml_pipeline", {})

        # Training state
        self.active_training: Dict[str, asyncio.Task] = {}
        self.training_history: List[TrainingResult] = []

        # Feature extractor
        self.feature_extractor = EnhancedFeatureExtractor()

        # Model factory
        self.model_factory = ModelFactory()

        # Configuration
        self.max_concurrent_training = self.training_config.get(
            "max_concurrent_training", 2
        )
        self.enable_hyperparameter_optimization = self.training_config.get(
            "enable_hyperparameter_optimization", True
        )

        logger.info("Model trainer initialized")

    async def train_model(
        self,
        config: ModelConfig,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
    ) -> TrainingResult:
        """Train a model"""
        try:
            # Register model if not exists
            metadata = self.registry.get_model(config.model_id)
            if not metadata:
                metadata = self.registry.register_model(config)

            # Check if already training
            if config.model_id in self.active_training:
                raise Exception(f"Model {config.model_id} is already training")

            # Start training
            training_task = asyncio.create_task(
                self._train_model_async(config, training_data, validation_data)
            )

            self.active_training[config.model_id] = training_task

            # Wait for completion
            result = await training_task

            # Clean up
            del self.active_training[config.model_id]

            # Update registry
            metadata.training_history.append(result)
            metadata.updated_at = datetime.now()
            self.registry._save_registry()

            # Auto-deploy if enabled
            if config.auto_deploy and result.status == ModelStatus.VALIDATED:
                self.registry.deploy_model(config.model_id)

            return result

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise

    async def _train_model_async(
        self,
        config: ModelConfig,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
    ) -> TrainingResult:
        """Async model training"""
        start_time = datetime.now()
        result = TrainingResult(
            model_id=config.model_id,
            training_timestamp=start_time,
            training_duration_seconds=0.0,
        )

        try:
            # Validate data
            if training_data.empty:
                raise ValueError("Training data is empty")

            if config.target_column not in training_data.columns:
                raise ValueError(
                    f"Target column {config.target_column} not found in training data"
                )

            # Prepare features and target
            X, y = self._prepare_data(training_data, config)

            # Split data
            if validation_data is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=config.test_size, random_state=config.random_state
                )
            else:
                X_train, y_train = X, y
                X_val, y_val = self._prepare_data(validation_data, config)

            result.training_samples = len(X_train)
            result.validation_samples = len(X_val)
            result.feature_count = X_train.shape[1]

            # Create model pipeline
            pipeline = self._create_pipeline(config)

            # Hyperparameter optimization
            if self.enable_hyperparameter_optimization:
                pipeline = await self._optimize_hyperparameters(
                    pipeline, X_train, y_train, config
                )

            # Train model
            pipeline.fit(X_train, y_train)

            # Evaluate performance
            train_score = pipeline.score(X_train, y_train)
            val_score = pipeline.score(X_val, y_val)

            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=config.cv_folds)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Update result
            result.train_score = train_score
            result.validation_score = val_score
            result.cv_mean_score = cv_mean
            result.cv_std_score = cv_std

            # Calculate feature importance
            result.feature_importance = self._calculate_feature_importance(
                pipeline, config
            )

            # Detailed evaluation
            if config.model_type == ModelType.CLASSIFICATION:
                y_pred = pipeline.predict(X_val)
                result.classification_report = classification_report(
                    y_val, y_pred, output_dict=True
                )
            else:
                y_pred = pipeline.predict(X_val)
                result.regression_metrics = {
                    "mse": mean_squared_error(y_val, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
                    "r2": r2_score(y_val, y_pred),
                }

            # Validate model
            result.status = self._validate_model(result, config)

            # Save model
            if result.status == ModelStatus.VALIDATED:
                await self._save_model(pipeline, config, result)

            # Update training duration
            result.training_duration_seconds = (
                datetime.now() - start_time
            ).total_seconds()

            logger.info(
                f"Model training completed: {config.model_id}, score: {val_score:.3f}"
            )

            return result

        except Exception as e:
            result.status = ModelStatus.FAILED
            result.error_message = str(e)
            result.training_duration_seconds = (
                datetime.now() - start_time
            ).total_seconds()

            logger.error(f"Model training failed: {e}")
            return result

    def _prepare_data(
        self, data: pd.DataFrame, config: ModelConfig
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target"""
        try:
            # Extract features
            if config.feature_columns:
                X = data[config.feature_columns]
            else:
                # Use all columns except target
                X = data.drop(columns=[config.target_column])

            # Extract target
            y = data[config.target_column]

            # Handle missing values
            X = X.fillna(X.mean())

            return X, y

        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise

    def _create_pipeline(self, config: ModelConfig) -> Pipeline:
        """Create sklearn pipeline"""
        try:
            # Create preprocessing steps
            numeric_transformer = StandardScaler()
            categorical_transformer = "passthrough"  # Simplified for now

            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=[("num", numeric_transformer, config.numerical_features)],
                remainder="passthrough",
            )

            # Create model
            model = self.model_factory.create_model(config)

            # Create pipeline
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

            return pipeline

        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            raise

    async def _optimize_hyperparameters(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: ModelConfig,
    ) -> Pipeline:
        """Optimize hyperparameters"""
        try:
            if not config.hyperparameters:
                return pipeline

            # Create parameter grid
            param_grid = {}

            # Add model parameters
            model_param_prefix = "model__"
            for param, values in config.hyperparameters.items():
                param_grid[model_param_prefix + param] = values

            # Create grid search
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=config.cv_folds,
                scoring="accuracy"
                if config.model_type == ModelType.CLASSIFICATION
                else "r2",
                n_jobs=-1,
                verbose=0,
            )

            # Fit grid search
            grid_search.fit(X_train, y_train)

            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.3f}")

            return grid_search.best_estimator_

        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            return pipeline

    def _calculate_feature_importance(
        self, pipeline: Pipeline, config: ModelConfig
    ) -> Dict[str, float]:
        """Calculate feature importance"""
        try:
            # Get trained model
            model = pipeline.named_steps["model"]

            # Check if model has feature importance
            if hasattr(model, "feature_importances_"):
                # Get feature names from preprocessor
                preprocessor = pipeline.named_steps["preprocessor"]
                feature_names = self._get_feature_names(preprocessor, config)

                # Map importance to feature names
                importance_dict = dict(zip(feature_names, model.feature_importances_))

                # Sort by importance
                importance_dict = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                )

                return importance_dict
            else:
                return {}

        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return {}

    def _get_feature_names(self, preprocessor, config: ModelConfig) -> List[str]:
        """Get feature names after preprocessing"""
        try:
            # Simplified - return configured feature names
            return config.feature_columns if config.feature_columns else []
        except Exception:
            return []

    def _validate_model(
        self, result: TrainingResult, config: ModelConfig
    ) -> ModelStatus:
        """Validate model performance"""
        try:
            # Check minimum performance thresholds
            if config.model_type == ModelType.CLASSIFICATION:
                if result.validation_score < config.min_accuracy:
                    return ModelStatus.FAILED
            else:  # Regression
                if result.regression_metrics.get("r2", 0) < config.min_r2_score:
                    return ModelStatus.FAILED

            # Check for overfitting
            overfitting_ratio = result.train_score / max(0.001, result.validation_score)
            if overfitting_ratio > config.max_overfitting_ratio:
                logger.warning(
                    f"Model may be overfitting: train/val ratio = {overfitting_ratio:.2f}"
                )

            return ModelStatus.VALIDATED

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return ModelStatus.FAILED

    async def _save_model(
        self, pipeline: Pipeline, config: ModelConfig, result: TrainingResult
    ) -> None:
        """Save trained model"""
        try:
            # Create model directory
            model_dir = Path(f"data/models/{config.model_id}")
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save pipeline
            model_path = (
                model_dir
                / f"model_v{self.registry.get_model(config.model_id).version}.joblib"
            )
            joblib.dump(pipeline, model_path)
            result.model_path = str(model_path)

            # Save scaler if exists
            if "preprocessor" in pipeline.named_steps:
                preprocessor = pipeline.named_steps["preprocessor"]
                if hasattr(preprocessor, "scalers_"):
                    scaler_path = (
                        model_dir
                        / f"scaler_v{self.registry.get_model(config.model_id).version}.joblib"
                    )
                    joblib.dump(preprocessor, scaler_path)
                    result.scaler_path = str(scaler_path)

            # Save metadata
            metadata_path = (
                model_dir
                / f"metadata_v{self.registry.get_model(config.model_id).version}.json"
            )
            metadata = {
                "model_config": config.__dict__,
                "training_result": {
                    "train_score": result.train_score,
                    "validation_score": result.validation_score,
                    "cv_mean_score": result.cv_mean_score,
                    "feature_importance": result.feature_importance,
                    "training_samples": result.training_samples,
                    "feature_count": result.feature_count,
                },
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            result.metadata_path = str(metadata_path)

            logger.info(f"Model saved: {config.model_id}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")


class ModelFactory:
    """Factory for creating sklearn models"""

    def __init__(self):
        self.model_registry = {
            "RandomForestClassifier": RandomForestClassifier,
            "RandomForestRegressor": RandomForestRegressor,
            "LogisticRegression": LogisticRegression,
            "LinearRegression": LinearRegression,
        }

    def create_model(self, config: ModelConfig):
        """Create a model instance"""
        try:
            model_class = self.model_registry.get(config.model_class)
            if not model_class:
                raise ValueError(f"Unknown model class: {config.model_class}")

            # Create model with hyperparameters
            if config.hyperparameters:
                model = model_class(**config.hyperparameters)
            else:
                model = model_class()

            return model

        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise


class ProductionMLPipeline:
    """Production ML pipeline with automated training and deployment"""

    def __init__(self, config: Config):
        self.config = config
        self.pipeline_config = config.get("ml_pipeline", {})

        # Components
        self.registry = ModelRegistry()
        self.trainer = ModelTrainer(config, self.registry)
        self.feature_extractor = EnhancedFeatureExtractor()

        # Pipeline state
        self.running = False
        self.training_scheduler: Optional[asyncio.Task] = None
        self.model_monitor: Optional[asyncio.Task] = None

        # Configuration
        self.enable_automated_training = self.pipeline_config.get(
            "enable_automated_training", True
        )
        self.training_check_interval_hours = self.pipeline_config.get(
            "training_check_interval_hours", 6
        )
        self.enable_performance_monitoring = self.pipeline_config.get(
            "enable_performance_monitoring", True
        )

        logger.info("Production ML pipeline initialized")

    async def start(self) -> None:
        """Start the ML pipeline"""
        try:
            self.running = True

            # Start automated training scheduler
            if self.enable_automated_training:
                self.training_scheduler = asyncio.create_task(
                    self._training_scheduler_loop()
                )

            # Start performance monitor
            if self.enable_performance_monitoring:
                self.model_monitor = asyncio.create_task(
                    self._performance_monitor_loop()
                )

            logger.info("Production ML pipeline started")

        except Exception as e:
            logger.error(f"Failed to start ML pipeline: {e}")
            raise

    async def stop(self) -> None:
        """Stop the ML pipeline"""
        try:
            self.running = False

            # Cancel tasks
            if self.training_scheduler:
                self.training_scheduler.cancel()
            if self.model_monitor:
                self.model_monitor.cancel()

            logger.info("Production ML pipeline stopped")

        except Exception as e:
            logger.error(f"Failed to stop ML pipeline: {e}")

    async def train_and_deploy_model(
        self,
        config: ModelConfig,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
    ) -> TrainingResult:
        """Train and deploy a model"""
        try:
            # Train model
            result = await self.trainer.train_model(
                config, training_data, validation_data
            )

            # Deploy if training successful
            if result.status == ModelStatus.VALIDATED and config.auto_deploy:
                deployed = self.registry.deploy_model(config.model_id)
                if deployed:
                    logger.info(f"Model deployed: {config.model_id}")

            return result

        except Exception as e:
            logger.error(f"Failed to train and deploy model: {e}")
            raise

    async def predict(self, model_id: str, features: Dict[str, Any]) -> Any:
        """Make prediction using deployed model"""
        try:
            # Get deployed model
            deployed_model = self.registry.get_deployed_model(model_id)
            if not deployed_model:
                raise Exception(f"No deployed model found: {model_id}")

            metadata, version = deployed_model

            # Load model
            model_dir = Path(f"data/models/{model_id}")
            model_path = model_dir / f"model_v{version}.joblib"

            if not model_path.exists():
                raise Exception(f"Model file not found: {model_path}")

            pipeline = joblib.load(model_path)

            # Prepare features
            feature_df = pd.DataFrame([features])

            # Make prediction
            prediction = pipeline.predict(feature_df)

            # Update prediction count
            metadata.prediction_count += 1
            metadata.last_prediction_timestamp = datetime.now()
            self.registry._save_registry()

            return prediction[0] if len(prediction) == 1 else prediction

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    async def _training_scheduler_loop(self) -> None:
        """Background loop for automated training"""
        while self.running:
            try:
                # Check models that need retraining
                await self._check_models_for_retraining()

                # Wait for next check
                await asyncio.sleep(self.training_check_interval_hours * 3600)

            except Exception as e:
                logger.error(f"Training scheduler error: {e}")
                await asyncio.sleep(300)  # 5 minutes on error

    async def _performance_monitor_loop(self) -> None:
        """Background loop for performance monitoring"""
        while self.running:
            try:
                # Monitor model performance
                await self._monitor_model_performance()

                # Wait for next check
                await asyncio.sleep(3600)  # 1 hour

            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(300)

    async def _check_models_for_retraining(self) -> None:
        """Check if any models need retraining"""
        try:
            for model_id, metadata in self.registry.models.items():
                # Check if model needs retraining
                if self._should_retrain_model(metadata):
                    logger.info(f"Model {model_id} needs retraining")

                    # Trigger retraining (simplified - would load new data)
                    # await self._retrain_model(model_id)

        except Exception as e:
            logger.error(f"Failed to check models for retraining: {e}")

    async def _monitor_model_performance(self) -> None:
        """Monitor model performance"""
        try:
            for model_id, metadata in self.registry.models.items():
                if metadata.is_deployed:
                    # Check performance degradation
                    degradation = self._calculate_performance_degradation(metadata)
                    metadata.performance_degradation = degradation

                    # Rollback if significant degradation
                    if degradation > 0.2:  # 20% degradation
                        logger.warning(
                            f"Model {model_id} performance degraded: {degradation:.1%}"
                        )

                        # Check rollback policy
                        model_config = self._get_model_config(model_id)
                        if model_config and model_config.rollback_on_degradation:
                            success = self.registry.rollback_model(model_id)
                            if success:
                                logger.info(f"Rolled back model: {model_id}")

                    self.registry._save_registry()

        except Exception as e:
            logger.error(f"Failed to monitor model performance: {e}")

    def _should_retrain_model(self, metadata: ModelMetadata) -> bool:
        """Check if model should be retrained"""
        try:
            # Check time-based retraining
            if metadata.updated_at:
                hours_since_update = (
                    datetime.now() - metadata.updated_at
                ).total_seconds() / 3600
                model_config = self._get_model_config(metadata.model_id)

                if (
                    model_config
                    and hours_since_update >= model_config.retrain_interval_hours
                ):
                    return True

            return False

        except Exception:
            return False

    def _calculate_performance_degradation(self, metadata: ModelMetadata) -> float:
        """Calculate performance degradation"""
        try:
            if not metadata.best_performance or not metadata.latest_performance:
                return 0.0

            # Get key performance metric
            best_perf = max(metadata.best_performance.values())
            latest_perf = max(metadata.latest_performance.values())

            if best_perf == 0:
                return 0.0

            degradation = (best_perf - latest_perf) / best_perf
            return max(0.0, degradation)

        except Exception:
            return 0.0

    def _get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration (simplified)"""
        # In production, this would load from config or database
        return None

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        try:
            return {
                "running": self.running,
                "automated_training": self.enable_automated_training,
                "performance_monitoring": self.enable_performance_monitoring,
                "registered_models": len(self.registry.models),
                "deployed_models": len(self.registry.deployed_models),
                "active_training": len(self.trainer.active_training),
                "model_registry": {
                    model_id: {
                        "version": metadata.version,
                        "type": metadata.model_type.value,
                        "deployed": metadata.is_deployed,
                        "performance": metadata.latest_performance,
                    }
                    for model_id, metadata in self.registry.models.items()
                },
            }

        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return {}


# Utility functions
def create_model_config(
    model_id: str, model_type: ModelType, model_class: str, target_column: str, **kwargs
) -> ModelConfig:
    """Create model configuration"""
    return ModelConfig(
        model_id=model_id,
        model_type=model_type,
        model_class=model_class,
        target_column=target_column,
        **kwargs,
    )


def create_production_pipeline(config: Config) -> ProductionMLPipeline:
    """Create and return production ML pipeline"""
    return ProductionMLPipeline(config)
