from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from prophet import Prophet
from darts import TimeSeries
from darts.models import TFTModel  # Using TFT as alternative to LSTM
from darts.dataprocessing.transformers import Scaler

from src.utils.logging_utils import get_logger
from src.utils.database import Database

logger = get_logger("predictive_models")


class ProphetForecaster:
    """Prophet-based price forecaster for short-term predictions"""

    def __init__(self, retrain_interval_hours: int = 24, prediction_horizon: int = 5):
        self.retrain_interval_hours = retrain_interval_hours
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.last_retrain = datetime.utcnow()
        self.db = Database()

    def _prepare_data(self, market_id: str) -> pd.DataFrame:
        """Prepare data for Prophet (requires ds and y columns)"""
        try:
            query = """
                SELECT timestamp as ds, mid_price as y 
                FROM market_data 
                WHERE market_id = ? 
                AND timestamp >= datetime('now', '-7 days')
                ORDER BY timestamp
            """
            df = self.db.query(query, (market_id,))
            if df.empty:
                logger.warning(f"No historical data found for market {market_id}")
                return pd.DataFrame()
            return df
        except Exception as e:
            logger.error(f"Error preparing data for market {market_id}: {e}")
            return pd.DataFrame()

    def retrain(self, market_id: str) -> bool:
        """Retrain Prophet model on historical data"""
        try:
            df = self._prepare_data(market_id)
            if df.empty:
                return False

            # Add seasonality components for prediction markets
            df["hour"] = pd.to_datetime(df["ds"]).dt.hour
            df["day_of_week"] = pd.to_datetime(df["ds"]).dt.dayofweek

            self.model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.01,
                seasonality_prior_scale=0.1,
            )

            self.model.fit(df)
            self.last_retrain = datetime.utcnow()

            logger.info(f"Successfully retrained Prophet model for market {market_id}")
            return True

        except Exception as e:
            logger.error(f"Error retraining Prophet model for market {market_id}: {e}")
            return False

    def predict(
        self, market_id: str, include_uncertainty: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Generate predictions using Prophet"""
        if not self.model or datetime.utcnow() - self.last_retrain > timedelta(
            hours=self.retrain_interval_hours
        ):
            if not self.retrain(market_id):
                return None

        try:
            future = self.model.make_future_dataframe(
                periods=self.prediction_horizon, freq="min"
            )
            forecast = self.model.predict(future)

            # Get the last prediction
            prediction = forecast.iloc[-1]

            result = {
                "market_id": market_id,
                "predicted_price": float(prediction["yhat"]),
                "prediction_timestamp": datetime.utcnow().isoformat(),
                "method": "prophet",
                "horizon_minutes": self.prediction_horizon,
                "model_last_retrain": self.last_retrain.isoformat(),
            }

            if include_uncertainty:
                result["lower_bound"] = float(prediction["yhat_lower"])
                result["upper_bound"] = float(prediction["yhat_upper"])
                result["uncertainty"] = float(
                    prediction["yhat_upper"] - prediction["yhat_lower"]
                )

            return result

        except Exception as e:
            logger.error(f"Error generating prediction for market {market_id}: {e}")
            return None


class TFTForecaster:
    """TFT-based forecaster for complex price patterns (alternative to LSTM)"""

    def __init__(self, sequence_length: int = 20, prediction_horizon: int = 5):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = Scaler()
        self.trained = False
        self.db = Database()

    def _prepare_series_data(self, market_id: str) -> Optional[TimeSeries]:
        """Prepare data for TFT using Darts"""
        try:
            query = """
                SELECT timestamp, mid_price 
                FROM market_data 
                WHERE market_id = ? 
                AND timestamp >= datetime('now', '-30 days')
                ORDER BY timestamp
            """
            df = self.db.query(query, (market_id,))
            if df.empty or len(df) < self.sequence_length + 10:
                logger.warning(
                    f"Insufficient data for TFT training on market {market_id}"
                )
                return None

            # Convert to TimeSeries
            series = TimeSeries.from_dataframe(df, "timestamp", "mid_price")
            return series

        except Exception as e:
            logger.error(f"Error preparing TFT data for market {market_id}: {e}")
            return None

    def train(self, market_id: str) -> bool:
        """Train TFT model"""
        try:
            series = self._prepare_series_data(market_id)
            if not series:
                return False

            # Scale the data
            scaled_series = self.scaler.fit_transform(series)

            # Split data
            train, val = scaled_series.split_before(0.8)

            # Create and train model
            self.model = TFTModel(
                input_chunk_length=self.sequence_length,
                output_chunk_length=self.prediction_horizon,
                n_epochs=20,  # Reduced for faster training
                random_state=42,
                pl_trainer_kwargs={"accelerator": "cpu", "devices": "auto"},
            )

            self.model.fit(train, val_series=val)
            self.trained = True

            logger.info(f"Successfully trained TFT model for market {market_id}")
            return True

        except Exception as e:
            logger.error(f"Error training TFT model for market {market_id}: {e}")
            return False

    def predict(
        self, market_id: str, recent_prices: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Generate TFT predictions"""
        if not self.trained or not self.model:
            return None

        try:
            # Create recent series from provided prices
            if len(recent_prices) < self.sequence_length:
                logger.warning(
                    f"Insufficient recent prices for TFT prediction on market {market_id}"
                )
                return None

            recent_series = TimeSeries.from_values(
                np.array(recent_prices).reshape(-1, 1)
            )
            scaled_series = self.scaler.transform(recent_series)

            # Generate predictions
            prediction = self.model.predict(
                self.prediction_horizon, series=scaled_series
            )
            unscaled_prediction = self.scaler.inverse_transform(prediction)

            result = {
                "market_id": market_id,
                "predicted_price": float(unscaled_prediction.values[-1][0]),
                "prediction_timestamp": datetime.utcnow().isoformat(),
                "method": "tft",
                "horizon_minutes": self.prediction_horizon,
                "confidence": float(unscaled_prediction.values[-1][0])
                / float(unscaled_prediction.values[0][0]),  # Simple confidence metric
            }

            return result

        except Exception as e:
            logger.error(f"Error generating TFT prediction for market {market_id}: {e}")
            return None

            # Convert to TimeSeries
            series = TimeSeries.from_dataframe(df, "timestamp", "mid_price")
            return series

        except Exception as e:
            logger.error(f"Error preparing LSTM data for market {market_id}: {e}")
            return None

    def train(self, market_id: str) -> bool:
        """Train LSTM model"""
        try:
            series = self._prepare_series_data(market_id)
            if not series:
                return False

            # Scale the data
            scaled_series = self.scaler.fit_transform(series)

            # Split data
            train, val = scaled_series.split_before(0.8)

            # Create and train model
            self.model = LSTMModel(
                input_chunk_length=self.sequence_length,
                output_chunk_length=self.prediction_horizon,
                n_epochs=50,
                random_state=42,
                pl_trainer_kwargs={"accelerator": "cpu", "devices": "auto"},
            )

            self.model.fit(train, val_series=val)
            self.trained = True

            logger.info(f"Successfully trained LSTM model for market {market_id}")
            return True

        except Exception as e:
            logger.error(f"Error training LSTM model for market {market_id}: {e}")
            return False

    def predict(
        self, market_id: str, recent_prices: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Generate LSTM predictions"""
        if not self.trained or not self.model:
            return None

        try:
            # Create recent series from provided prices
            if len(recent_prices) < self.sequence_length:
                logger.warning(
                    f"Insufficient recent prices for LSTM prediction on market {market_id}"
                )
                return None

            recent_series = TimeSeries.from_values(
                np.array(recent_prices).reshape(-1, 1)
            )
            scaled_series = self.scaler.transform(recent_series)

            # Generate predictions
            prediction = self.model.predict(
                self.prediction_horizon, series=scaled_series
            )
            unscaled_prediction = self.scaler.inverse_transform(prediction)

            result = {
                "market_id": market_id,
                "predicted_price": float(unscaled_prediction.values[-1][0]),
                "prediction_timestamp": datetime.utcnow().isoformat(),
                "method": "lstm",
                "horizon_minutes": self.prediction_horizon,
                "confidence": float(unscaled_prediction.values[-1][0])
                / float(unscaled_prediction.values[0][0]),  # Simple confidence metric
            }

            return result

        except Exception as e:
            logger.error(
                f"Error generating LSTM prediction for market {market_id}: {e}"
            )
            return None


class EnsembleForecaster:
    """Ensemble of multiple forecasting models"""

    def __init__(self):
        self.prophet = ProphetForecaster()
        self.tft = TFTForecaster()
        self.db = Database()

    def get_predictions(
        self, market_id: str, recent_prices: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Get predictions from all available models"""
        predictions = []

        # Prophet prediction
        prophet_pred = self.prophet.predict(market_id)
        if prophet_pred:
            predictions.append(prophet_pred)

        # TFT prediction (if recent prices provided)
        if recent_prices:
            tft_pred = self.tft.predict(market_id, recent_prices)
            if tft_pred:
                predictions.append(tft_pred)

        return predictions

    def get_weighted_prediction(
        self, market_id: str, recent_prices: Optional[List[float]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get weighted ensemble prediction"""
        predictions = self.get_predictions(market_id, recent_prices)

        if not predictions:
            return None

        # Weight predictions (prophet gets higher weight due to interpretability)
        weights = {"prophet": 0.7, "lstm": 0.3}

        weighted_prediction = {
            "market_id": market_id,
            "predicted_price": 0,
            "lower_bound": float("inf"),
            "upper_bound": 0,
            "uncertainty": 0,
            "confidence": 0,
            "prediction_timestamp": datetime.utcnow().isoformat(),
            "method": "ensemble",
        }

        total_weight = 0
        price_sum = 0
        confidence_sum = 0

        for pred in predictions:
            method = pred["method"]
            weight = weights.get(method, 0.5)
            total_weight += weight

            price_sum += pred["predicted_price"] * weight
            confidence_sum += pred.get("confidence", 0.5) * weight

            # Aggregate bounds
            if (
                "lower_bound" in pred
                and pred["lower_bound"] < weighted_prediction["lower_bound"]
            ):
                weighted_prediction["lower_bound"] = pred["lower_bound"]
            if (
                "upper_bound" in pred
                and pred["upper_bound"] > weighted_prediction["upper_bound"]
            ):
                weighted_prediction["upper_bound"] = pred["upper_bound"]
            if (
                "uncertainty" in pred
                and pred["uncertainty"] > weighted_prediction["uncertainty"]
            ):
                weighted_prediction["uncertainty"] = pred["uncertainty"]

        if total_weight > 0:
            weighted_prediction["predicted_price"] = price_sum / total_weight
            weighted_prediction["confidence"] = confidence_sum / total_weight

        return weighted_prediction


async def get_arbitrage_timing_signal(
    market_id: str,
    current_spread: float,
    recent_prices: List[float],
    prediction_horizon_minutes: int = 5,
) -> Dict[str, Any]:
    """
    Get timing signal for arbitrage based on predictions
    Returns signal indicating whether to enter/hold/wait
    """
    try:
        forecaster = EnsembleForecaster()
        prediction = forecaster.get_weighted_prediction(market_id, recent_prices)

        if not prediction:
            return {
                "signal": "wait",
                "reason": "No predictions available",
                "confidence": 0,
            }

        predicted_price = prediction["predicted_price"]
        confidence = prediction.get("confidence", 0.5)

        # Determine timing signal based on predicted price movement
        if predicted_price > current_spread * 1.02:  # Predicted increase
            signal = "enter_buy"
            reason = "Predicted price increase favors buy-side arbitrage"
        elif predicted_price < current_spread * 0.98:  # Predicted decrease
            signal = "enter_sell"
            reason = "Predicted price decrease favors sell-side arbitrage"
        else:
            signal = "wait"
            reason = "Predicted price movement neutral"

        return {
            "signal": signal,
            "reason": reason,
            "confidence": confidence,
            "predicted_price": predicted_price,
            "current_spread": current_spread,
            "market_id": market_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error generating timing signal for market {market_id}: {e}")
        return {
            "signal": "wait",
            "reason": f"Error generating signal: {str(e)}",
            "confidence": 0,
        }
