"""
Main prediction service for ML-based data quality forecasting.

This module provides the primary interface for making predictions, coordinating
input validation, feature engineering, model loading, and prediction generation.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, date
from pathlib import Path
from typing import Union, Any
import structlog

from .validator import InputValidator
from .feature_pipeline import FeaturePipeline

logger = structlog.get_logger(__name__)


class Predictor:
    """
    Main prediction service for data quality pass percentage forecasting.
    
    Coordinates input validation, feature engineering, model loading, and prediction
    generation for individual prediction requests.
    """
    
    def __init__(self, model_path: Union[str, Path], historical_data: pd.DataFrame):
        """
        Initialize the Predictor service.
        
        Args:
            model_path: Path to the trained model file
            historical_data: DataFrame containing historical pass percentage data
        """
        self.model_path = Path(model_path)
        self.historical_data = historical_data.copy() if not historical_data.empty else pd.DataFrame()
        
        # Initialize components
        self.validator = InputValidator()
        self.feature_pipeline = FeaturePipeline(self.historical_data)
        
        # Model will be loaded on first prediction
        self._model = None
        
        logger.info("Predictor initialized",
                   model_path=str(self.model_path),
                   historical_records=len(self.historical_data))
    
    def load_model(self):
        """
        Load the trained model from file.
        
        Returns:
            The loaded model object
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info("Model loaded successfully", model_path=str(self.model_path))
            return model
            
        except Exception as e:
            logger.error("Failed to load model", 
                        model_path=str(self.model_path), 
                        error=str(e))
            raise
    
    def predict(
        self, 
        dataset_uuid: Any, 
        rule_code: Any, 
        business_date: Any
    ) -> float:
        """
        Make a prediction for the given inputs.
        
        Args:
            dataset_uuid: The dataset UUID
            rule_code: The rule code (string or integer)
            business_date: The business date (string, date, or datetime)
            
        Returns:
            Predicted pass percentage (0.0 to 100.0)
            
        Raises:
            ValueError: If inputs are invalid
            Exception: If prediction fails
        """
        # Validate inputs
        validated_inputs = self.validator.validate_all_inputs(
            dataset_uuid, rule_code, business_date
        )
        
        logger.debug("Prediction request received",
                    dataset_uuid=validated_inputs["dataset_uuid"],
                    rule_code=validated_inputs["rule_code"],
                    business_date=str(validated_inputs["business_date"]))
        
        # Load model if not already loaded
        if self._model is None:
            self._model = self.load_model()
        
        # Engineer features for prediction
        features = self.feature_pipeline.engineer_features_for_prediction(
            validated_inputs["dataset_uuid"],
            validated_inputs["rule_code"],
            validated_inputs["business_date"]
        )
        
        # Convert features to model input format
        model_input = self._prepare_model_input(features)
        
        # Make prediction
        try:
            raw_prediction = self._model.predict(model_input)[0]
            
            # Validate and clip prediction
            prediction = self._validate_and_clip_prediction(raw_prediction)
            
            logger.info("Prediction completed successfully",
                       dataset_uuid=validated_inputs["dataset_uuid"],
                       rule_code=validated_inputs["rule_code"],
                       business_date=str(validated_inputs["business_date"]),
                       prediction=prediction)
            
            return prediction
            
        except Exception as e:
            logger.error("Prediction failed",
                        dataset_uuid=validated_inputs["dataset_uuid"],
                        rule_code=validated_inputs["rule_code"],
                        business_date=str(validated_inputs["business_date"]),
                        error=str(e))
            raise
    
    def _prepare_model_input(self, features: dict) -> np.ndarray:
        """
        Prepare features for model input.
        
        CRITICAL FIX: This method must use the EXACT same feature selection logic as training
        to ensure feature count consistency (fixes 9 vs 11 feature mismatch).
        
        Args:
            features: Dictionary of engineered features
            
        Returns:
            Numpy array formatted for model input
        """
        # Use the SAME feature selection logic as training pipeline
        # This ensures we include both numeric AND categorical features
        # Previously this only included 9 numeric features, but training uses 11 (9 + 2 categorical)
        
        # Define expected feature order matching training (pipeline.py lines 140-145)
        feature_columns = [
            # Numeric features (time-based) - 4 features
            'day_of_week', 'day_of_month', 'week_of_year', 'month',
            # Numeric features (lag features) - 3 features  
            'lag_1_day', 'lag_2_day', 'lag_7_day',
            # Numeric features (moving averages) - 2 features
            'ma_3_day', 'ma_7_day',
            # Categorical features (these were missing!) - 2 features
            'dataset_uuid', 'rule_code'
        ]
        # Total: 11 features (9 numeric + 2 categorical)
        
        # Create a DataFrame with the features in the correct order for LightGBM
        # This approach ensures proper categorical handling
        feature_row = {}
        for col in feature_columns:
            value = features.get(col, np.nan)
            # Handle NaN values
            if pd.isna(value):
                value = 0.0 if col in ['day_of_week', 'day_of_month', 'week_of_year', 'month',
                                     'lag_1_day', 'lag_2_day', 'lag_7_day', 'ma_3_day', 'ma_7_day'] else value
            feature_row[col] = value
        
        # Create DataFrame for proper categorical handling
        feature_df = pd.DataFrame([feature_row])
        
        # Prepare categorical features (same as training)
        from .model_trainer import prepare_categorical_features_for_prediction
        categorical_cols = ['dataset_uuid', 'rule_code']
        prepared_df = prepare_categorical_features_for_prediction(feature_df, categorical_cols, self._model)
        
        # For LightGBM prediction, we need to pass the DataFrame directly
        # LightGBM handles categorical features automatically when they're in category dtype
        model_input = prepared_df[feature_columns]
        
        logger.debug("Model input prepared",
                    feature_count=len(feature_columns),
                    input_shape=model_input.shape)
        
        return model_input
    
    def _validate_and_clip_prediction(self, raw_prediction: float) -> float:
        """
        Validate and clip prediction to valid range.
        
        Args:
            raw_prediction: Raw model prediction
            
        Returns:
            Validated and clipped prediction
            
        Raises:
            ValueError: If prediction is invalid (NaN or infinite)
        """
        if pd.isna(raw_prediction) or np.isinf(raw_prediction):
            raise ValueError("Model returned invalid prediction (NaN or infinite)")
        
        # Clip to valid percentage range
        clipped_prediction = np.clip(raw_prediction, 0.0, 100.0)
        
        if clipped_prediction != raw_prediction:
            logger.warning("Prediction clipped to valid range",
                          raw_prediction=raw_prediction,
                          clipped_prediction=clipped_prediction)
        
        return float(clipped_prediction)
    
    def _validate_prediction_quality(self, predictions: np.ndarray) -> bool:
        """
        Validate prediction quality to detect constant predictions.
        
        This method implements US3.2: Prediction quality assurance with 
        constant prediction detection. It checks if predictions show
        sufficient variance to be considered valid.
        
        Args:
            predictions: Array of model predictions
            
        Returns:
            True if predictions show sufficient variance, False otherwise
        """
        if len(predictions) == 0:
            logger.warning("Empty prediction array provided")
            return False
        
        # Calculate prediction statistics
        pred_std = np.std(predictions)
        unique_predictions = len(np.unique(predictions))
        pred_range = np.max(predictions) - np.min(predictions)
        
        # Variance threshold: standard deviation should be > 0.1%
        variance_threshold = 0.1
        variance_sufficient = pred_std > variance_threshold
        
        # Uniqueness threshold: should have multiple unique values
        uniqueness_sufficient = unique_predictions > 1
        
        # Range threshold: range should be meaningful
        range_threshold = 1.0  # At least 1% range
        range_sufficient = pred_range > range_threshold
        
        overall_valid = variance_sufficient and uniqueness_sufficient and range_sufficient
        
        logger.info("Prediction quality validation",
                   prediction_count=len(predictions),
                   std_deviation=pred_std,
                   unique_values=unique_predictions,
                   prediction_range=pred_range,
                   variance_sufficient=variance_sufficient,
                   uniqueness_sufficient=uniqueness_sufficient,
                   range_sufficient=range_sufficient,
                   overall_valid=overall_valid)
        
        if not overall_valid:
            logger.warning("Prediction quality validation failed",
                          std=pred_std,
                          unique_count=unique_predictions,
                          range=pred_range)
        
        return overall_valid


class BaselinePredictor:
    """
    Baseline prediction service using historical averages as fallback.
    
    This class implements a simple fallback mechanism for when the main
    ML model fails or produces constant predictions. It uses historical
    averages per dataset-rule combination to provide reasonable predictions.
    """
    
    def __init__(self, historical_data: pd.DataFrame = None):
        """
        Initialize baseline predictor with historical data.
        
        Args:
            historical_data: DataFrame with historical pass percentage data
        """
        self.historical_data = historical_data if historical_data is not None else pd.DataFrame()
        self._baseline_cache = {}
        self._compute_baselines()
        
        logger.info("BaselinePredictor initialized",
                   historical_records=len(self.historical_data),
                   baseline_combinations=len(self._baseline_cache))
    
    def _compute_baselines(self):
        """Compute baseline predictions from historical data."""
        if self.historical_data.empty:
            return
        
        # Group by dataset and rule to compute averages
        if all(col in self.historical_data.columns for col in ['dataset_uuid', 'rule_code', 'pass_percentage']):
            baselines = self.historical_data.groupby(['dataset_uuid', 'rule_code'])['pass_percentage'].mean()
            self._baseline_cache = baselines.to_dict()
            
            logger.debug("Computed baseline predictions",
                        baseline_count=len(self._baseline_cache))
    
    def predict(self, dataset_uuid: str, rule_code: str) -> float:
        """
        Predict pass percentage using historical average.
        
        Args:
            dataset_uuid: Dataset identifier
            rule_code: Rule code
            
        Returns:
            Predicted pass percentage based on historical average
        """
        # Try to get specific baseline for this dataset-rule combination
        baseline_key = (dataset_uuid, rule_code)
        
        if baseline_key in self._baseline_cache:
            prediction = self._baseline_cache[baseline_key]
            logger.debug("Baseline prediction from specific combination",
                        dataset_uuid=dataset_uuid,
                        rule_code=rule_code,
                        prediction=prediction)
            return prediction
        
        # Fallback to rule-specific average
        rule_baselines = [v for k, v in self._baseline_cache.items() if k[1] == rule_code]
        if rule_baselines:
            prediction = np.mean(rule_baselines)
            logger.debug("Baseline prediction from rule average",
                        rule_code=rule_code,
                        prediction=prediction)
            return prediction
        
        # Fallback to dataset-specific average  
        dataset_baselines = [v for k, v in self._baseline_cache.items() if k[0] == dataset_uuid]
        if dataset_baselines:
            prediction = np.mean(dataset_baselines)
            logger.debug("Baseline prediction from dataset average",
                        dataset_uuid=dataset_uuid,
                        prediction=prediction)
            return prediction
        
        # Final fallback to global average or 50%
        if self._baseline_cache:
            prediction = np.mean(list(self._baseline_cache.values()))
            logger.debug("Baseline prediction from global average",
                        prediction=prediction)
            return prediction
        
        # Ultimate fallback
        prediction = 50.0
        logger.warning("Using ultimate fallback prediction",
                      dataset_uuid=dataset_uuid,
                      rule_code=rule_code,
                      prediction=prediction)
        return prediction