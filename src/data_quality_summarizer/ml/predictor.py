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
        
        Args:
            features: Dictionary of engineered features
            
        Returns:
            Numpy array formatted for model input
        """
        # Define expected feature order (based on training)
        feature_columns = [
            'day_of_week', 'day_of_month', 'week_of_year', 'month',
            'lag_1_day', 'lag_2_day', 'lag_7_day',
            'ma_3_day', 'ma_7_day'
        ]
        
        # Extract numeric features in consistent order
        feature_values = []
        for col in feature_columns:
            value = features.get(col, np.nan)
            # Handle NaN values
            if pd.isna(value):
                value = 0.0  # Default value for missing features
            feature_values.append(float(value))
        
        # Convert to 2D array for model input (single row)
        model_input = np.array([feature_values])
        
        logger.debug("Model input prepared",
                    feature_count=len(feature_values),
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