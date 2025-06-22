"""
Model Validator for ML pipeline.

This module provides model quality validation, drift detection capabilities,
and performance monitoring utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Validates ML model quality and monitors performance over time.
    
    This class provides comprehensive model validation including quality metrics,
    drift detection, and performance monitoring capabilities.
    """
    
    def __init__(self):
        """Initialize the model validator."""
        self.performance_history = {}
        self.validation_thresholds = {
            'mae_good': 3.0,
            'mae_fair': 7.0,
            'mae_poor': 15.0
        }
        logger.info("ModelValidator initialized")
    
    def validate_model_quality(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Validate model quality using various metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            thresholds: Custom thresholds for quality assessment
            
        Returns:
            Dictionary containing quality metrics and status
        """
        logger.info(f"Validating model quality for {len(y_true)} predictions")
        
        if thresholds:
            validation_thresholds = thresholds
        else:
            validation_thresholds = self.validation_thresholds
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Handle edge cases for R2 score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r2 = r2_score(y_true, y_pred)
                if np.isnan(r2) or np.isinf(r2):
                    r2 = 0.0
            except:
                r2 = 0.0
        
        # Determine quality status
        if mae <= validation_thresholds.get('mae_good', 3.0):
            quality_status = 'GOOD'
        elif mae <= validation_thresholds.get('mae_fair', 7.0):
            quality_status = 'FAIR'
        elif mae <= validation_thresholds.get('mae_poor', 15.0):
            quality_status = 'POOR'
        else:
            quality_status = 'NEEDS_RETRAINING'
        
        quality_report = {
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'quality_status': quality_status,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Model quality: {quality_status} (MAE: {mae:.3f})")
        
        return quality_report
    
    def detect_data_drift(self, train_data: pd.DataFrame, new_data: pd.DataFrame,
                         feature_columns: List[str]) -> Dict[str, Any]:
        """
        Detect data drift between training and new data.
        
        Args:
            train_data: Training dataset
            new_data: New dataset to compare
            feature_columns: List of feature columns to analyze
            
        Returns:
            Dictionary containing drift detection results
        """
        logger.info(f"Detecting data drift for {len(feature_columns)} features")
        
        drift_scores = {}
        affected_features = []
        missing_features = []
        
        for feature in feature_columns:
            if feature not in train_data.columns:
                missing_features.append(feature)
                continue
            if feature not in new_data.columns:
                missing_features.append(feature)
                continue
            
            try:
                if pd.api.types.is_numeric_dtype(train_data[feature]):
                    # Statistical drift detection for numeric features
                    train_mean = train_data[feature].mean()
                    new_mean = new_data[feature].mean()
                    train_std = train_data[feature].std()
                    
                    if train_std > 0:
                        drift_score = abs(new_mean - train_mean) / train_std
                    else:
                        drift_score = 0.0
                else:
                    # Categorical drift detection
                    train_categories = set(train_data[feature].unique())
                    new_categories = set(new_data[feature].unique())
                    
                    # Calculate Jaccard similarity
                    intersection = len(train_categories.intersection(new_categories))
                    union = len(train_categories.union(new_categories))
                    drift_score = 1.0 - (intersection / union if union > 0 else 0.0)
                    
                    # Boost drift score if completely new categories appear
                    if len(new_categories - train_categories) > 0:
                        drift_score = max(drift_score, 0.7)
                
                drift_scores[feature] = drift_score
                
                # Consider feature affected if drift score > 0.3
                if drift_score > 0.3:
                    affected_features.append(feature)
                    
            except Exception as e:
                logger.warning(f"Could not calculate drift for feature {feature}: {e}")
                drift_scores[feature] = 0.0
        
        # Overall drift score (average of individual scores)
        overall_drift_score = np.mean(list(drift_scores.values())) if drift_scores else 0.0
        drift_detected = overall_drift_score > 0.3 or len(affected_features) > 0
        
        drift_report = {
            'drift_detected': drift_detected,
            'drift_score': overall_drift_score,
            'affected_features': affected_features,
            'feature_drift_scores': drift_scores,
            'missing_features': missing_features,
            'detection_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Drift detection: {'DETECTED' if drift_detected else 'NOT DETECTED'} "
                   f"(score: {overall_drift_score:.3f})")
        
        return drift_report
    
    def monitor_performance(self, predictions: List[float], actuals: List[float], 
                          timestamp: Optional[str] = None) -> None:
        """
        Monitor and record model performance over time.
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            timestamp: Timestamp for this performance record
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        logger.info(f"Recording performance for {len(predictions)} predictions at {timestamp}")
        
        # Calculate performance metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        performance_record = {
            'mae': mae,
            'rmse': rmse,
            'prediction_count': len(predictions),
            'timestamp': timestamp
        }
        
        self.performance_history[timestamp] = performance_record
        
        logger.debug(f"Performance recorded: MAE={mae:.3f}, RMSE={rmse:.3f}")
    
    def get_performance_history(self) -> Dict[str, Dict[str, Any]]:
        """
        Get complete performance history.
        
        Returns:
            Dictionary of performance records indexed by timestamp
        """
        return self.performance_history.copy()
    
    def analyze_performance_trend(self) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Returns:
            Dictionary containing trend analysis results
        """
        logger.info("Analyzing performance trends")
        
        if len(self.performance_history) < 2:
            return {
                'trend_direction': 'INSUFFICIENT_DATA',
                'trend_significance': 0.0,
                'analysis_timestamp': datetime.now().isoformat()
            }
        
        # Get MAE values in chronological order
        sorted_history = sorted(self.performance_history.items(), key=lambda x: x[0])
        mae_values = [record[1]['mae'] for record in sorted_history]
        
        # Simple trend analysis: compare first half with second half
        half_point = len(mae_values) // 2
        first_half_avg = np.mean(mae_values[:half_point])
        second_half_avg = np.mean(mae_values[half_point:])
        
        trend_change = (second_half_avg - first_half_avg) / first_half_avg
        
        if trend_change > 0.1:
            trend_direction = 'DEGRADING'
        elif trend_change < -0.1:
            trend_direction = 'IMPROVING'
        else:
            trend_direction = 'STABLE'
        
        trend_report = {
            'trend_direction': trend_direction,
            'trend_significance': abs(trend_change),
            'first_half_avg_mae': first_half_avg,
            'second_half_avg_mae': second_half_avg,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Performance trend: {trend_direction} (significance: {abs(trend_change):.3f})")
        
        return trend_report
    
    def validate_model_metadata(self, model, expected_features: List[str]) -> Dict[str, Any]:
        """
        Validate model metadata including feature names and types.
        
        Stage 2 enhancement: Validates that model expects correct features
        and provides metadata consistency checking.
        
        Args:
            model: Trained model object
            expected_features: List of expected feature names
            
        Returns:
            Dictionary containing metadata validation results
        """
        logger.info(f"Validating model metadata for {len(expected_features)} expected features")
        
        validation_result = {
            'metadata_valid': True,
            'expected_features': expected_features,
            'feature_count_match': True,
            'validation_timestamp': datetime.now().isoformat(),
            'issues': []
        }
        
        # For LightGBM models, check feature count
        try:
            if hasattr(model, 'num_feature'):
                model_feature_count = model.num_feature
                expected_count = len(expected_features)
                
                if model_feature_count != expected_count:
                    validation_result['metadata_valid'] = False
                    validation_result['feature_count_match'] = False
                    validation_result['issues'].append(
                        f"Feature count mismatch: model expects {model_feature_count}, "
                        f"provided {expected_count}"
                    )
                    
            validation_result['model_feature_count'] = getattr(model, 'num_feature', None)
                    
        except Exception as e:
            validation_result['metadata_valid'] = False
            validation_result['issues'].append(f"Error validating model metadata: {str(e)}")
        
        logger.info(f"Model metadata validation: {'VALID' if validation_result['metadata_valid'] else 'INVALID'}")
        
        return validation_result