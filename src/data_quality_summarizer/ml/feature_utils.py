"""
Feature consistency utilities for Stage 3 implementation.

This module provides utilities to ensure consistent feature handling
between training and prediction pipelines, addressing the critical
9 vs 11 feature mismatch issue identified in the PRD.

Key functionality:
- Standard feature list definition
- Feature alignment validation
- Categorical feature preparation consistency
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger(__name__)


class FeatureConsistency:
    """
    Utility class for maintaining feature consistency between training and prediction.
    
    This class addresses US3.1: Consistent feature handling between training and prediction.
    It ensures that both training and prediction pipelines use the exact same feature
    set and categorical handling logic.
    """
    
    def __init__(self):
        """Initialize the FeatureConsistency utility."""
        self._standard_features = self._define_standard_features()
        logger.debug("FeatureConsistency initialized", feature_count=len(self._standard_features))
    
    def _define_standard_features(self) -> List[str]:
        """
        Define the canonical list of 11 features used throughout the ML pipeline.
        
        This is the authoritative source for feature ordering and selection,
        preventing mismatches between training and prediction.
        
        Returns:
            List of 11 feature names in the exact order expected by LightGBM
        """
        return [
            # Time-based numeric features (4 features)
            'day_of_week',      # 1-7
            'day_of_month',     # 1-31
            'week_of_year',     # 1-53
            'month',            # 1-12
            
            # Lag-based numeric features (3 features)
            'lag_1_day',        # 1-day lag pass percentage
            'lag_2_day',        # 2-day lag pass percentage
            'lag_7_day',        # 7-day lag pass percentage
            
            # Moving average numeric features (2 features)
            'ma_3_day',         # 3-day moving average
            'ma_7_day',         # 7-day moving average
            
            # Categorical features (2 features)
            'dataset_uuid',     # Dataset identifier
            'rule_code'         # Rule code identifier
        ]
    
    def get_standard_features(self) -> List[str]:
        """
        Get the canonical list of 11 features for ML pipeline.
        
        Returns:
            List of feature names that should be used consistently
            across training and prediction pipelines
        """
        return self._standard_features.copy()
    
    def get_numeric_features(self) -> List[str]:
        """
        Get only the numeric features (first 9 features).
        
        Returns:
            List of numeric feature names
        """
        return self._standard_features[:9]
    
    def get_categorical_features(self) -> List[str]:
        """
        Get only the categorical features (last 2 features).
        
        Returns:
            List of categorical feature names
        """
        return self._standard_features[9:]
    
    def validate_feature_alignment(self, train_features: List[str], pred_features: List[str]) -> bool:
        """
        Validate that training and prediction features are perfectly aligned.
        
        Args:
            train_features: List of features used in training
            pred_features: List of features used in prediction
            
        Returns:
            True if features are perfectly aligned, False otherwise
        """
        standard_features = self.get_standard_features()
        
        # Check if both match the standard
        train_matches = (set(train_features) == set(standard_features) and 
                        len(train_features) == len(standard_features))
        pred_matches = (set(pred_features) == set(standard_features) and 
                       len(pred_features) == len(standard_features))
        
        # Check if they match each other
        features_match = (set(train_features) == set(pred_features) and 
                         len(train_features) == len(pred_features))
        
        alignment_status = train_matches and pred_matches and features_match
        
        if not alignment_status:
            logger.warning("Feature alignment mismatch detected",
                          train_count=len(train_features),
                          pred_count=len(pred_features),
                          expected_count=len(standard_features),
                          train_features=train_features,
                          pred_features=pred_features)
        
        return alignment_status
    
    def prepare_categorical_features(self, data: pd.DataFrame, 
                                   training_categories: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Prepare categorical features with consistent categories from training.
        
        This ensures that prediction-time categorical features use the same
        category levels as training, preventing encoding mismatches.
        
        Args:
            data: DataFrame containing features to prepare
            training_categories: Dictionary mapping categorical column names 
                                to their training-time category levels
            
        Returns:
            DataFrame with categorical features properly prepared
        """
        data_copy = data.copy()
        categorical_features = self.get_categorical_features()
        
        for col in categorical_features:
            if col in data_copy.columns and col in training_categories:
                # Convert to categorical with training categories
                data_copy[col] = pd.Categorical(
                    data_copy[col], 
                    categories=training_categories[col],
                    ordered=False
                )
                logger.debug("Prepared categorical feature",
                           column=col,
                           training_categories=len(training_categories[col]),
                           unique_values=data_copy[col].nunique())
            elif col in data_copy.columns:
                # Fallback: convert to category without specific levels
                data_copy[col] = data_copy[col].astype('category')
                logger.warning("Categorical feature prepared without training categories",
                              column=col)
        
        return data_copy
    
    def validate_feature_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that features have expected data types.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'issues': [],
            'feature_types': {}
        }
        
        standard_features = self.get_standard_features()
        numeric_features = self.get_numeric_features()
        categorical_features = self.get_categorical_features()
        
        for feature in standard_features:
            if feature not in data.columns:
                validation_results['valid'] = False
                validation_results['issues'].append(f"Missing feature: {feature}")
                continue
            
            feature_dtype = data[feature].dtype
            validation_results['feature_types'][feature] = str(feature_dtype)
            
            # Validate numeric features
            if feature in numeric_features:
                if not pd.api.types.is_numeric_dtype(feature_dtype):
                    validation_results['valid'] = False
                    validation_results['issues'].append(
                        f"Feature {feature} should be numeric, got {feature_dtype}")
            
            # Validate categorical features
            elif feature in categorical_features:
                if not pd.api.types.is_categorical_dtype(feature_dtype) and \
                   not pd.api.types.is_object_dtype(feature_dtype):
                    validation_results['valid'] = False
                    validation_results['issues'].append(
                        f"Feature {feature} should be categorical/object, got {feature_dtype}")
        
        logger.info("Feature data type validation completed",
                   valid=validation_results['valid'],
                   issues_count=len(validation_results['issues']))
        
        return validation_results
    
    def ensure_feature_order(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure features are in the standard order expected by the model.
        
        Args:
            data: DataFrame with features potentially in wrong order
            
        Returns:
            DataFrame with features reordered to standard order
        """
        standard_features = self.get_standard_features()
        
        # Select only standard features in the correct order
        available_features = [f for f in standard_features if f in data.columns]
        
        if len(available_features) != len(standard_features):
            missing_features = set(standard_features) - set(available_features)
            logger.warning("Some standard features missing",
                          missing=list(missing_features),
                          available_count=len(available_features))
        
        # Return data with features in standard order
        return data[available_features]
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the standard feature configuration.
        
        Returns:
            Dictionary with feature configuration summary
        """
        standard_features = self.get_standard_features()
        numeric_features = self.get_numeric_features()
        categorical_features = self.get_categorical_features()
        
        return {
            'total_features': len(standard_features),
            'numeric_features': len(numeric_features),
            'categorical_features': len(categorical_features),
            'feature_list': standard_features,
            'numeric_feature_list': numeric_features,
            'categorical_feature_list': categorical_features
        }