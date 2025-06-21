"""
Feature engineering pipeline for single prediction requests.

This module provides feature engineering capabilities for individual prediction
requests, including historical data lookup and lag feature calculation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class FeaturePipeline:
    """
    Feature engineering pipeline for single prediction requests.
    
    Handles historical data lookup, lag feature creation, and moving averages
    for individual prediction requests.
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialize the FeaturePipeline with historical data.
        
        Args:
            historical_data: DataFrame containing historical pass percentage data
        """
        self.historical_data = historical_data.copy() if not historical_data.empty else pd.DataFrame()
        
        # Ensure business_date is datetime if data exists
        if not self.historical_data.empty and 'business_date' in self.historical_data.columns:
            self.historical_data['business_date'] = pd.to_datetime(self.historical_data['business_date'])
    
    def lookup_historical_data(self, dataset_uuid: str, rule_code: str) -> pd.DataFrame:
        """
        Look up historical data for specific dataset and rule combination.
        
        Args:
            dataset_uuid: The dataset UUID to filter for
            rule_code: The rule code to filter for
            
        Returns:
            DataFrame containing matching historical records
        """
        if self.historical_data.empty:
            return pd.DataFrame()
        
        # Filter for matching dataset and rule
        mask = (
            (self.historical_data['dataset_uuid'] == dataset_uuid) &
            (self.historical_data['rule_code'] == rule_code)
        )
        
        result = self.historical_data[mask].copy()
        
        # Sort by date for chronological processing
        if not result.empty and 'business_date' in result.columns:
            result = result.sort_values('business_date')
        
        logger.debug("Historical data lookup completed",
                    dataset_uuid=dataset_uuid,
                    rule_code=rule_code,
                    records_found=len(result))
        
        return result
    
    def create_prediction_features(
        self,
        dataset_uuid: str,
        rule_code: str,
        prediction_date: date,
        historical_subset: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Create features for a single prediction request.
        
        Args:
            dataset_uuid: The dataset UUID
            rule_code: The rule code
            prediction_date: The date to predict for
            historical_subset: Historical data for this dataset/rule combination
            
        Returns:
            Dictionary containing engineered features
        """
        features = {
            'dataset_uuid': dataset_uuid,
            'rule_code': rule_code,
            'business_date': prediction_date
        }
        
        # Extract time-based features from prediction date
        dt = pd.to_datetime(prediction_date)
        features['day_of_week'] = dt.dayofweek
        features['day_of_month'] = dt.day
        features['week_of_year'] = dt.isocalendar().week
        features['month'] = dt.month
        
        # Create lag features
        features.update(self._create_lag_features(prediction_date, historical_subset))
        
        # Create moving average features
        features.update(self._create_moving_average_features(prediction_date, historical_subset))
        
        logger.debug("Prediction features created",
                    dataset_uuid=dataset_uuid,
                    rule_code=rule_code,
                    prediction_date=str(prediction_date),
                    feature_count=len(features))
        
        return features
    
    def _create_lag_features(self, prediction_date: date, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Create lag features from historical data."""
        lag_features = {}
        
        if historical_data.empty:
            # Return NaN for missing data
            lag_features['lag_1_day'] = np.nan
            lag_features['lag_2_day'] = np.nan
            lag_features['lag_7_day'] = np.nan
            return lag_features
        
        # Convert prediction_date to datetime for comparison
        pred_dt = pd.to_datetime(prediction_date)
        
        # Sort historical data by date
        sorted_data = historical_data.sort_values('business_date')
        
        # Calculate lag features
        for lag_days in [1, 2, 7]:
            lag_date = pred_dt - pd.Timedelta(days=lag_days)
            
            # Find closest date on or before lag_date
            valid_dates = sorted_data[sorted_data['business_date'] <= lag_date]
            
            if not valid_dates.empty:
                # Get the most recent valid date
                closest_record = valid_dates.iloc[-1]
                lag_features[f'lag_{lag_days}_day'] = closest_record['pass_percentage']
            else:
                lag_features[f'lag_{lag_days}_day'] = np.nan
        
        return lag_features
    
    def _create_moving_average_features(self, prediction_date: date, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Create moving average features from historical data."""
        ma_features = {}
        
        if historical_data.empty:
            # Return NaN for missing data
            ma_features['ma_3_day'] = np.nan
            ma_features['ma_7_day'] = np.nan
            return ma_features
        
        # Convert prediction_date to datetime for comparison
        pred_dt = pd.to_datetime(prediction_date)
        
        # Calculate moving averages
        for window in [3, 7]:
            # Get data from the window period before prediction date
            start_date = pred_dt - pd.Timedelta(days=window)
            
            window_data = historical_data[
                (historical_data['business_date'] >= start_date) &
                (historical_data['business_date'] < pred_dt)
            ]
            
            if not window_data.empty:
                ma_features[f'ma_{window}_day'] = window_data['pass_percentage'].mean()
            else:
                ma_features[f'ma_{window}_day'] = np.nan
        
        return ma_features
    
    def engineer_features_for_prediction(
        self,
        dataset_uuid: str,
        rule_code: str,
        prediction_date: date
    ) -> Dict[str, Any]:
        """
        Complete feature engineering workflow for a single prediction.
        
        Args:
            dataset_uuid: The dataset UUID
            rule_code: The rule code
            prediction_date: The date to predict for
            
        Returns:
            Dictionary containing all engineered features
        """
        # Look up historical data for this dataset/rule combination
        historical_subset = self.lookup_historical_data(dataset_uuid, rule_code)
        
        # Create prediction features
        features = self.create_prediction_features(
            dataset_uuid, rule_code, prediction_date, historical_subset
        )
        
        logger.info("Feature engineering completed for prediction",
                   dataset_uuid=dataset_uuid,
                   rule_code=rule_code,
                   prediction_date=str(prediction_date),
                   historical_records=len(historical_subset),
                   total_features=len(features))
        
        return features