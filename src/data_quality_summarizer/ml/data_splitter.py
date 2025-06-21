"""
Data Splitter module for ML pipeline.

This module provides chronological train/test splitting functionality
for time series data to ensure proper temporal validation.
"""
import pandas as pd
from datetime import datetime
from typing import Tuple
import logging


logger = logging.getLogger(__name__)


def split_data_chronologically(
    data: pd.DataFrame, 
    cutoff_date: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train and test sets.
    
    Args:
        data: DataFrame with business_date column
        cutoff_date: Date to split on (train < cutoff_date <= test)
        
    Returns:
        Tuple of (train_data, test_data)
        
    Raises:
        TypeError: If cutoff_date is not a datetime object
    """
    if not isinstance(cutoff_date, datetime):
        raise TypeError("cutoff_date must be a datetime object")
    
    if data.empty:
        return data.copy(), data.copy()
    
    train_data = data[data['business_date'] < cutoff_date].copy()
    test_data = data[data['business_date'] >= cutoff_date].copy()
    
    logger.info(f"Split data: {len(train_data)} train, {len(test_data)} test")
    
    return train_data, test_data


def validate_temporal_ordering(data: pd.DataFrame) -> bool:
    """
    Validate that data is ordered chronologically by business_date.
    
    Args:
        data: DataFrame with business_date column
        
    Returns:
        True if data is temporally ordered, False otherwise
    """
    if data.empty:
        return True
    
    dates = data['business_date'].values
    return all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))


def determine_optimal_cutoff_date(
    data: pd.DataFrame, 
    train_ratio: float = 0.8
) -> datetime:
    """
    Determine optimal cutoff date for given train/test ratio.
    
    Args:
        data: DataFrame with business_date column
        train_ratio: Fraction of data for training (default 0.8)
        
    Returns:
        Optimal cutoff date for the specified ratio
    """
    if data.empty:
        return datetime.now()
    
    sorted_dates = data['business_date'].sort_values()
    cutoff_index = int(len(sorted_dates) * train_ratio)
    
    if cutoff_index >= len(sorted_dates):
        cutoff_index = len(sorted_dates) - 1
    elif cutoff_index < 0:
        cutoff_index = 0
    
    cutoff_date = sorted_dates.iloc[cutoff_index]
    logger.info(f"Determined cutoff date: {cutoff_date} (ratio: {train_ratio})")
    
    return cutoff_date