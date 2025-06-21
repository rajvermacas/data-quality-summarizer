"""
Feature engineering module for ML pipeline.

Creates time-based features, lag features, and moving averages
for time series prediction of data quality pass percentages.
"""
import pandas as pd
from typing import List
import structlog

logger = structlog.get_logger(__name__)


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from business_date column.
    
    Args:
        df: DataFrame with business_date column (datetime)
        
    Returns:
        DataFrame with additional time feature columns
    """
    if len(df) == 0:
        # Handle empty DataFrame
        result_df = df.copy()
        result_df['day_of_week'] = pd.Series(dtype='int64')
        result_df['day_of_month'] = pd.Series(dtype='int64')
        result_df['week_of_year'] = pd.Series(dtype='int64')
        result_df['month'] = pd.Series(dtype='int64')
        return result_df
    
    result_df = df.copy()
    
    # Ensure business_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df['business_date']):
        result_df['business_date'] = pd.to_datetime(result_df['business_date'])
    
    # Extract time features
    result_df['day_of_week'] = result_df['business_date'].dt.dayofweek
    result_df['day_of_month'] = result_df['business_date'].dt.day
    result_df['week_of_year'] = result_df['business_date'].dt.isocalendar().week
    result_df['month'] = result_df['business_date'].dt.month
    
    logger.info("Time features extracted", 
                rows=len(result_df),
                features=['day_of_week', 'day_of_month', 'week_of_year', 'month'])
    
    return result_df


def create_lag_features(df: pd.DataFrame, lag_days: List[int] = [1, 2, 7]) -> pd.DataFrame:
    """
    Create lag features for pass percentages.
    
    Args:
        df: DataFrame sorted by (dataset_uuid, rule_code, business_date)
        lag_days: List of lag periods in days
        
    Returns:
        DataFrame with lag feature columns
    """
    if len(df) == 0:
        # Handle empty DataFrame
        result_df = df.copy()
        for lag in lag_days:
            result_df[f'lag_{lag}_day'] = pd.Series(dtype='float64')
        return result_df
    
    result_df = df.copy()
    
    # Ensure business_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df['business_date']):
        result_df['business_date'] = pd.to_datetime(result_df['business_date'])
    
    # Create lag features for each group
    for lag in lag_days:
        result_df[f'lag_{lag}_day'] = pd.Series(dtype='float64', index=result_df.index)
    
    # Group by dataset and rule to create lags within each group
    for (dataset_uuid, rule_code), group in result_df.groupby(['dataset_uuid', 'rule_code']):
        group_sorted = group.sort_values('business_date')
        
        for lag in lag_days:
            lag_column = f'lag_{lag}_day'
            
            # Calculate lag values
            for i, (idx, row) in enumerate(group_sorted.iterrows()):
                if i >= lag:
                    # Check if the lag date exists and is exactly lag days before
                    current_date = row['business_date']
                    expected_lag_date = current_date - pd.Timedelta(days=lag)
                    
                    # Find the row with the expected lag date
                    lag_mask = (group_sorted['business_date'] == expected_lag_date)
                    if lag_mask.any():
                        lag_value = group_sorted.loc[lag_mask, 'pass_percentage'].iloc[0]
                        result_df.loc[idx, lag_column] = lag_value
                    else:
                        # No exact match for lag date, set to NaN
                        result_df.loc[idx, lag_column] = pd.NA
                else:
                    # Not enough historical data for this lag
                    result_df.loc[idx, lag_column] = pd.NA
    
    logger.info("Lag features created", 
                rows=len(result_df),
                lag_periods=lag_days)
    
    return result_df


def calculate_moving_averages(df: pd.DataFrame, windows: List[int] = [3, 7]) -> pd.DataFrame:
    """
    Calculate moving averages for pass percentages.
    
    Args:
        df: DataFrame sorted by (dataset_uuid, rule_code, business_date)
        windows: List of window sizes in days
        
    Returns:
        DataFrame with moving average columns
    """
    if len(df) == 0:
        # Handle empty DataFrame
        result_df = df.copy()
        for window in windows:
            result_df[f'ma_{window}_day'] = pd.Series(dtype='float64')
        return result_df
    
    result_df = df.copy()
    
    # Initialize moving average columns
    for window in windows:
        result_df[f'ma_{window}_day'] = pd.Series(dtype='float64', index=result_df.index)
    
    # Group by dataset and rule to calculate moving averages within each group
    for (dataset_uuid, rule_code), group in result_df.groupby(['dataset_uuid', 'rule_code']):
        group_sorted = group.sort_values('business_date')
        
        for window in windows:
            ma_column = f'ma_{window}_day'
            
            # Calculate moving average for each row
            for i, (idx, row) in enumerate(group_sorted.iterrows()):
                if i >= window - 1:  # Need at least 'window' points
                    # Get the window of values ending at current row
                    window_values = group_sorted.iloc[i - window + 1:i + 1]['pass_percentage']
                    ma_value = window_values.mean()
                    result_df.loc[idx, ma_column] = ma_value
                else:
                    # Not enough data for moving average
                    result_df.loc[idx, ma_column] = pd.NA
    
    logger.info("Moving averages calculated", 
                rows=len(result_df),
                windows=windows)
    
    return result_df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline.
    
    Args:
        df: DataFrame with aggregated pass percentages
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("Starting complete feature engineering pipeline", rows=len(df))
    
    # Step 1: Extract time features
    result_df = extract_time_features(df)
    
    # Step 2: Sort data for lag and moving average calculations
    if len(result_df) > 0:
        result_df = result_df.sort_values(['dataset_uuid', 'rule_code', 'business_date'])
    
    # Step 3: Create lag features
    result_df = create_lag_features(result_df, lag_days=[1, 2, 7])
    
    # Step 4: Calculate moving averages
    result_df = calculate_moving_averages(result_df, windows=[3, 7])
    
    logger.info("Feature engineering pipeline completed", 
                input_rows=len(df),
                output_rows=len(result_df),
                total_features=len(result_df.columns))
    
    return result_df