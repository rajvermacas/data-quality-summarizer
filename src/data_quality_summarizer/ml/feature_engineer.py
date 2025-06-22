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


def find_closest_lag_value(
    group_sorted: pd.DataFrame, 
    current_date: pd.Timestamp, 
    lag_days: int, 
    tolerance_days: int = 3
) -> float:
    """
    Find the closest lag value within a tolerance window.
    
    This function implements nearest-neighbor lag calculation to handle
    data gaps more gracefully than exact date matching.
    
    Args:
        group_sorted: DataFrame sorted by business_date for a specific group
        current_date: Current date to calculate lag from
        lag_days: Number of days to look back
        tolerance_days: Maximum deviation from exact lag date (default: 3)
        
    Returns:
        Closest lag value within tolerance, or NaN if none found
    """
    if group_sorted.empty:
        return pd.NA
        
    # Calculate target lag date
    target_lag_date = current_date - pd.Timedelta(days=lag_days)
    
    # Find all dates before or equal to current date (can't use future data)
    valid_dates = group_sorted[group_sorted['business_date'] <= current_date].copy()
    
    if valid_dates.empty:
        return pd.NA
    
    # Calculate distance from target lag date
    valid_dates['distance'] = abs((valid_dates['business_date'] - target_lag_date).dt.days)
    
    # Find closest date within tolerance
    within_tolerance = valid_dates[valid_dates['distance'] <= tolerance_days]
    
    if within_tolerance.empty:
        return pd.NA
    
    # Return the value from the closest date (smallest distance)
    closest_row = within_tolerance.loc[within_tolerance['distance'].idxmin()]
    return closest_row['pass_percentage']


def get_imputation_strategy(historical_data: pd.DataFrame = None) -> dict:
    """
    Get imputation strategy for feature engineering.
    
    Args:
        historical_data: Optional historical data to calculate averages
        
    Returns:
        Dictionary containing imputation configuration
    """
    if historical_data is not None and 'pass_percentage' in historical_data.columns:
        # Use historical average as default value
        default_value = float(historical_data['pass_percentage'].mean())
    else:
        # Use reasonable default when no historical data available
        default_value = 50.0
    
    return {
        'lag_features': {
            'method': 'nearest_neighbor',
            'tolerance_days': 3,
            'default_value': default_value
        },
        'moving_averages': {
            'method': 'flexible_window',
            'min_periods': 1,
            'default_value': default_value
        },
        'time_features': {
            'allow_nan': False  # Time features should never be NaN
        }
    }


def calculate_flexible_moving_average(
    data: pd.DataFrame,
    window_size: int,
    min_periods: int = 1
) -> pd.DataFrame:
    """
    Calculate moving averages with flexible minimum periods.
    
    Args:
        data: DataFrame with business_date and pass_percentage columns
        window_size: Size of moving window in days
        min_periods: Minimum periods required for calculation
        
    Returns:
        DataFrame with additional moving average column
    """
    result_df = data.copy()
    
    if len(result_df) == 0:
        result_df[f'avg_{window_size}_day'] = pd.Series(dtype='float64')
        return result_df
    
    # Ensure data is sorted by date
    result_df = result_df.sort_values('business_date').reset_index(drop=True)
    
    # Calculate rolling average with flexible minimum periods
    result_df[f'avg_{window_size}_day'] = result_df['pass_percentage'].rolling(
        window=window_size,
        min_periods=min_periods,
        center=False
    ).mean()
    
    logger.info(f"Flexible moving average calculated",
                window_size=window_size,
                min_periods=min_periods,
                rows=len(result_df))
    
    return result_df