"""
ML aggregator module for data quality pass percentage calculations.

Groups data by (dataset_uuid, rule_code, business_date) and calculates
pass percentages for each group.
"""
import pandas as pd
from typing import List
import structlog

logger = structlog.get_logger(__name__)


def aggregate_pass_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data by (dataset_uuid, rule_code, business_date) and calculate pass percentages.
    
    Args:
        df: DataFrame with is_pass column and grouping columns
        
    Returns:
        DataFrame with aggregated pass percentages
    """
    if len(df) == 0:
        # Return empty DataFrame with correct structure
        empty_df = pd.DataFrame(columns=[
            'source', 'tenant_id', 'dataset_uuid', 'dataset_name',
            'business_date', 'rule_code', 'pass_percentage'
        ])
        logger.info("Empty input DataFrame, returning empty result")
        return empty_df
    
    # Group by the key columns
    grouping_columns = ['source', 'tenant_id', 'dataset_uuid', 'dataset_name', 
                       'business_date', 'rule_code']
    
    # Aggregate with custom function
    aggregated = df.groupby(grouping_columns).agg({
        'is_pass': lambda x: calculate_group_pass_percentage(x.tolist())
    }).reset_index()
    
    # Rename the aggregated column
    aggregated.rename(columns={'is_pass': 'pass_percentage'}, inplace=True)
    
    logger.info("Pass percentage aggregation completed", 
                input_rows=len(df), 
                output_groups=len(aggregated))
    
    return aggregated


def calculate_group_pass_percentage(is_pass_values: List[int]) -> float:
    """
    Calculate pass percentage for a group of binary pass/fail values.
    
    Args:
        is_pass_values: List of binary values (1 for pass, 0 for fail)
        
    Returns:
        Pass percentage as float (0.0 to 100.0)
    """
    if not is_pass_values:
        return handle_empty_groups(is_pass_values)
    
    pass_count = sum(is_pass_values)
    total_count = len(is_pass_values)
    
    pass_percentage = (pass_count / total_count) * 100.0
    
    return round(pass_percentage, 1)


def handle_empty_groups(empty_values: List) -> float:
    """
    Handle empty groups by returning default pass percentage.
    
    Args:
        empty_values: Empty list
        
    Returns:
        Default pass percentage (0.0)
    """
    logger.warning("Empty group encountered, defaulting to 0% pass rate")
    return 0.0