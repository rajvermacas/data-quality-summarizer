#!/usr/bin/env python3
"""
Debug script to analyze why predictions are extremely high (8187%).
Investigates the model training data and prediction pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_quality_summarizer.ml.data_loader import load_and_validate_csv, parse_results_column, create_binary_pass_column
from data_quality_summarizer.ml.aggregator import aggregate_pass_percentages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Debug the prediction calibration issue."""
    print("🔍 Debugging Prediction Calibration Issue\n")
    
    # Load and analyze the training data
    data_path = "test_data/large_sample.csv"
    
    print("=" * 60)
    print("1. RAW DATA ANALYSIS")
    print("=" * 60)
    
    # Load raw data
    raw_data = load_and_validate_csv(data_path)
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Columns: {list(raw_data.columns)}")
    
    # Parse results and create binary pass column
    parsed_results = parse_results_column(raw_data['results'].tolist())
    raw_data['is_pass'] = create_binary_pass_column(parsed_results)
    
    print(f"\nPass/Fail distribution:")
    print(f"Total records: {len(raw_data)}")
    print(f"Passes: {raw_data['is_pass'].sum()}")
    print(f"Fails: {len(raw_data) - raw_data['is_pass'].sum()}")
    print(f"Overall pass rate: {raw_data['is_pass'].mean():.1%}")
    
    print("=" * 60)
    print("2. AGGREGATED DATA ANALYSIS")
    print("=" * 60)
    
    # Aggregate data (this is what the model trains on)
    aggregated_data = aggregate_pass_percentages(raw_data)
    print(f"Aggregated data shape: {aggregated_data.shape}")
    print(f"Columns: {list(aggregated_data.columns)}")
    
    print(f"\nPass percentage statistics:")
    print(f"Min: {aggregated_data['pass_percentage'].min():.1f}%")
    print(f"Max: {aggregated_data['pass_percentage'].max():.1f}%")
    print(f"Mean: {aggregated_data['pass_percentage'].mean():.1f}%")
    print(f"Median: {aggregated_data['pass_percentage'].median():.1f}%")
    print(f"Std: {aggregated_data['pass_percentage'].std():.1f}%")
    
    print(f"\nSample of aggregated data:")
    print(aggregated_data[['dataset_uuid', 'rule_code', 'business_date', 'pass_percentage']].head(10))
    
    print("=" * 60)
    print("3. MODEL TARGET VARIABLE ANALYSIS")
    print("=" * 60)
    
    # The model is trained to predict pass_percentage (0-100 range)
    # But our predictions are coming out as 8000+%
    
    target_values = aggregated_data['pass_percentage'].values
    print(f"Target variable (pass_percentage) range: {target_values.min():.1f} to {target_values.max():.1f}")
    print(f"Expected range: 0.0 to 100.0")
    
    if target_values.max() > 100:
        print("⚠️  WARNING: Target values exceed 100% - this is the problem!")
        print("Pass percentages should be capped at 100%")
        
        # Show problematic records
        problematic = aggregated_data[aggregated_data['pass_percentage'] > 100]
        print(f"\nProblematic records (pass_percentage > 100%):")
        print(problematic[['dataset_uuid', 'rule_code', 'pass_percentage']].head())
    
    print("=" * 60)
    print("4. RECOMMENDATIONS")
    print("=" * 60)
    
    print("The issue is likely one of these:")
    print("1. Pass percentage calculation error in aggregation")
    print("2. Model not properly constraining output to 0-100% range")
    print("3. Feature scaling issues causing unrealistic predictions")
    print("4. Data corruption where passes > total_records")
    
    print("\nNext steps:")
    print("1. Fix aggregation to cap pass_percentage at 100%")
    print("2. Add output constraints to model predictions")
    print("3. Validate data quality in aggregation step")

if __name__ == "__main__":
    main()