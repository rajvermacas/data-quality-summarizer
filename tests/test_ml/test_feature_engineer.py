"""
Tests for feature_engineer module.

Following TDD approach: Red -> Green -> Refactor
"""
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.data_quality_summarizer.ml.feature_engineer import (
    extract_time_features,
    create_lag_features,
    calculate_moving_averages,
    engineer_all_features
)


class TestFeatureEngineer:
    """Test suite for feature engineering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample aggregated data for feature engineering
        dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        self.sample_data = {
            'source': ['system_a'] * 5,
            'tenant_id': ['tenant1'] * 5,
            'dataset_uuid': ['uuid1'] * 5,
            'dataset_name': ['dataset1'] * 5,
            'business_date': dates,
            'rule_code': ['R001'] * 5,
            'pass_percentage': [80.0, 85.0, 75.0, 90.0, 88.0]
        }
        self.sample_df = pd.DataFrame(self.sample_data)
        # Convert business_date to datetime
        self.sample_df['business_date'] = pd.to_datetime(self.sample_df['business_date'])
    
    def test_extract_time_features_basic(self):
        """Test basic time feature extraction from business_date."""
        result_df = extract_time_features(self.sample_df.copy())
        
        # Check that time features are added
        time_columns = ['day_of_week', 'day_of_month', 'week_of_year', 'month']
        for col in time_columns:
            assert col in result_df.columns, f"Missing time feature: {col}"
        
        # Check specific values for first row (2024-01-01 is a Monday)
        first_row = result_df.iloc[0]
        assert first_row['day_of_week'] == 0  # Monday = 0
        assert first_row['day_of_month'] == 1
        assert first_row['month'] == 1
        assert first_row['week_of_year'] == 1
    
    def test_extract_time_features_preserves_data(self):
        """Test that time feature extraction preserves original data."""
        original_df = self.sample_df.copy()
        result_df = extract_time_features(original_df)
        
        # Original columns should be preserved
        for col in original_df.columns:
            assert col in result_df.columns
        
        # Original data should be unchanged
        assert len(result_df) == len(original_df)
        assert result_df['pass_percentage'].tolist() == original_df['pass_percentage'].tolist()
    
    def test_create_lag_features_basic(self):
        """Test creation of lag features for pass percentages."""
        # Sort by date to ensure proper lag calculation
        sorted_df = self.sample_df.sort_values(['dataset_uuid', 'rule_code', 'business_date'])
        
        result_df = create_lag_features(sorted_df, lag_days=[1, 2, 7])
        
        # Check that lag columns are created
        lag_columns = ['lag_1_day', 'lag_2_day', 'lag_7_day']
        for col in lag_columns:
            assert col in result_df.columns, f"Missing lag feature: {col}"
    
    def test_create_lag_features_values(self):
        """Test correct values in lag features."""
        sorted_df = self.sample_df.sort_values(['dataset_uuid', 'rule_code', 'business_date'])
        
        result_df = create_lag_features(sorted_df, lag_days=[1])
        
        # Second row should have lag_1_day = first row's pass_percentage
        assert pd.isna(result_df.iloc[0]['lag_1_day'])  # First row has no lag
        assert result_df.iloc[1]['lag_1_day'] == 80.0  # Second row lag = first row value
        assert result_df.iloc[2]['lag_1_day'] == 85.0  # Third row lag = second row value
    
    def test_create_lag_features_missing_dates(self):
        """Test lag features with missing dates (gaps in time series)."""
        # Create data with date gaps
        gap_dates = ['2024-01-01', '2024-01-03', '2024-01-05']  # Missing Jan 2 and 4
        gap_data = {
            'dataset_uuid': ['uuid1'] * 3,
            'rule_code': ['R001'] * 3,
            'business_date': pd.to_datetime(gap_dates),
            'pass_percentage': [80.0, 85.0, 90.0]
        }
        gap_df = pd.DataFrame(gap_data)
        
        result_df = create_lag_features(gap_df, lag_days=[1])
        
        # Should handle gaps gracefully (NaN for missing lags)
        assert pd.isna(result_df.iloc[0]['lag_1_day'])  # First row
        assert pd.isna(result_df.iloc[1]['lag_1_day'])  # Gap - no 1-day lag available
        assert pd.isna(result_df.iloc[2]['lag_1_day'])  # Gap - no 1-day lag available
    
    def test_calculate_moving_averages_basic(self):
        """Test calculation of moving averages."""
        sorted_df = self.sample_df.sort_values(['dataset_uuid', 'rule_code', 'business_date'])
        
        result_df = calculate_moving_averages(sorted_df, windows=[3, 7])
        
        # Check that moving average columns are created
        ma_columns = ['ma_3_day', 'ma_7_day']
        for col in ma_columns:
            assert col in result_df.columns, f"Missing moving average feature: {col}"
    
    def test_calculate_moving_averages_values(self):
        """Test correct values in moving averages."""
        sorted_df = self.sample_df.sort_values(['dataset_uuid', 'rule_code', 'business_date'])
        
        result_df = calculate_moving_averages(sorted_df, windows=[3])
        
        # Third row should have 3-day MA = average of first 3 values
        expected_ma_3 = (80.0 + 85.0 + 75.0) / 3
        assert abs(result_df.iloc[2]['ma_3_day'] - expected_ma_3) < 0.01
        
        # First two rows should have NaN for 3-day MA
        assert pd.isna(result_df.iloc[0]['ma_3_day'])
        assert pd.isna(result_df.iloc[1]['ma_3_day'])
    
    def test_engineer_all_features_integration(self):
        """Test complete feature engineering pipeline."""
        result_df = engineer_all_features(self.sample_df.copy())
        
        # Should have all feature types
        expected_features = [
            'day_of_week', 'day_of_month', 'week_of_year', 'month',  # Time features
            'lag_1_day', 'lag_2_day', 'lag_7_day',  # Lag features  
            'ma_3_day', 'ma_7_day'  # Moving averages
        ]
        
        for feature in expected_features:
            assert feature in result_df.columns, f"Missing feature: {feature}"
        
        # Original data should be preserved
        assert len(result_df) == len(self.sample_df)
        original_columns = self.sample_df.columns.tolist()
        for col in original_columns:
            assert col in result_df.columns
    
    def test_engineer_all_features_empty_dataframe(self):
        """Test feature engineering with empty DataFrame."""
        empty_df = pd.DataFrame(columns=self.sample_df.columns)
        
        result_df = engineer_all_features(empty_df)
        
        # Should return empty DataFrame with feature columns
        assert len(result_df) == 0
        assert 'day_of_week' in result_df.columns
        assert 'lag_1_day' in result_df.columns
        assert 'ma_3_day' in result_df.columns
    
    def test_engineer_all_features_single_row(self):
        """Test feature engineering with single row."""
        single_row = self.sample_df.iloc[:1].copy()
        
        result_df = engineer_all_features(single_row)
        
        # Should have time features but NaN for lag and MA features
        assert len(result_df) == 1
        assert pd.notna(result_df.iloc[0]['day_of_week'])
        assert pd.isna(result_df.iloc[0]['lag_1_day'])
        assert pd.isna(result_df.iloc[0]['ma_3_day'])