"""
Test module for ML feature pipeline functionality.

This module tests the feature engineering pipeline for single prediction requests,
including historical data lookup and lag feature calculation.
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from src.data_quality_summarizer.ml.feature_pipeline import FeaturePipeline


class TestFeaturePipeline:
    """Test the FeaturePipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample historical data
        self.sample_data = pd.DataFrame({
            'dataset_uuid': ['abc123'] * 10,
            'rule_code': ['R001'] * 10,
            'business_date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'pass_percentage': [85.0, 87.5, 90.0, 82.3, 88.9, 91.2, 86.7, 89.1, 92.5, 88.0]
        })
        
        self.pipeline = FeaturePipeline(self.sample_data)
    
    def test_pipeline_initialization(self):
        """Test FeaturePipeline initializes correctly."""
        pipeline = FeaturePipeline(self.sample_data)
        assert pipeline is not None
        assert hasattr(pipeline, 'historical_data')
        assert hasattr(pipeline, 'lookup_historical_data')
        assert hasattr(pipeline, 'create_prediction_features')
        assert hasattr(pipeline, 'engineer_features_for_prediction')
    
    def test_initialization_with_empty_data(self):
        """Test initialization with empty DataFrame."""
        empty_data = pd.DataFrame()
        pipeline = FeaturePipeline(empty_data)
        assert pipeline.historical_data.empty
    
    def test_lookup_historical_data_success(self):
        """Test successful historical data lookup."""
        result = self.pipeline.lookup_historical_data('abc123', 'R001')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert all(result['dataset_uuid'] == 'abc123')
        assert all(result['rule_code'] == 'R001')
    
    def test_lookup_historical_data_no_matches(self):
        """Test historical data lookup with no matching data."""
        result = self.pipeline.lookup_historical_data('xyz789', 'R999')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_lookup_historical_data_partial_matches(self):
        """Test historical data lookup with partial matches."""
        # Test with matching dataset but different rule
        result = self.pipeline.lookup_historical_data('abc123', 'R999')
        assert len(result) == 0
        
        # Test with different dataset but matching rule
        result = self.pipeline.lookup_historical_data('xyz789', 'R001')
        assert len(result) == 0
    
    def test_create_prediction_features_basic(self):
        """Test basic prediction feature creation."""
        prediction_date = date(2024, 1, 15)
        historical_subset = self.sample_data.head(5)
        
        features = self.pipeline.create_prediction_features(
            'abc123', 'R001', prediction_date, historical_subset
        )
        
        assert isinstance(features, dict)
        assert 'dataset_uuid' in features
        assert 'rule_code' in features
        assert 'business_date' in features
        assert features['dataset_uuid'] == 'abc123'
        assert features['rule_code'] == 'R001'
        assert features['business_date'] == prediction_date
    
    def test_create_prediction_features_time_features(self):
        """Test time-based feature extraction in prediction features."""
        prediction_date = date(2024, 6, 15)  # Saturday
        historical_subset = self.sample_data.head(3)
        
        features = self.pipeline.create_prediction_features(
            'abc123', 'R001', prediction_date, historical_subset
        )
        
        # Check time-based features
        assert 'day_of_week' in features
        assert 'day_of_month' in features
        assert 'month' in features
        assert 'week_of_year' in features
        
        assert features['day_of_week'] == 5  # Saturday
        assert features['day_of_month'] == 15
        assert features['month'] == 6
    
    def test_create_prediction_features_lag_features(self):
        """Test lag feature creation in prediction features."""
        prediction_date = date(2024, 1, 15)
        historical_subset = self.sample_data.copy()
        historical_subset['business_date'] = pd.to_datetime(historical_subset['business_date'])
        
        features = self.pipeline.create_prediction_features(
            'abc123', 'R001', prediction_date, historical_subset
        )
        
        # Check lag features
        assert 'lag_1_day' in features
        assert 'lag_2_day' in features
        assert 'lag_7_day' in features
    
    def test_create_prediction_features_moving_averages(self):
        """Test moving average calculation in prediction features."""
        prediction_date = date(2024, 1, 15)
        historical_subset = self.sample_data.copy()
        historical_subset['business_date'] = pd.to_datetime(historical_subset['business_date'])
        
        features = self.pipeline.create_prediction_features(
            'abc123', 'R001', prediction_date, historical_subset
        )
        
        # Check moving average features
        assert 'ma_3_day' in features
        assert 'ma_7_day' in features
    
    def test_create_prediction_features_missing_historical_data(self):
        """Test feature creation with insufficient historical data."""
        prediction_date = date(2024, 1, 15)
        empty_data = pd.DataFrame(columns=self.sample_data.columns)
        
        features = self.pipeline.create_prediction_features(
            'abc123', 'R001', prediction_date, empty_data
        )
        
        # Should still have basic features
        assert 'dataset_uuid' in features
        assert 'rule_code' in features
        assert 'business_date' in features
        
        # Lag and moving average features should be NaN or default values
        assert 'lag_1_day' in features
        assert 'ma_3_day' in features
    
    def test_engineer_features_for_prediction_complete_workflow(self):
        """Test complete feature engineering workflow for prediction."""
        prediction_date = date(2024, 1, 15)
        
        features = self.pipeline.engineer_features_for_prediction(
            'abc123', 'R001', prediction_date
        )
        
        assert isinstance(features, dict)
        assert len(features) > 10  # Should have multiple feature types
        
        # Check core features
        assert features['dataset_uuid'] == 'abc123'
        assert features['rule_code'] == 'R001'
        assert features['business_date'] == prediction_date
        
        # Check time features
        assert 'day_of_week' in features
        assert 'month' in features
        
        # Check lag features
        assert 'lag_1_day' in features
        
        # Check moving averages
        assert 'ma_3_day' in features
    
    def test_engineer_features_for_prediction_no_historical_data(self):
        """Test feature engineering with no historical data available."""
        # Create pipeline with empty data
        empty_pipeline = FeaturePipeline(pd.DataFrame())
        
        features = empty_pipeline.engineer_features_for_prediction(
            'xyz789', 'R999', date(2024, 1, 15)
        )
        
        # Should still create basic features
        assert isinstance(features, dict)
        assert features['dataset_uuid'] == 'xyz789'
        assert features['rule_code'] == 'R999'
        assert features['business_date'] == date(2024, 1, 15)


class TestFeaturePipelineEdgeCases:
    """Test edge cases for feature pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create data with edge cases
        self.edge_case_data = pd.DataFrame({
            'dataset_uuid': ['abc123', 'abc123', 'def456'],
            'rule_code': ['R001', 'R002', 'R001'], 
            'business_date': [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 1)
            ],
            'pass_percentage': [100.0, 0.0, 50.0]
        })
        
        self.pipeline = FeaturePipeline(self.edge_case_data)
    
    def test_lookup_with_mixed_data_types(self):
        """Test lookup with mixed dataset and rule types."""
        # Test with different combinations of data present
        result1 = self.pipeline.lookup_historical_data('abc123', 'R001')
        assert len(result1) == 1
        
        result2 = self.pipeline.lookup_historical_data('abc123', 'R002')
        assert len(result2) == 1
        
        result3 = self.pipeline.lookup_historical_data('def456', 'R001')
        assert len(result3) == 1
    
    def test_create_features_with_extreme_values(self):
        """Test feature creation with extreme pass percentage values."""
        historical_data = pd.DataFrame({
            'dataset_uuid': ['test'] * 5,
            'rule_code': ['R001'] * 5,
            'business_date': pd.date_range('2024-01-01', periods=5),
            'pass_percentage': [0.0, 25.0, 50.0, 75.0, 100.0]
        })
        
        features = self.pipeline.create_prediction_features(
            'test', 'R001', date(2024, 1, 10), historical_data
        )
        
        # Features should be created successfully even with extreme values
        assert isinstance(features, dict)
        assert 'ma_3_day' in features
        assert 'lag_1_day' in features
    
    def test_date_edge_cases(self):
        """Test feature creation with various date edge cases."""
        # Test with year boundary
        year_boundary = date(2024, 1, 1)
        features = self.pipeline.engineer_features_for_prediction(
            'abc123', 'R001', year_boundary
        )
        assert features['month'] == 1
        assert features['day_of_month'] == 1
        
        # Test with leap year date
        leap_year = date(2024, 2, 29)
        features = self.pipeline.engineer_features_for_prediction(
            'abc123', 'R001', leap_year
        )
        assert features['month'] == 2
        assert features['day_of_month'] == 29
    
    def test_single_historical_record(self):
        """Test feature creation with only one historical record."""
        single_record = pd.DataFrame({
            'dataset_uuid': ['single'],
            'rule_code': ['R001'],
            'business_date': [datetime(2024, 1, 1)],
            'pass_percentage': [85.0]
        })
        
        pipeline = FeaturePipeline(single_record)
        features = pipeline.engineer_features_for_prediction(
            'single', 'R001', date(2024, 1, 5)
        )
        
        assert isinstance(features, dict)
        assert features['dataset_uuid'] == 'single'
        assert features['rule_code'] == 'R001'


class TestFeaturePipelineIntegration:
    """Test integration between FeaturePipeline and other ML components."""
    
    def test_feature_compatibility_with_existing_engineer(self):
        """Test that pipeline features are compatible with existing feature engineer."""
        # Create data similar to what feature_engineer expects
        sample_data = pd.DataFrame({
            'dataset_uuid': ['test'] * 7,
            'rule_code': ['R001'] * 7,
            'business_date': pd.date_range('2024-01-01', periods=7),
            'pass_percentage': [85.0, 87.5, 90.0, 82.3, 88.9, 91.2, 86.7]
        })
        
        pipeline = FeaturePipeline(sample_data)
        features = pipeline.engineer_features_for_prediction(
            'test', 'R001', date(2024, 1, 10)
        )
        
        # Check that features have expected structure for model input
        expected_keys = [
            'dataset_uuid', 'rule_code', 'business_date',
            'day_of_week', 'day_of_month', 'month', 'week_of_year',
            'lag_1_day', 'lag_2_day', 'lag_7_day',
            'ma_3_day', 'ma_7_day'
        ]
        
        for key in expected_keys:
            assert key in features, f"Missing expected feature: {key}"
    
    def test_output_format_for_model_input(self):
        """Test that output format is suitable for model input."""
        sample_data = pd.DataFrame({
            'dataset_uuid': ['test'] * 5,
            'rule_code': ['R001'] * 5,
            'business_date': pd.date_range('2024-01-01', periods=5),
            'pass_percentage': [85.0, 87.5, 90.0, 82.3, 88.9]
        })
        
        pipeline = FeaturePipeline(sample_data)
        features = pipeline.engineer_features_for_prediction(
            'test', 'R001', date(2024, 1, 10)
        )
        
        # Check data types are appropriate for model input
        for key, value in features.items():
            if key in ['dataset_uuid', 'rule_code']:
                assert isinstance(value, str)
            elif key == 'business_date':
                assert isinstance(value, date)
            else:
                # Numeric features should be float or int
                assert isinstance(value, (int, float, np.integer, np.floating, type(None)))