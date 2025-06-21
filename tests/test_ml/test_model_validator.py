"""
Tests for Model Validator module.

This module tests model quality validation, drift detection capabilities,
and performance monitoring utilities.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
from src.data_quality_summarizer.ml.model_validator import ModelValidator


class TestModelValidator:
    """Test cases for ModelValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ModelValidator()
        
        # Create sample training data
        self.train_data = pd.DataFrame({
            'dataset_uuid': ['uuid1', 'uuid2'] * 50,
            'rule_code': ['R001', 'R002'] * 50,
            'business_date': pd.date_range('2024-01-01', periods=100),
            'pass_percentage': np.random.uniform(50, 95, 100),
            'day_of_week': [1, 2] * 50,
            'month': [1, 2] * 50,
            'lag_1_day': np.random.uniform(50, 95, 100),
            'lag_7_day': np.random.uniform(50, 95, 100),
            'ma_3_day': np.random.uniform(50, 95, 100),
            'ma_7_day': np.random.uniform(50, 95, 100)
        })
        
        # Create sample new data with potential drift
        self.new_data = pd.DataFrame({
            'dataset_uuid': ['uuid1', 'uuid2'] * 25,
            'rule_code': ['R001', 'R002'] * 25,
            'business_date': pd.date_range('2024-04-01', periods=50),
            'pass_percentage': np.random.uniform(30, 70, 50),  # Different distribution
            'day_of_week': [1, 2] * 25,
            'month': [4, 4] * 25,
            'lag_1_day': np.random.uniform(30, 70, 50),
            'lag_7_day': np.random.uniform(30, 70, 50),
            'ma_3_day': np.random.uniform(30, 70, 50),
            'ma_7_day': np.random.uniform(30, 70, 50)
        })
        
        # Mock model
        self.mock_model = Mock()
        self.mock_model.predict.return_value = np.random.uniform(40, 80, 50)
    
    def test_validator_initialization(self):
        """Test that validator initializes correctly."""
        assert self.validator is not None
        assert hasattr(self.validator, 'validate_model_quality')
        assert hasattr(self.validator, 'detect_data_drift')
        assert hasattr(self.validator, 'monitor_performance')
    
    def test_model_quality_validation_good_model(self):
        """Test model quality validation with a good performing model."""
        # Create predictions close to actual values
        y_true = np.array([85.0, 72.5, 91.2, 68.8, 95.1])
        y_pred = np.array([83.5, 74.0, 89.8, 70.2, 93.5])  # Close predictions
        
        quality_report = self.validator.validate_model_quality(y_true, y_pred)
        
        assert 'mae' in quality_report
        assert 'rmse' in quality_report
        assert 'r2_score' in quality_report
        assert 'quality_status' in quality_report
        
        assert quality_report['mae'] < 5.0  # Good MAE
        assert quality_report['quality_status'] == 'GOOD'
    
    def test_model_quality_validation_poor_model(self):
        """Test model quality validation with a poor performing model."""
        # Create predictions far from actual values
        y_true = np.array([85.0, 72.5, 91.2, 68.8, 95.1])
        y_pred = np.array([45.0, 30.0, 25.0, 40.0, 35.0])  # Poor predictions
        
        quality_report = self.validator.validate_model_quality(y_true, y_pred)
        
        assert quality_report['mae'] > 20.0  # Poor MAE
        assert quality_report['quality_status'] in ['POOR', 'NEEDS_RETRAINING']
    
    def test_data_drift_detection_no_drift(self):
        """Test drift detection when no significant drift exists."""
        # Create similar data distributions
        similar_data = self.train_data.copy()
        similar_data['pass_percentage'] += np.random.normal(0, 2, len(similar_data))
        
        drift_report = self.validator.detect_data_drift(
            self.train_data,
            similar_data,
            feature_columns=['pass_percentage', 'lag_1_day', 'ma_3_day']
        )
        
        assert 'drift_detected' in drift_report
        assert 'drift_score' in drift_report
        assert 'affected_features' in drift_report
        
        assert drift_report['drift_detected'] is False
        assert drift_report['drift_score'] < 0.5  # Low drift score
    
    def test_data_drift_detection_with_drift(self):
        """Test drift detection when significant drift exists."""
        drift_report = self.validator.detect_data_drift(
            self.train_data,
            self.new_data,  # Has different distribution
            feature_columns=['pass_percentage', 'lag_1_day', 'ma_3_day']
        )
        
        assert drift_report['drift_detected'] == True
        assert drift_report['drift_score'] > 0.3  # Significant drift
        assert len(drift_report['affected_features']) > 0
    
    def test_performance_monitoring_tracks_metrics(self):
        """Test that performance monitoring tracks key metrics."""
        predictions = [85.5, 72.3, 91.2, 68.8, 95.1]
        actuals = [83.0, 74.0, 89.0, 70.0, 93.0]
        
        self.validator.monitor_performance(predictions, actuals, timestamp='2024-01-15')
        
        performance_history = self.validator.get_performance_history()
        
        assert len(performance_history) == 1
        assert '2024-01-15' in performance_history
        assert 'mae' in performance_history['2024-01-15']
        assert 'prediction_count' in performance_history['2024-01-15']
    
    def test_performance_monitoring_accumulates_data(self):
        """Test that performance monitoring accumulates data over time."""
        # Add multiple monitoring entries
        for i, date in enumerate(['2024-01-15', '2024-01-16', '2024-01-17']):
            predictions = np.random.uniform(70, 90, 5)
            actuals = np.random.uniform(70, 90, 5)
            self.validator.monitor_performance(predictions, actuals, timestamp=date)
        
        performance_history = self.validator.get_performance_history()
        
        assert len(performance_history) == 3
        assert all(date in performance_history for date in ['2024-01-15', '2024-01-16', '2024-01-17'])
    
    def test_model_quality_handles_edge_cases(self):
        """Test model quality validation handles edge cases."""
        # Test with all same predictions
        y_true = np.array([85.0, 72.5, 91.2])
        y_pred = np.array([80.0, 80.0, 80.0])  # All same prediction
        
        quality_report = self.validator.validate_model_quality(y_true, y_pred)
        
        assert 'mae' in quality_report
        assert quality_report['mae'] >= 0
        assert quality_report['quality_status'] in ['GOOD', 'FAIR', 'POOR', 'NEEDS_RETRAINING']
    
    def test_drift_detection_handles_categorical_features(self):
        """Test drift detection with categorical features."""
        train_categorical = self.train_data.copy()
        train_categorical['category'] = pd.Categorical(['A', 'B'] * 50)
        
        new_categorical = self.new_data.copy()
        new_categorical['category'] = pd.Categorical(['C', 'D'] * 25)  # New categories
        
        drift_report = self.validator.detect_data_drift(
            train_categorical,
            new_categorical,
            feature_columns=['pass_percentage', 'category']
        )
        
        assert 'drift_detected' in drift_report
        # Should detect drift due to new categories
        assert drift_report['drift_detected'] == True
    
    def test_performance_monitoring_calculates_trends(self):
        """Test that performance monitoring can calculate trends."""
        # Add data with degrading performance
        dates = ['2024-01-15', '2024-01-16', '2024-01-17']
        base_error = 2.0
        
        for i, date in enumerate(dates):
            # Gradually increase error
            error_increase = i * 5.0
            predictions = np.array([80.0, 75.0, 85.0])
            actuals = predictions + base_error + error_increase
            
            self.validator.monitor_performance(predictions, actuals, timestamp=date)
        
        trend_report = self.validator.analyze_performance_trend()
        
        assert 'trend_direction' in trend_report
        assert 'trend_significance' in trend_report
        assert trend_report['trend_direction'] in ['IMPROVING', 'DEGRADING', 'STABLE']
    
    def test_validation_thresholds_configurable(self):
        """Test that validation thresholds are configurable."""
        # Test with custom thresholds
        y_true = np.array([85.0, 72.5, 91.2])
        y_pred = np.array([82.0, 75.0, 88.0])
        
        custom_thresholds = {
            'mae_good': 5.0,  # More lenient threshold for test
            'mae_fair': 10.0,
            'mae_poor': 15.0
        }
        
        quality_report = self.validator.validate_model_quality(
            y_true, 
            y_pred, 
            thresholds=custom_thresholds
        )
        
        assert quality_report['quality_status'] == 'GOOD'  # Should meet strict threshold
    
    def test_drift_detection_with_missing_features(self):
        """Test drift detection handles missing features gracefully."""
        incomplete_data = self.new_data.drop(['lag_1_day', 'ma_3_day'], axis=1)
        
        drift_report = self.validator.detect_data_drift(
            self.train_data,
            incomplete_data,
            feature_columns=['pass_percentage', 'lag_1_day', 'ma_3_day']  # Some missing
        )
        
        assert 'drift_detected' in drift_report
        assert 'missing_features' in drift_report
        assert len(drift_report['missing_features']) == 2