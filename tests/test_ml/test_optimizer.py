"""
Tests for Performance Optimizer module.

This module tests memory usage optimization, training time optimization,
and prediction latency optimization capabilities.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import time
from unittest.mock import Mock, patch
from src.data_quality_summarizer.ml.optimizer import PerformanceOptimizer


class TestPerformanceOptimizer:
    """Test cases for PerformanceOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = PerformanceOptimizer()
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'dataset_uuid': ['uuid1', 'uuid2'] * 50,
            'rule_code': ['R001', 'R002'] * 50,
            'business_date': pd.date_range('2024-01-01', periods=100),
            'pass_percentage': np.random.uniform(0, 100, 100),
            'day_of_week': [1, 2] * 50,
            'month': [1, 2] * 50,
            'lag_1_day': np.random.uniform(0, 100, 100),
            'lag_7_day': np.random.uniform(0, 100, 100),
            'ma_3_day': np.random.uniform(0, 100, 100),
            'ma_7_day': np.random.uniform(0, 100, 100)
        })
    
    def test_optimizer_initialization(self):
        """Test that optimizer initializes correctly."""
        assert self.optimizer is not None
        assert hasattr(self.optimizer, 'optimize_memory_usage')
        assert hasattr(self.optimizer, 'optimize_training_time')
        assert hasattr(self.optimizer, 'optimize_prediction_latency')
    
    def test_memory_optimization_reduces_usage(self):
        """Test that memory optimization reduces memory footprint."""
        # Create large dataset to test memory optimization
        large_data = pd.DataFrame({
            'dataset_uuid': ['uuid' + str(i) for i in range(10000)],
            'rule_code': ['R' + str(i % 100) for i in range(10000)],
            'business_date': pd.date_range('2024-01-01', periods=10000),
            'pass_percentage': np.random.uniform(0, 100, 10000),
            'text_features': ['feature_' + str(i) for i in range(10000)]  # Memory-heavy column
        })
        
        original_memory = large_data.memory_usage(deep=True).sum()
        optimized_data = self.optimizer.optimize_memory_usage(large_data)
        optimized_memory = optimized_data.memory_usage(deep=True).sum()
        
        assert optimized_memory < original_memory
        assert optimized_data.shape == large_data.shape  # Same data, less memory
    
    def test_training_time_optimization_config(self):
        """Test that training time optimization returns optimized config."""
        base_config = {
            'objective': 'regression',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9
        }
        
        optimized_config = self.optimizer.optimize_training_time(
            base_config, 
            data_size=10000
        )
        
        assert 'num_threads' in optimized_config
        assert 'verbosity' in optimized_config
        assert optimized_config['num_threads'] > 0
        assert optimized_config['learning_rate'] >= base_config['learning_rate']
    
    def test_prediction_latency_optimization(self):
        """Test that prediction latency optimization improves speed."""
        # Mock predictor with predict method
        mock_predictor = Mock()
        mock_predictor.predict.return_value = np.array([85.5])
        
        sample_input = self.sample_data.iloc[0:1].drop('pass_percentage', axis=1)
        
        # Time original prediction
        start_time = time.time()
        mock_predictor.predict(sample_input)
        original_time = time.time() - start_time
        
        # Apply optimization
        optimized_predictor = self.optimizer.optimize_prediction_latency(mock_predictor)
        
        # Time optimized prediction
        start_time = time.time()
        result = optimized_predictor.predict(sample_input)
        optimized_time = time.time() - start_time
        
        assert result is not None
        assert optimized_time <= original_time * 1.1  # Allow small variance
    
    def test_memory_optimization_preserves_data_integrity(self):
        """Test that memory optimization doesn't corrupt data."""
        original_data = self.sample_data.copy()
        optimized_data = self.optimizer.optimize_memory_usage(original_data)
        
        # Check that all values are preserved (allowing for dtype changes)
        for col in original_data.columns:
            if col in optimized_data.columns:
                if pd.api.types.is_numeric_dtype(original_data[col]):
                    np.testing.assert_allclose(
                        original_data[col].values,
                        optimized_data[col].values,
                        rtol=1e-10
                    )
                else:
                    # For categorical/object columns, check values match (allow dtype change)
                    pd.testing.assert_series_equal(
                        original_data[col].astype(str), 
                        optimized_data[col].astype(str),
                        check_dtype=False
                    )
    
    def test_training_optimization_with_small_dataset(self):
        """Test training optimization handles small datasets appropriately."""
        small_config = {
            'objective': 'regression',
            'num_leaves': 31,
            'learning_rate': 0.1
        }
        
        optimized_config = self.optimizer.optimize_training_time(
            small_config,
            data_size=100  # Small dataset
        )
        
        assert optimized_config['num_leaves'] <= small_config['num_leaves']
        assert 'early_stopping_rounds' in optimized_config
    
    def test_prediction_optimization_batch_processing(self):
        """Test prediction optimization for batch operations."""
        mock_predictor = Mock()
        mock_predictor.predict.return_value = np.array([85.5, 72.3, 91.2])
        
        batch_input = self.sample_data.iloc[0:3].drop('pass_percentage', axis=1)
        
        optimized_predictor = self.optimizer.optimize_prediction_latency(mock_predictor)
        results = optimized_predictor.predict(batch_input)
        
        assert len(results) == 3
        assert all(isinstance(x, (int, float, np.number)) for x in results)
    
    def test_memory_optimization_handles_mixed_dtypes(self):
        """Test memory optimization with mixed data types."""
        mixed_data = pd.DataFrame({
            'strings': ['a', 'b', 'c'] * 100,
            'integers': list(range(300)),
            'floats': np.random.random(300),
            'booleans': [True, False] * 150,
            'categories': pd.Categorical(['cat1', 'cat2', 'cat3'] * 100)
        })
        
        optimized_data = self.optimizer.optimize_memory_usage(mixed_data)
        
        assert optimized_data.shape == mixed_data.shape
        assert optimized_data.memory_usage(deep=True).sum() <= mixed_data.memory_usage(deep=True).sum()
    
    def test_optimization_config_validation(self):
        """Test that optimizer validates configuration parameters."""
        invalid_config = {
            'objective': 'invalid_objective',
            'num_leaves': -5,  # Invalid negative value
            'learning_rate': 2.0  # Invalid rate > 1
        }
        
        optimized_config = self.optimizer.optimize_training_time(invalid_config, 1000)
        
        # Should correct invalid values
        assert optimized_config['num_leaves'] > 0
        assert optimized_config['learning_rate'] <= 1.0
        assert optimized_config['objective'] in ['regression', 'regression_l1', 'regression_l2']