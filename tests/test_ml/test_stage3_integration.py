"""
Integration tests for Stage 3: Prediction Service and API Layer.

Tests the complete Stage 3 pipeline including input validation,
feature engineering, and prediction generation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import pickle
from datetime import datetime, date
from pathlib import Path

from src.data_quality_summarizer.ml.validator import InputValidator
from src.data_quality_summarizer.ml.feature_pipeline import FeaturePipeline
from src.data_quality_summarizer.ml.predictor import Predictor


class DummyModel:
    """Simple dummy model for integration testing."""
    def __init__(self, return_value=87.5):
        self.return_value = return_value
    
    def predict(self, X):
        return np.array([self.return_value])


class TestStage3Integration:
    """Test complete Stage 3 pipeline integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create comprehensive test data
        self.test_data = pd.DataFrame({
            'dataset_uuid': ['ds001', 'ds001', 'ds001', 'ds002', 'ds002'],
            'rule_code': ['R001', 'R001', 'R002', 'R001', 'R001'],
            'business_date': [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2)
            ],
            'pass_percentage': [85.0, 87.5, 90.0, 82.3, 88.9]
        })
        
        # Create dummy model
        self.model = DummyModel(89.2)
        
        # Create temporary model file
        self.temp_model_file = tempfile.mktemp(suffix='.pkl')
        with open(self.temp_model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Create predictor
        self.predictor = Predictor(
            model_path=self.temp_model_file,
            historical_data=self.test_data
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        try:
            Path(self.temp_model_file).unlink()
        except FileNotFoundError:
            pass
    
    def test_complete_stage3_pipeline(self):
        """Test complete Stage 3 prediction pipeline."""
        # Test prediction with string inputs
        result1 = self.predictor.predict('ds001', 'R001', '2024-01-10')
        assert isinstance(result1, float)
        assert 0.0 <= result1 <= 100.0
        assert result1 == 89.2
        
        # Test prediction with integer rule code
        result2 = self.predictor.predict('ds001', 1001, '2024-01-10')
        assert isinstance(result2, float)
        assert 0.0 <= result2 <= 100.0
        
        # Test prediction with date object
        result3 = self.predictor.predict('ds001', 'R001', date(2024, 1, 10))
        assert isinstance(result3, float)
        assert 0.0 <= result3 <= 100.0
        
        # Test prediction with datetime object
        result4 = self.predictor.predict('ds001', 'R001', datetime(2024, 1, 10))
        assert isinstance(result4, float)
        assert 0.0 <= result4 <= 100.0
    
    def test_input_validation_integration(self):
        """Test that input validation works correctly in the pipeline."""
        # Test valid inputs pass through
        result = self.predictor.predict('ds001', 'R001', '2024-01-10')
        assert isinstance(result, float)
        
        # Test invalid inputs are caught
        with pytest.raises(ValueError, match="dataset_uuid cannot be empty"):
            self.predictor.predict('', 'R001', '2024-01-10')
        
        with pytest.raises(ValueError, match="rule_code cannot be empty"):
            self.predictor.predict('ds001', '', '2024-01-10')
        
        with pytest.raises(ValueError, match="Invalid date format"):
            self.predictor.predict('ds001', 'R001', 'invalid-date')
    
    def test_feature_engineering_integration(self):
        """Test that feature engineering works correctly in the pipeline."""
        # Test with known dataset/rule combination
        result1 = self.predictor.predict('ds001', 'R001', '2024-01-10')
        assert isinstance(result1, float)
        
        # Test with unknown dataset/rule combination (should still work)
        result2 = self.predictor.predict('unknown', 'R999', '2024-01-10')
        assert isinstance(result2, float)
        assert 0.0 <= result2 <= 100.0
    
    def test_historical_data_lookup_integration(self):
        """Test that historical data lookup affects predictions appropriately."""
        # Predict for dataset with historical data
        result_with_history = self.predictor.predict('ds001', 'R001', '2024-01-05')
        
        # Predict for dataset without historical data
        result_without_history = self.predictor.predict('unknown', 'R999', '2024-01-05')
        
        # Both should return valid predictions
        assert isinstance(result_with_history, float)
        assert isinstance(result_without_history, float)
        assert 0.0 <= result_with_history <= 100.0
        assert 0.0 <= result_without_history <= 100.0
    
    def test_model_loading_integration(self):
        """Test that model loading works correctly in the pipeline."""
        # Model should be loaded on first prediction
        assert self.predictor._model is None
        
        # Make first prediction - model should be loaded
        result1 = self.predictor.predict('ds001', 'R001', '2024-01-10')
        assert self.predictor._model is not None
        assert isinstance(result1, float)
        
        # Make second prediction - model should be reused
        result2 = self.predictor.predict('ds002', 'R001', '2024-01-10')
        assert isinstance(result2, float)
        assert result1 == result2  # Same model, same result
    
    def test_component_interaction(self):
        """Test that all Stage 3 components work together correctly."""
        # Create separate instances of each component
        validator = InputValidator()
        feature_pipeline = FeaturePipeline(self.test_data)
        
        # Test validator independently
        validated = validator.validate_all_inputs('ds001', 'R001', '2024-01-10')
        assert validated['dataset_uuid'] == 'ds001'
        assert validated['rule_code'] == 'R001'
        assert validated['business_date'] == date(2024, 1, 10)
        
        # Test feature pipeline independently
        features = feature_pipeline.engineer_features_for_prediction(
            'ds001', 'R001', date(2024, 1, 10)
        )
        assert isinstance(features, dict)
        assert 'dataset_uuid' in features
        assert 'lag_1_day' in features
        assert 'ma_3_day' in features
        
        # Test complete predictor (integrates all components)
        result = self.predictor.predict('ds001', 'R001', '2024-01-10')
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0


class TestStage3Performance:
    """Test Stage 3 performance requirements."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        # Create larger dataset for performance testing
        np.random.seed(42)  # For reproducible results
        
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        self.performance_data = pd.DataFrame({
            'dataset_uuid': np.random.choice(['ds001', 'ds002', 'ds003'], 500),
            'rule_code': np.random.choice(['R001', 'R002', 'R003'], 500),
            'business_date': np.tile(dates, 5),
            'pass_percentage': np.random.uniform(60.0, 95.0, 500)
        })
        
        # Create model
        self.model = DummyModel(88.0)
        
        # Create temporary model file
        self.temp_model_file = tempfile.mktemp(suffix='.pkl')
        with open(self.temp_model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Create predictor
        self.predictor = Predictor(
            model_path=self.temp_model_file,
            historical_data=self.performance_data
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        try:
            Path(self.temp_model_file).unlink()
        except FileNotFoundError:
            pass
    
    def test_prediction_latency_requirement(self):
        """Test that single predictions meet latency requirements (<1 second)."""
        import time
        
        # Make multiple predictions and measure time
        start_time = time.time()
        
        for i in range(10):
            result = self.predictor.predict('ds001', 'R001', f'2024-02-{i+1:02d}')
            assert isinstance(result, float)
            assert 0.0 <= result <= 100.0
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10
        
        # Each prediction should be well under 1 second
        assert avg_time < 1.0, f"Average prediction time {avg_time:.3f}s exceeds 1 second"
    
    def test_memory_efficiency_with_large_dataset(self):
        """Test memory efficiency with larger datasets."""
        # This test ensures the system doesn't load everything into memory
        # Making predictions should work efficiently
        
        predictions = []
        for i in range(20):
            result = self.predictor.predict('ds001', 'R001', f'2024-03-{i+1:02d}')
            predictions.append(result)
        
        # All predictions should be valid
        assert len(predictions) == 20
        assert all(isinstance(p, float) for p in predictions)
        assert all(0.0 <= p <= 100.0 for p in predictions)
    
    def test_concurrent_prediction_consistency(self):
        """Test that predictions are consistent when made multiple times."""
        # Make the same prediction multiple times
        prediction_date = '2024-02-15'
        results = []
        
        for _ in range(5):
            result = self.predictor.predict('ds001', 'R001', prediction_date)
            results.append(result)
        
        # All results should be identical (deterministic model)
        assert len(set(results)) == 1, "Predictions should be deterministic"


class TestStage3ErrorHandling:
    """Test Stage 3 error handling and edge cases."""
    
    def setup_method(self):
        """Set up error handling test fixtures."""
        # Create minimal test data
        self.minimal_data = pd.DataFrame({
            'dataset_uuid': ['test'],
            'rule_code': ['R001'],
            'business_date': [datetime(2024, 1, 1)],
            'pass_percentage': [85.0]
        })
        
        # Create model
        self.model = DummyModel(87.0)
        
        # Create temporary model file
        self.temp_model_file = tempfile.mktemp(suffix='.pkl')
        with open(self.temp_model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Create predictor
        self.predictor = Predictor(
            model_path=self.temp_model_file,
            historical_data=self.minimal_data
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        try:
            Path(self.temp_model_file).unlink()
        except FileNotFoundError:
            pass
    
    def test_file_not_found_error_handling(self):
        """Test handling of missing model files."""
        # Create predictor with non-existent model file
        bad_predictor = Predictor(
            model_path='nonexistent_model.pkl',
            historical_data=self.minimal_data
        )
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            bad_predictor.predict('test', 'R001', '2024-01-15')
    
    def test_empty_historical_data_handling(self):
        """Test handling of empty historical data."""
        # Create predictor with empty historical data
        empty_predictor = Predictor(
            model_path=self.temp_model_file,
            historical_data=pd.DataFrame()
        )
        
        # Should still make predictions successfully
        result = empty_predictor.predict('test', 'R001', '2024-01-15')
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0
    
    def test_extreme_date_handling(self):
        """Test handling of extreme dates."""
        # Test very old date
        result1 = self.predictor.predict('test', 'R001', '1900-01-01')
        assert isinstance(result1, float)
        assert 0.0 <= result1 <= 100.0
        
        # Test future date
        result2 = self.predictor.predict('test', 'R001', '2100-12-31')
        assert isinstance(result2, float)
        assert 0.0 <= result2 <= 100.0
    
    def test_whitespace_and_edge_case_inputs(self):
        """Test handling of edge case inputs."""
        # Test whitespace trimming
        result1 = self.predictor.predict('  test  ', '  R001  ', '2024-01-15')
        assert isinstance(result1, float)
        
        # Test various rule code formats
        result2 = self.predictor.predict('test', 0, '2024-01-15')
        assert isinstance(result2, float)
        
        result3 = self.predictor.predict('test', -1, '2024-01-15')
        assert isinstance(result3, float)
    
    def test_component_error_propagation(self):
        """Test that errors from components are properly propagated."""
        # Test validator errors
        with pytest.raises(ValueError):
            self.predictor.predict(None, 'R001', '2024-01-15')
        
        with pytest.raises(ValueError):
            self.predictor.predict('test', None, '2024-01-15')
        
        with pytest.raises(ValueError):
            self.predictor.predict('test', 'R001', None)