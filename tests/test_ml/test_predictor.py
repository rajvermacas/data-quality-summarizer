"""
Test module for ML prediction service functionality.

This module tests the main prediction service that coordinates input validation,
feature engineering, model loading, and prediction generation.
"""

import pytest
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import pickle

from src.data_quality_summarizer.ml.predictor import Predictor


# Global dummy model classes for testing (needed for pickling)
class DummyModel:
    """Simple dummy model for testing."""
    def __init__(self, return_value=87.5):
        self.return_value = return_value
    
    def predict(self, X):
        return np.array([self.return_value])


class ConfigurableDummyModel:
    """Dummy model with configurable return value."""
    def __init__(self):
        self.return_value = 87.5
    
    def predict(self, X):
        return np.array([self.return_value])


class TestPredictor:
    """Test the Predictor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample historical data
        self.sample_data = pd.DataFrame({
            'dataset_uuid': ['abc123'] * 10,
            'rule_code': ['R001'] * 10,
            'business_date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'pass_percentage': [85.0, 87.5, 90.0, 82.3, 88.9, 91.2, 86.7, 89.1, 92.5, 88.0]
        })
        
        # Create a simple dummy model for testing
        self.mock_model = DummyModel(87.5)
        
        # Create temporary model file
        self.temp_model_file = tempfile.mktemp(suffix='.pkl')
        with open(self.temp_model_file, 'wb') as f:
            pickle.dump(self.mock_model, f)
        
        self.predictor = Predictor(
            model_path=self.temp_model_file,
            historical_data=self.sample_data
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        try:
            Path(self.temp_model_file).unlink()
        except FileNotFoundError:
            pass
    
    def test_predictor_initialization(self):
        """Test Predictor initializes correctly."""
        assert self.predictor is not None
        assert hasattr(self.predictor, 'model_path')
        assert hasattr(self.predictor, 'historical_data')
        assert hasattr(self.predictor, 'validator')
        assert hasattr(self.predictor, 'feature_pipeline')
        assert hasattr(self.predictor, 'predict')
        assert hasattr(self.predictor, 'load_model')
    
    def test_predictor_initialization_with_string_path(self):
        """Test Predictor initialization with string model path."""
        predictor = Predictor(
            model_path=str(self.temp_model_file),
            historical_data=self.sample_data
        )
        assert predictor.model_path == Path(self.temp_model_file)
    
    def test_predictor_initialization_with_path_object(self):
        """Test Predictor initialization with Path object."""
        predictor = Predictor(
            model_path=Path(self.temp_model_file),
            historical_data=self.sample_data
        )
        assert predictor.model_path == Path(self.temp_model_file)
    
    def test_load_model_success(self):
        """Test successful model loading."""
        model = self.predictor.load_model()
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_load_model_file_not_found(self):
        """Test model loading with non-existent file."""
        predictor = Predictor(
            model_path='nonexistent_model.pkl',
            historical_data=self.sample_data
        )
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            predictor.load_model()
    
    def test_predict_single_valid_input(self):
        """Test successful single prediction with valid inputs."""
        result = self.predictor.predict('abc123', 'R001', '2024-01-15')
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0
        assert result == 87.5  # Mock model returns 87.5
    
    def test_predict_with_integer_rule_code(self):
        """Test prediction with integer rule code."""
        result = self.predictor.predict('abc123', 1001, '2024-01-15')
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0
    
    def test_predict_with_date_object(self):
        """Test prediction with date object."""
        result = self.predictor.predict('abc123', 'R001', date(2024, 1, 15))
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0
    
    def test_predict_with_datetime_object(self):
        """Test prediction with datetime object."""
        result = self.predictor.predict('abc123', 'R001', datetime(2024, 1, 15))
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0
    
    def test_predict_invalid_dataset_uuid(self):
        """Test prediction with invalid dataset UUID."""
        with pytest.raises(ValueError, match="dataset_uuid cannot be empty"):
            self.predictor.predict('', 'R001', '2024-01-15')
    
    def test_predict_invalid_rule_code(self):
        """Test prediction with invalid rule code."""
        with pytest.raises(ValueError, match="rule_code cannot be empty"):
            self.predictor.predict('abc123', '', '2024-01-15')
    
    def test_predict_invalid_date_format(self):
        """Test prediction with invalid date format."""
        with pytest.raises(ValueError, match="Invalid date format"):
            self.predictor.predict('abc123', 'R001', 'invalid-date')
    
    def test_predict_none_inputs(self):
        """Test prediction with None inputs."""
        with pytest.raises(ValueError, match="dataset_uuid must be a string"):
            self.predictor.predict(None, 'R001', '2024-01-15')
        
        with pytest.raises(ValueError, match="rule_code must be a string or integer"):
            self.predictor.predict('abc123', None, '2024-01-15')
        
        with pytest.raises(ValueError, match="business_date cannot be None"):
            self.predictor.predict('abc123', 'R001', None)
    
    @patch('src.data_quality_summarizer.ml.predictor.Predictor.load_model')
    def test_predict_model_loading_error(self, mock_load_model):
        """Test prediction when model loading fails."""
        mock_load_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            self.predictor.predict('abc123', 'R001', '2024-01-15')
    
    def test_predict_no_historical_data(self):
        """Test prediction with no historical data available."""
        # Create predictor with empty historical data
        empty_predictor = Predictor(
            model_path=self.temp_model_file,
            historical_data=pd.DataFrame()
        )
        
        # Should still make prediction, just with NaN features
        result = empty_predictor.predict('xyz789', 'R999', '2024-01-15')
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0
    
    def test_predict_future_date(self):
        """Test prediction with future date."""
        future_date = '2030-12-31'
        result = self.predictor.predict('abc123', 'R001', future_date)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0
    
    def test_predict_very_old_date(self):
        """Test prediction with very old date."""
        old_date = '1900-01-01'
        result = self.predictor.predict('abc123', 'R001', old_date)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0


class TestPredictorIntegration:
    """Test integration between Predictor and other ML components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data with multiple datasets and rules
        self.integration_data = pd.DataFrame({
            'dataset_uuid': ['abc123'] * 5 + ['def456'] * 5,
            'rule_code': ['R001'] * 3 + ['R002'] * 2 + ['R001'] * 3 + ['R003'] * 2,
            'business_date': pd.date_range('2024-01-01', periods=5).tolist() + 
                           pd.date_range('2024-01-01', periods=5).tolist(),
            'pass_percentage': [85.0, 87.5, 90.0, 75.0, 88.0,
                              92.0, 89.5, 86.3, 91.2, 87.8]
        })
        
        # Create simple dummy model
        self.mock_model = DummyModel(88.5)
        
        # Create temporary model file
        self.temp_model_file = tempfile.mktemp(suffix='.pkl')
        with open(self.temp_model_file, 'wb') as f:
            pickle.dump(self.mock_model, f)
        
        self.predictor = Predictor(
            model_path=self.temp_model_file,
            historical_data=self.integration_data
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        try:
            Path(self.temp_model_file).unlink()
        except FileNotFoundError:
            pass
    
    def test_complete_prediction_workflow(self):
        """Test complete prediction workflow with real data flow."""
        # Test prediction for dataset/rule with historical data
        result1 = self.predictor.predict('abc123', 'R001', '2024-01-10')
        assert isinstance(result1, float)
        assert 0.0 <= result1 <= 100.0
        
        # Test prediction for different dataset/rule combination
        result2 = self.predictor.predict('def456', 'R003', '2024-01-10')
        assert isinstance(result2, float)
        assert 0.0 <= result2 <= 100.0
    
    def test_feature_engineering_integration(self):
        """Test that feature engineering produces expected structure."""
        # This test verifies that the complete feature pipeline works
        result = self.predictor.predict('abc123', 'R001', '2024-01-10')
        
        # Should successfully make prediction
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0
    
    def test_multiple_predictions_consistency(self):
        """Test that multiple predictions for same inputs are consistent."""
        # Make multiple predictions with same inputs
        result1 = self.predictor.predict('abc123', 'R001', '2024-01-10')
        result2 = self.predictor.predict('abc123', 'R001', '2024-01-10')
        
        # Results should be identical
        assert result1 == result2


class TestPredictorEdgeCases:
    """Test edge cases for the Predictor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create edge case data
        self.edge_data = pd.DataFrame({
            'dataset_uuid': ['test'],
            'rule_code': ['R001'],
            'business_date': [datetime(2024, 1, 1)],
            'pass_percentage': [100.0]  # Perfect pass rate
        })
        
        # Create simple dummy model that returns extreme values
        self.mock_model = ConfigurableDummyModel()
        
        # Create temporary model file
        self.temp_model_file = tempfile.mktemp(suffix='.pkl')
        with open(self.temp_model_file, 'wb') as f:
            pickle.dump(self.mock_model, f)
        
        self.predictor = Predictor(
            model_path=self.temp_model_file,
            historical_data=self.edge_data
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        try:
            Path(self.temp_model_file).unlink()
        except FileNotFoundError:
            pass
    
    def test_prediction_clipping_above_100(self):
        """Test that predictions above 100% are clipped."""
        # Create new model with extreme value
        extreme_model = DummyModel(150.0)
        temp_file = tempfile.mktemp(suffix='.pkl')
        with open(temp_file, 'wb') as f:
            pickle.dump(extreme_model, f)
        
        predictor = Predictor(temp_file, self.edge_data)
        result = predictor.predict('test', 'R001', '2024-01-15')
        assert result == 100.0
        
        Path(temp_file).unlink()
    
    def test_prediction_clipping_below_0(self):
        """Test that predictions below 0% are clipped."""
        # Create new model with extreme value
        extreme_model = DummyModel(-25.0)
        temp_file = tempfile.mktemp(suffix='.pkl')
        with open(temp_file, 'wb') as f:
            pickle.dump(extreme_model, f)
        
        predictor = Predictor(temp_file, self.edge_data)
        result = predictor.predict('test', 'R001', '2024-01-15')
        assert result == 0.0
        
        Path(temp_file).unlink()
    
    def test_prediction_with_nan_model_output(self):
        """Test handling of NaN model predictions."""
        # Create new model with NaN value
        extreme_model = DummyModel(np.nan)
        temp_file = tempfile.mktemp(suffix='.pkl')
        with open(temp_file, 'wb') as f:
            pickle.dump(extreme_model, f)
        
        predictor = Predictor(temp_file, self.edge_data)
        with pytest.raises(ValueError, match="Model returned invalid prediction"):
            predictor.predict('test', 'R001', '2024-01-15')
        
        Path(temp_file).unlink()
    
    def test_prediction_with_infinite_model_output(self):
        """Test handling of infinite model predictions."""
        # Create new model with infinite value
        extreme_model = DummyModel(np.inf)
        temp_file = tempfile.mktemp(suffix='.pkl')
        with open(temp_file, 'wb') as f:
            pickle.dump(extreme_model, f)
        
        predictor = Predictor(temp_file, self.edge_data)
        with pytest.raises(ValueError, match="Model returned invalid prediction"):
            predictor.predict('test', 'R001', '2024-01-15')
        
        Path(temp_file).unlink()
    
    def test_whitespace_handling_in_inputs(self):
        """Test that whitespace in inputs is handled correctly."""
        result = self.predictor.predict('  test  ', '  R001  ', '2024-01-15')
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0
    
    def test_year_boundary_predictions(self):
        """Test predictions at year boundaries."""
        # Test New Year's Day
        result1 = self.predictor.predict('test', 'R001', '2024-01-01')
        assert isinstance(result1, float)
        
        # Test New Year's Eve
        result2 = self.predictor.predict('test', 'R001', '2024-12-31')
        assert isinstance(result2, float)
    
    def test_leap_year_date_prediction(self):
        """Test prediction for leap year date."""
        result = self.predictor.predict('test', 'R001', '2024-02-29')
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0