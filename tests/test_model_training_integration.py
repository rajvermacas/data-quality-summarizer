"""Test integration of data validation into model training pipeline."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from src.data_quality_summarizer.ml.model_trainer import train_lightgbm_model_with_validation
from src.data_quality_summarizer.ml.data_validator import DataQualityException


class TestModelTrainingIntegration:
    """Test cases for integrated model training with validation."""
    
    @pytest.fixture
    def valid_training_data(self):
        """Create valid training data for testing."""
        np.random.seed(42)
        data = []
        
        # Create sufficient data per group with good variance
        for dataset_uuid in ['uuid1', 'uuid2']:
            for rule_code in ['R001', 'R002']:
                for i in range(30):  # 30 samples per group (>20 threshold)
                    data.append({
                        'dataset_uuid': dataset_uuid,
                        'rule_code': rule_code,
                        'pass_percentage': np.random.uniform(10, 90),  # Good variance
                        'feature1': np.random.normal(50, 10),
                        'feature2': np.random.normal(0, 1),
                        'feature3': np.random.uniform(0, 100)
                    })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def invalid_training_data_low_variance(self):
        """Create training data with low target variance."""
        data = []
        
        for i in range(100):
            data.append({
                'dataset_uuid': 'uuid1',
                'rule_code': 'R001',
                'pass_percentage': 50.0 + np.random.normal(0, 0.01),  # Very low variance
                'feature1': np.random.normal(50, 10),
                'feature2': np.random.normal(0, 1)
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def invalid_training_data_insufficient_samples(self):
        """Create training data with insufficient samples per group."""
        data = []
        
        for i in range(10):  # Only 10 samples total (insufficient)
            data.append({
                'dataset_uuid': 'uuid1',
                'rule_code': 'R001',
                'pass_percentage': np.random.uniform(0, 100),
                'feature1': np.random.normal(50, 10),
                'feature2': np.random.normal(0, 1)
            })
        
        return pd.DataFrame(data)
    
    def test_train_lightgbm_model_with_validation_function_exists(self):
        """Test that enhanced training function exists."""
        # This should fail initially since the function doesn't exist yet
        from src.data_quality_summarizer.ml.model_trainer import train_lightgbm_model_with_validation
        
        assert callable(train_lightgbm_model_with_validation)
    
    def test_training_with_validation_succeeds_with_good_data(self, valid_training_data):
        """Test that training succeeds with valid data."""
        feature_cols = ['feature1', 'feature2', 'feature3']
        categorical_cols = ['dataset_uuid', 'rule_code']
        target_col = 'pass_percentage'
        
        # Should complete without raising validation errors
        model = train_lightgbm_model_with_validation(
            data=valid_training_data,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            target_col=target_col
        )
        
        # Should return a trained model
        assert model is not None
        assert hasattr(model, 'predict')  # LightGBM model interface
    
    def test_training_validation_fails_with_low_variance(self, invalid_training_data_low_variance):
        """Test that training fails validation with low target variance."""
        feature_cols = ['feature1', 'feature2']
        categorical_cols = ['dataset_uuid', 'rule_code']
        target_col = 'pass_percentage'
        
        with pytest.raises(DataQualityException, match="Insufficient target variable variance"):
            train_lightgbm_model_with_validation(
                data=invalid_training_data_low_variance,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                target_col=target_col
            )
    
    def test_training_validation_fails_with_insufficient_samples(self, invalid_training_data_insufficient_samples):
        """Test that training fails validation with insufficient samples."""
        feature_cols = ['feature1', 'feature2']
        categorical_cols = ['dataset_uuid', 'rule_code']
        target_col = 'pass_percentage'
        
        with pytest.raises(DataQualityException, match="Insufficient sample size"):
            train_lightgbm_model_with_validation(
                data=invalid_training_data_insufficient_samples,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                target_col=target_col
            )
    
    def test_validation_report_generated(self, valid_training_data):
        """Test that validation report is generated during training."""
        feature_cols = ['feature1', 'feature2', 'feature3']
        categorical_cols = ['dataset_uuid', 'rule_code']
        target_col = 'pass_percentage'
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_diagnostics_dir = Path(temp_dir) / "model_diagnostics"
            
            model = train_lightgbm_model_with_validation(
                data=valid_training_data,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                target_col=target_col,
                model_diagnostics_dir=model_diagnostics_dir
            )
            
            # Check that validation report was created
            report_path = model_diagnostics_dir / "data_quality_report.json"
            assert report_path.exists()
    
    def test_validation_with_custom_thresholds(self, valid_training_data):
        """Test training with custom validation thresholds."""
        feature_cols = ['feature1', 'feature2', 'feature3']
        categorical_cols = ['dataset_uuid', 'rule_code']
        target_col = 'pass_percentage'
        
        # Use stricter thresholds
        validation_config = {
            'min_variance': 0.5,  # Stricter than default 0.1
            'min_samples_per_group': 25  # Stricter than default 20
        }
        
        model = train_lightgbm_model_with_validation(
            data=valid_training_data,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            target_col=target_col,
            validation_config=validation_config
        )
        
        assert model is not None