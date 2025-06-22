"""
Interface Integration Tests.

Tests for interface compatibility between pipeline and components.
These tests expose the exact interface mismatches preventing pipeline execution.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from src.data_quality_summarizer.ml.model_trainer import ModelTrainer
from src.data_quality_summarizer.ml.evaluator import ModelEvaluator
from src.data_quality_summarizer.ml.pipeline import MLPipeline


class TestInterfaceIntegration:
    """Test interface compatibility between components."""
    
    def test_model_trainer_train_method_exists(self):
        """Test that ModelTrainer has train() method expected by pipeline."""
        trainer = ModelTrainer()
        assert hasattr(trainer, 'train'), "ModelTrainer missing train() method"
    
    def test_model_trainer_train_signature(self):
        """Test train() method has correct signature for pipeline interface."""
        trainer = ModelTrainer()
        
        # Create test data in expected format
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'dataset_uuid': ['uuid1', 'uuid2', 'uuid3'],
            'rule_code': [1, 2, 1]
        })
        y_train = pd.Series([90.0, 85.0, 95.0])
        model_params = {'n_estimators': 10}
        
        # This should work after fix
        try:
            model = trainer.train(X_train, y_train, model_params=model_params)
            assert model is not None
        except AttributeError as e:
            pytest.fail(f"train() method interface mismatch: {e}")
    
    def test_model_evaluator_evaluate_method_exists(self):
        """Test that ModelEvaluator has evaluate() method expected by pipeline."""
        evaluator = ModelEvaluator()
        assert hasattr(evaluator, 'evaluate'), "ModelEvaluator missing evaluate() method"
    
    def test_model_evaluator_evaluate_signature(self):
        """Test evaluate() method has correct signature for pipeline interface."""
        evaluator = ModelEvaluator()
        
        # Create mock model with predict method
        mock_model = Mock()
        mock_model.predict.return_value = np.array([90.0, 85.0, 95.0])
        
        # Create test data
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        y_test = pd.Series([88.0, 83.0, 97.0])
        
        # This should work after fix
        try:
            metrics = evaluator.evaluate(mock_model, X_test, y_test)
            assert isinstance(metrics, dict)
            assert 'mae' in metrics
        except AttributeError as e:
            pytest.fail(f"evaluate() method interface mismatch: {e}")
    
    def test_pipeline_calls_correct_trainer_methods(self):
        """Test pipeline can call ModelTrainer.train() successfully."""
        pipeline = MLPipeline()
        # Simulate the call from pipeline.py:147
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [3, 4, 5, 6, 7],
            'dataset_uuid': ['uuid1', 'uuid1', 'uuid2', 'uuid2', 'uuid1'],
            'rule_code': [1, 1, 2, 2, 1]
        })
        y_train = pd.Series([90.0, 85.0, 88.0, 92.0, 87.0])
        
        # This should now work without AttributeError
        model = pipeline.model_trainer.train(X_train, y_train, model_params={'n_estimators': 10})
        assert model is not None
    
    def test_pipeline_calls_correct_evaluator_methods(self):
        """Test pipeline can call ModelEvaluator.evaluate() successfully."""
        pipeline = MLPipeline()
        
        # Create a simple mock model that returns reasonable predictions
        mock_model = Mock()
        mock_model.predict.return_value = np.array([90.0, 85.0])
        
        X_test = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        y_test = pd.Series([88.0, 83.0])
        
        # This should now work without AttributeError
        metrics = pipeline.evaluator.evaluate(mock_model, X_test, y_test)
        assert isinstance(metrics, dict)
        assert 'mae' in metrics
        assert 'rmse' in metrics