"""
Test suite for ML Model Trainer module.

This module tests the LightGBM model training functionality
for the predictive model pipeline.
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from src.data_quality_summarizer.ml.model_trainer import (
    ModelTrainer,
    train_lightgbm_model,
    save_model,
    load_model,
    prepare_categorical_features,
    get_default_lgb_params
)


class TestModelTrainer(unittest.TestCase):
    """Test LightGBM model training functionality."""

    def setUp(self):
        """Create test data for model training tests."""
        # Create sample training data
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        
        self.train_data = pd.DataFrame({
            'dataset_uuid': ['dataset1'] * 50 + ['dataset2'] * 50,
            'rule_code': ['R001'] * 25 + ['R002'] * 25 + ['R001'] * 25 + ['R002'] * 25,
            'business_date': dates,
            'pass_percentage': np.random.uniform(70, 95, 100),  # Target variable
            'day_of_week': [d.weekday() for d in dates],
            'month': [d.month for d in dates],
            'lag_1_day': np.random.uniform(65, 90, 100),
            'lag_7_day': np.random.uniform(60, 95, 100),
            'ma_3_day': np.random.uniform(68, 92, 100),
            'ma_7_day': np.random.uniform(65, 90, 100)
        })
        
        # Create test data (smaller)
        test_dates = [datetime(2024, 4, 1) + timedelta(days=i) for i in range(20)]
        self.test_data = pd.DataFrame({
            'dataset_uuid': ['dataset1'] * 10 + ['dataset2'] * 10,
            'rule_code': ['R001'] * 5 + ['R002'] * 5 + ['R001'] * 5 + ['R002'] * 5,
            'business_date': test_dates,
            'pass_percentage': np.random.uniform(70, 95, 20),
            'day_of_week': [d.weekday() for d in test_dates],
            'month': [d.month for d in test_dates],
            'lag_1_day': np.random.uniform(65, 90, 20),
            'lag_7_day': np.random.uniform(60, 95, 20),
            'ma_3_day': np.random.uniform(68, 92, 20),
            'ma_7_day': np.random.uniform(65, 90, 20)
        })

    def test_model_trainer_initialization(self):
        """Test ModelTrainer class initialization."""
        trainer = ModelTrainer()
        
        self.assertIsNone(trainer.model)
        self.assertIsNotNone(trainer.params)
        self.assertIn('objective', trainer.params)
        self.assertEqual(trainer.params['objective'], 'regression')

    def test_model_trainer_with_custom_params(self):
        """Test ModelTrainer initialization with custom parameters."""
        custom_params = {
            'objective': 'regression',
            'num_leaves': 50,
            'learning_rate': 0.1
        }
        
        trainer = ModelTrainer(custom_params)
        self.assertEqual(trainer.params['num_leaves'], 50)
        self.assertEqual(trainer.params['learning_rate'], 0.1)

    def test_train_lightgbm_model_basic(self):
        """Test basic LightGBM model training."""
        feature_cols = ['day_of_week', 'month', 'lag_1_day', 'lag_7_day', 'ma_3_day', 'ma_7_day']
        categorical_cols = ['dataset_uuid', 'rule_code']
        target_col = 'pass_percentage'
        
        model = train_lightgbm_model(
            self.train_data, feature_cols, categorical_cols, target_col
        )
        
        self.assertIsNotNone(model)
        # Test that model can make predictions using prepared data
        test_data_prepared = prepare_categorical_features(
            self.test_data.copy(), categorical_cols
        )
        predictions = model.predict(test_data_prepared[feature_cols + categorical_cols])
        self.assertEqual(len(predictions), len(self.test_data))
        
        # Predictions should be in reasonable range (clipping will happen in trainer)
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))

    def test_prepare_categorical_features(self):
        """Test categorical feature preparation."""
        data = self.train_data.copy()
        categorical_cols = ['dataset_uuid', 'rule_code']
        
        prepared_data = prepare_categorical_features(data, categorical_cols)
        
        # Check that categorical columns are now category type
        for col in categorical_cols:
            self.assertEqual(prepared_data[col].dtype.name, 'category')
        
        # Non-categorical columns should remain unchanged
        self.assertEqual(prepared_data['pass_percentage'].dtype, np.float64)

    def test_get_default_lgb_params(self):
        """Test default LightGBM parameters."""
        params = get_default_lgb_params()
        
        required_params = ['objective', 'metric', 'num_leaves', 'learning_rate', 'verbosity']
        for param in required_params:
            self.assertIn(param, params)
        
        self.assertEqual(params['objective'], 'regression')
        self.assertEqual(params['metric'], 'mae')

    def test_model_trainer_fit_method(self):
        """Test ModelTrainer fit method."""
        trainer = ModelTrainer()
        feature_cols = ['day_of_week', 'month', 'lag_1_day', 'lag_7_day']
        categorical_cols = ['dataset_uuid', 'rule_code']
        target_col = 'pass_percentage'
        
        trained_model = trainer.fit(
            self.train_data, feature_cols, categorical_cols, target_col
        )
        
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trained_model, trainer.model)

    def test_model_trainer_predict_method(self):
        """Test ModelTrainer predict method."""
        trainer = ModelTrainer()
        feature_cols = ['day_of_week', 'month', 'lag_1_day', 'lag_7_day']
        categorical_cols = ['dataset_uuid', 'rule_code']
        target_col = 'pass_percentage'
        
        # Train first
        trainer.fit(self.train_data, feature_cols, categorical_cols, target_col)
        
        # Then predict
        predictions = trainer.predict(self.test_data)
        
        self.assertEqual(len(predictions), len(self.test_data))
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))

    def test_model_serialization(self):
        """Test model save and load functionality."""
        trainer = ModelTrainer()
        feature_cols = ['day_of_week', 'month', 'lag_1_day']
        categorical_cols = ['dataset_uuid', 'rule_code']
        target_col = 'pass_percentage'
        
        # Train model
        trainer.fit(self.train_data, feature_cols, categorical_cols, target_col)
        original_predictions = trainer.predict(self.test_data)
        
        # Save and load model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            save_model(trainer.model, model_path)
            loaded_model = load_model(model_path)
            
            # Test that loaded model makes same predictions
            test_data_prepared = prepare_categorical_features(
                self.test_data.copy(), categorical_cols
            )
            loaded_predictions = loaded_model.predict(
                test_data_prepared[feature_cols + categorical_cols]
            )
            
            np.testing.assert_array_almost_equal(
                original_predictions, loaded_predictions, decimal=5
            )
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestModelTrainerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for model trainer."""

    def test_empty_training_data(self):
        """Test handling of empty training data."""
        empty_data = pd.DataFrame(columns=[
            'dataset_uuid', 'rule_code', 'pass_percentage', 'day_of_week'
        ])
        
        with self.assertRaises(ValueError):
            train_lightgbm_model(
                empty_data, ['day_of_week'], ['dataset_uuid'], 'pass_percentage'
            )

    def test_missing_target_column(self):
        """Test handling of missing target column."""
        data = pd.DataFrame({
            'dataset_uuid': ['dataset1'],
            'day_of_week': [1]
        })
        
        with self.assertRaises(KeyError):
            train_lightgbm_model(
                data, ['day_of_week'], ['dataset_uuid'], 'missing_target'
            )

    def test_missing_feature_columns(self):
        """Test handling of missing feature columns."""
        data = pd.DataFrame({
            'dataset_uuid': ['dataset1'],
            'pass_percentage': [85.0]
        })
        
        with self.assertRaises(KeyError):
            train_lightgbm_model(
                data, ['missing_feature'], ['dataset_uuid'], 'pass_percentage'
            )

    def test_invalid_model_path_for_save(self):
        """Test handling of invalid model save path."""
        # Create a simple trained model
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [10, 20, 30]
        })
        
        model = train_lightgbm_model(data, ['feature1'], [], 'target')
        
        # Try to save to directory that doesn't exist (and can't be created)
        with self.assertRaises((OSError, IOError, PermissionError)):
            save_model(model, '/root/nonexistent/deep/path/model.pkl')

    def test_load_nonexistent_model(self):
        """Test loading non-existent model file."""
        with self.assertRaises(FileNotFoundError):
            load_model('/nonexistent/model.pkl')


class TestModelTrainerIntegration(unittest.TestCase):
    """Integration tests for complete model training workflow."""

    def test_complete_training_workflow(self):
        """Test complete training workflow from data to predictions."""
        # Create more realistic dataset
        np.random.seed(42)  # For reproducible results
        
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(200)]
        data = pd.DataFrame({
            'dataset_uuid': np.random.choice(['dataset1', 'dataset2', 'dataset3'], 200),
            'rule_code': np.random.choice(['R001', 'R002', 'R003'], 200),
            'business_date': dates,
            'day_of_week': [d.weekday() for d in dates],
            'month': [d.month for d in dates],
            'lag_1_day': np.random.uniform(60, 95, 200),
            'ma_7_day': np.random.uniform(65, 90, 200),
        })
        
        # Create realistic target with some correlation to features
        data['pass_percentage'] = (
            70 + 
            data['lag_1_day'] * 0.3 + 
            data['ma_7_day'] * 0.2 + 
            np.random.normal(0, 5, 200)
        )
        data['pass_percentage'] = np.clip(data['pass_percentage'], 0, 100)
        
        # Train model
        trainer = ModelTrainer()
        feature_cols = ['day_of_week', 'month', 'lag_1_day', 'ma_7_day']
        categorical_cols = ['dataset_uuid', 'rule_code']
        target_col = 'pass_percentage'
        
        trainer.fit(data, feature_cols, categorical_cols, target_col)
        
        # Make predictions on same data (just for testing)
        predictions = trainer.predict(data)
        
        # Basic sanity checks
        self.assertEqual(len(predictions), len(data))
        self.assertTrue(all(0 <= p <= 100 for p in predictions))
        
        # Model should have learned something (correlation > 0.3)
        correlation = np.corrcoef(data['pass_percentage'], predictions)[0, 1]
        self.assertGreater(abs(correlation), 0.3)


if __name__ == '__main__':
    unittest.main()