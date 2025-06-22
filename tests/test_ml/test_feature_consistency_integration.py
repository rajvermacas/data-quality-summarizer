"""
Integration tests for feature engineering consistency between training and prediction.
This addresses the critical issue where training uses 11 features and prediction uses 9.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from src.data_quality_summarizer.ml.predictor import Predictor
from src.data_quality_summarizer.ml.pipeline import MLPipeline


class TestFeatureEngineeringConsistency:
    """Test that training and prediction use the same feature engineering pipeline."""
    
    def test_training_prediction_feature_count_consistency(self):
        """
        CRITICAL TEST: Training and prediction must use the same number of features.
        
        This test reproduces the exact issue found in QA analysis:
        - CLI training works and saves model with 11 features
        - CLI prediction fails with 9 vs 11 feature mismatch
        
        This test uses the existing test data files to reproduce the real issue.
        """
        # Use existing test files that are known to work
        data_file = '/root/projects/data-quality-summarizer/test_ml_data.csv'
        rules_file = '/root/projects/data-quality-summarizer/test_rules.json'
        
        # Verify test files exist
        assert os.path.exists(data_file), f"Test data file not found: {data_file}"
        assert os.path.exists(rules_file), f"Test rules file not found: {rules_file}"
        
        # Read test data for predictor
        historical_data = pd.read_csv(data_file)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'test_model.pkl')
            
            # Step 1: Train model using CLI (this works according to QA analysis)
            train_cmd = f"python -m src.data_quality_summarizer train-model {data_file} {rules_file} --output-model {model_path}"
            import subprocess
            result = subprocess.run(train_cmd, shell=True, capture_output=True, text=True, cwd='/root/projects/data-quality-summarizer')
            
            print(f"Training command: {train_cmd}")
            print(f"Training exit code: {result.returncode}")
            print(f"Training stdout: {result.stdout}")
            if result.stderr:
                print(f"Training stderr: {result.stderr}")
            
            # Step 2: Try prediction using Predictor class (this should fail)
            if os.path.exists(model_path):
                predictor = Predictor(model_path, historical_data)
                
                try:
                    prediction = predictor.predict(
                        dataset_uuid='dataset-001',
                        rule_code=1,
                        business_date='2024-04-15'
                    )
                    prediction_succeeded = True
                except Exception as e:
                    prediction_succeeded = False
                    error_message = str(e)
                    print(f"Prediction error: {error_message}")
                    
                    # This should show the exact feature mismatch error
                    assert "number of features in data" in error_message, (
                        f"Expected feature mismatch error, got: {error_message}"
                    )
                    
                    # Extract feature counts from error message
                    import re
                    prediction_match = re.search(r'features in data \((\d+)\)', error_message)
                    training_match = re.search(r'training data \((\d+)\)', error_message)
                    
                    prediction_features = int(prediction_match.group(1)) if prediction_match else None
                    training_features = int(training_match.group(1)) if training_match else None
                    
                    print(f"Feature mismatch detected: prediction={prediction_features}, training={training_features}")
                    
                    # Verify this is the exact issue we expect
                    assert prediction_features == 9, f"Expected 9 prediction features, got {prediction_features}"
                    assert training_features == 11, f"Expected 11 training features, got {training_features}"
            else:
                pytest.fail(f"Model training failed - model file not created: {model_path}")
            
            # The test should fail here because prediction_succeeded should be False
            assert prediction_succeeded, (
                "EXPECTED FAILURE: Prediction should fail due to feature count mismatch. "
                "This test documents the bug that needs to be fixed in the GREEN phase."
            )
    
    def test_feature_names_consistency(self):
        """
        Test that training and prediction use the exact same feature names and order.
        """
        # This test will be implemented after the main fix
        pytest.skip("Feature names consistency test - implement after fixing count mismatch")
    
    def test_categorical_features_included_in_prediction(self):
        """
        Test that categorical features (dataset_uuid, rule_code) are included in predictions.
        
        The current bug is that prediction hardcodes only numeric features.
        """
        # Create test data with categorical features
        data = pd.DataFrame({
            'business_date': ['2024-01-01'] * 5,
            'dataset_uuid': ['uuid1', 'uuid2'] * 2 + ['uuid1'],
            'rule_code': [1, 2, 1, 2, 1],
            'source': ['src1'] * 5,
            'tenant_id': ['tenant1'] * 5, 
            'dataset_name': ['dataset1'] * 5,
            'results': ['{"status": "PASS"}'] * 5,
            'level_of_execution': ['row'] * 5,
            'attribute_name': ['attr1'] * 5
        })
        
        rule_metadata = {
            1: {"category": "completeness", "description": "Test rule 1"},
            2: {"category": "accuracy", "description": "Test rule 2"}
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_file = os.path.join(tmp_dir, 'test_data.csv')
            data.to_csv(data_file, index=False)
            model_path = os.path.join(tmp_dir, 'test_model.pkl')
            
            # Train model
            pipeline = MLPipeline()
            pipeline.train_model(data_file, rule_metadata, model_path)
            
            # Create predictor
            predictor = Predictor(model_path, data)
            
            # Verify that predictor can handle different categorical values
            predictions = []
            for uuid in ['uuid1', 'uuid2']:
                for rule in [1, 2]:
                    try:
                        pred = predictor.predict(
                            dataset_uuid=uuid,
                            rule_code=rule,
                            business_date='2024-01-02'
                        )
                        predictions.append((uuid, rule, pred, True))
                    except Exception as e:
                        predictions.append((uuid, rule, str(e), False))
            
            # All predictions should succeed if categorical features are properly handled
            failed_predictions = [p for p in predictions if not p[3]]
            assert len(failed_predictions) == 0, (
                f"Predictions failed for categorical combinations: {failed_predictions}"
            )
            
            # Verify predictions are reasonable (0-100 range)
            successful_predictions = [p[2] for p in predictions if p[3]]
            assert all(0 <= pred <= 100 for pred in successful_predictions), (
                f"Predictions outside valid range: {successful_predictions}"
            )