"""
Tests for ML Batch Predictor.
This module tests batch prediction functionality for multiple requests.
"""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, date
import pickle

from src.data_quality_summarizer.ml.batch_predictor import BatchPredictor


class TestBatchPredictor:
    """Test cases for the Batch Predictor."""
    
    def _create_mock_predictor_with_deterministic_output(self):
        """Create a mock predictor that returns deterministic predictions."""
        mock_predictor = Mock()
        
        def predict_function(dataset_uuid, rule_code, business_date):
            # Create deterministic output based on inputs
            # This ensures test stability while avoiding real predictor complexity
            base_score = 87.5
            uuid_mod = hash(str(dataset_uuid)) % 10
            rule_mod = hash(str(rule_code)) % 5
            date_mod = hash(str(business_date)) % 3
            
            return base_score + uuid_mod - rule_mod + date_mod
        
        mock_predictor.predict = predict_function
        return mock_predictor

    def test_batch_predictor_initialization(self):
        """Test that batch predictor initializes correctly."""
        batch_predictor = BatchPredictor()
        
        assert batch_predictor is not None
        assert hasattr(batch_predictor, 'predictor')
        assert hasattr(batch_predictor, 'progress_callback')

    def test_batch_predictor_with_model_path(self, real_pickled_model_file):
        """Test batch predictor initialization with model path."""
        # Use real pickled model file from fixture
        batch_predictor = BatchPredictor(model_path=real_pickled_model_file)
        
        assert batch_predictor.model_path == real_pickled_model_file
        assert Path(batch_predictor.model_path).exists()
        
        # Verify predictor is properly initialized
        assert batch_predictor.predictor is None  # Initially None until historical data provided

    def test_process_batch_csv_predictions(self):
        """Test processing batch predictions from CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test prediction requests CSV
            predictions_csv = Path(temp_dir) / "predictions.csv"
            prediction_data = pd.DataFrame({
                'dataset_uuid': ['uuid1', 'uuid2', 'uuid3'],
                'rule_code': ['R001', 'R002', 'R001'],
                'business_date': ['2024-01-15', '2024-01-16', '2024-01-17']
            })
            prediction_data.to_csv(predictions_csv, index=False)
            
            # Create output path
            results_csv = Path(temp_dir) / "results.csv"
            
            # Create mock historical data
            historical_data = pd.DataFrame({
                'dataset_uuid': ['uuid1', 'uuid2'] * 5,
                'rule_code': ['R001', 'R002'] * 5,
                'business_date': pd.date_range('2024-01-01', periods=10),
                'pass_percentage': [85.0, 90.0] * 5
            })
            
            batch_predictor = BatchPredictor()
            
            with patch.object(batch_predictor, '_load_historical_data', return_value=historical_data):
                # Use mock predictor with deterministic output instead of None predictor
                batch_predictor.predictor = self._create_mock_predictor_with_deterministic_output()
                result = batch_predictor.process_batch_csv(
                    input_csv=str(predictions_csv),
                    output_csv=str(results_csv),
                    historical_data_csv="dummy.csv"
                )
            
            assert result['success'] is True
            assert result['predictions_processed'] == 3
            assert results_csv.exists()
            
            # Verify output CSV format
            results_df = pd.read_csv(results_csv)
            assert len(results_df) == 3
            assert 'dataset_uuid' in results_df.columns
            assert 'rule_code' in results_df.columns
            assert 'business_date' in results_df.columns
            assert 'predicted_pass_percentage' in results_df.columns

    def test_process_batch_list_predictions(self):
        """Test processing batch predictions from list of requests."""
        prediction_requests = [
            {'dataset_uuid': 'uuid1', 'rule_code': 'R001', 'business_date': '2024-01-15'},
            {'dataset_uuid': 'uuid2', 'rule_code': 'R002', 'business_date': '2024-01-16'},
            {'dataset_uuid': 'uuid3', 'rule_code': 'R001', 'business_date': '2024-01-17'}
        ]
        
        # Create mock historical data
        historical_data = pd.DataFrame({
            'dataset_uuid': ['uuid1', 'uuid2', 'uuid3'] * 3,
            'rule_code': ['R001', 'R002', 'R001'] * 3,
            'business_date': pd.date_range('2024-01-01', periods=9),
            'pass_percentage': [85.0, 90.0, 88.0] * 3
        })
        
        batch_predictor = BatchPredictor()
        
        # Use mock predictor with deterministic output instead of None predictor
        batch_predictor.predictor = self._create_mock_predictor_with_deterministic_output()
        results = batch_predictor.process_batch_list(
            prediction_requests=prediction_requests,
            historical_data=historical_data
        )
        
        assert len(results) == 3
        for result in results:
            assert 'dataset_uuid' in result
            assert 'rule_code' in result
            assert 'business_date' in result
            assert 'predicted_pass_percentage' in result
            # Prediction should be deterministic based on our mock function
            assert isinstance(result['predicted_pass_percentage'], (int, float))

    def test_batch_predictor_with_progress_callback(self):
        """Test batch predictor with progress tracking."""
        prediction_requests = [
            {'dataset_uuid': f'uuid{i}', 'rule_code': 'R001', 'business_date': '2024-01-15'}
            for i in range(5)
        ]
        
        historical_data = pd.DataFrame({
            'dataset_uuid': [f'uuid{i}' for i in range(5)] * 2,
            'rule_code': ['R001'] * 10,
            'business_date': pd.date_range('2024-01-01', periods=10),
            'pass_percentage': [85.0] * 10
        })
        
        progress_callback = Mock()
        batch_predictor = BatchPredictor()
        batch_predictor.set_progress_callback(progress_callback)
        
        # Use mock predictor with deterministic output instead of None predictor
        batch_predictor.predictor = self._create_mock_predictor_with_deterministic_output()
        results = batch_predictor.process_batch_list(
            prediction_requests=prediction_requests,
            historical_data=historical_data
        )
        
        assert len(results) == 5
        assert progress_callback.called

    def test_batch_predictor_invalid_csv_format(self):
        """Test batch predictor with invalid CSV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid CSV (missing required columns)
            invalid_csv = Path(temp_dir) / "invalid.csv"
            invalid_data = pd.DataFrame({
                'dataset_uuid': ['uuid1'],
                'rule_code': ['R001']
                # Missing business_date column
            })
            invalid_data.to_csv(invalid_csv, index=False)
            
            results_csv = Path(temp_dir) / "results.csv"
            
            batch_predictor = BatchPredictor()
            
            result = batch_predictor.process_batch_csv(
                input_csv=str(invalid_csv),
                output_csv=str(results_csv),
                historical_data_csv="dummy.csv"
            )
            
            assert result['success'] is False
            assert 'error' in result

    def test_batch_predictor_nonexistent_input_file(self):
        """Test batch predictor with nonexistent input file."""
        batch_predictor = BatchPredictor()
        
        result = batch_predictor.process_batch_csv(
            input_csv="nonexistent.csv",
            output_csv="results.csv",
            historical_data_csv="dummy.csv"
        )
        
        assert result['success'] is False
        assert 'error' in result

    def test_batch_predictor_empty_requests(self):
        """Test batch predictor with empty prediction requests."""
        batch_predictor = BatchPredictor()
        
        results = batch_predictor.process_batch_list(
            prediction_requests=[],
            historical_data=pd.DataFrame()
        )
        
        assert results == []

    def test_batch_predictor_prediction_error_handling(self):
        """Test batch predictor handles individual prediction errors."""
        prediction_requests = [
            {'dataset_uuid': 'uuid1', 'rule_code': 'R001', 'business_date': '2024-01-15'},
            {'dataset_uuid': 'uuid2', 'rule_code': 'R002', 'business_date': '2024-01-16'},
        ]
        
        historical_data = pd.DataFrame({
            'dataset_uuid': ['uuid1', 'uuid2'] * 2,
            'rule_code': ['R001', 'R002'] * 2,
            'business_date': pd.date_range('2024-01-01', periods=4),
            'pass_percentage': [85.0, 90.0] * 2
        })
        
        batch_predictor = BatchPredictor()
        
        # Create mock predictor that raises error on second prediction
        mock_predictor = Mock()
        def mock_predict(*args, **kwargs):
            if 'uuid2' in str(args) or 'uuid2' in str(kwargs):
                raise ValueError("Prediction error")
            return 87.5
        mock_predictor.predict = mock_predict
        
        batch_predictor.predictor = mock_predictor
        results = batch_predictor.process_batch_list(
            prediction_requests=prediction_requests,
            historical_data=historical_data
        )
        
        assert len(results) == 2
        assert results[0]['predicted_pass_percentage'] == 87.5
        assert 'error' in results[1]

    def test_batch_predictor_large_batch_performance(self):
        """Test batch predictor performance with large batches."""
        # Create large batch of prediction requests
        prediction_requests = [
            {'dataset_uuid': f'uuid{i}', 'rule_code': f'R00{i%3+1}', 'business_date': '2024-01-15'}
            for i in range(100)
        ]
        
        historical_data = pd.DataFrame({
            'dataset_uuid': [f'uuid{i}' for i in range(100)] * 2,
            'rule_code': [f'R00{i%3+1}' for i in range(100)] * 2,
            'business_date': pd.date_range('2024-01-01', periods=200),
            'pass_percentage': [85.0] * 200
        })
        
        batch_predictor = BatchPredictor()
        
        import time
        start_time = time.time()
        
        # Use mock predictor with deterministic output instead of None predictor
        batch_predictor.predictor = self._create_mock_predictor_with_deterministic_output()
        results = batch_predictor.process_batch_list(
            prediction_requests=prediction_requests,
            historical_data=historical_data
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert len(results) == 100
        assert processing_time < 60  # Should complete within 1 minute

    def test_batch_predictor_output_csv_format_validation(self):
        """Test that output CSV has the correct format and columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictions_csv = Path(temp_dir) / "predictions.csv"
            prediction_data = pd.DataFrame({
                'dataset_uuid': ['uuid1', 'uuid2'],
                'rule_code': ['R001', 'R002'],
                'business_date': ['2024-01-15', '2024-01-16']
            })
            prediction_data.to_csv(predictions_csv, index=False)
            
            results_csv = Path(temp_dir) / "results.csv"
            historical_data = pd.DataFrame({
                'dataset_uuid': ['uuid1', 'uuid2'] * 2,
                'rule_code': ['R001', 'R002'] * 2,
                'business_date': pd.date_range('2024-01-01', periods=4),
                'pass_percentage': [85.0, 90.0] * 2
            })
            
            batch_predictor = BatchPredictor()
            
            with patch.object(batch_predictor, '_load_historical_data', return_value=historical_data):
                # Use mock predictor with deterministic output instead of None predictor
                batch_predictor.predictor = self._create_mock_predictor_with_deterministic_output()
                result = batch_predictor.process_batch_csv(
                    input_csv=str(predictions_csv),
                    output_csv=str(results_csv),
                    historical_data_csv="dummy.csv"
                )
            
            assert result['success'] is True
            
            # Validate output CSV format
            results_df = pd.read_csv(results_csv)
            
            # Check required columns
            required_columns = ['dataset_uuid', 'rule_code', 'business_date', 'predicted_pass_percentage']
            for col in required_columns:
                assert col in results_df.columns
            
            # Check data types and ranges
            assert all(isinstance(val, str) for val in results_df['dataset_uuid'])
            assert all(isinstance(val, str) for val in results_df['rule_code'])
            assert all(0 <= val <= 100 for val in results_df['predicted_pass_percentage'])

    def test_batch_predictor_memory_efficiency(self):
        """Test that batch predictor is memory efficient for large datasets."""
        batch_predictor = BatchPredictor()
        
        # Should have method to get memory usage
        assert hasattr(batch_predictor, 'get_memory_usage')
        
        memory_usage = batch_predictor.get_memory_usage()
        assert isinstance(memory_usage, (int, float))
        assert memory_usage >= 0