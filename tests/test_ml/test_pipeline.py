"""
Tests for ML Pipeline Orchestrator.
This module tests the end-to-end training pipeline coordination.
"""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, date

from src.data_quality_summarizer.ml.pipeline import MLPipeline


class TestMLPipeline:
    """Test cases for the ML Pipeline orchestrator."""

    def test_pipeline_initialization(self):
        """Test that pipeline initializes with correct configuration."""
        pipeline = MLPipeline()
        
        assert pipeline is not None
        assert hasattr(pipeline, 'config')
        assert hasattr(pipeline, 'logger')
        assert hasattr(pipeline, 'model_trainer')
        assert hasattr(pipeline, 'evaluator')
        assert hasattr(pipeline, 'progress_callback')

    def test_pipeline_initialization_with_config(self):
        """Test pipeline initialization with custom configuration."""
        config = {
            'chunk_size': 10000,
            'test_size': 0.3,
            'random_state': 42
        }
        
        pipeline = MLPipeline(config=config)
        
        assert pipeline.config['chunk_size'] == 10000
        assert pipeline.config['test_size'] == 0.3
        assert pipeline.config['random_state'] == 42

    def test_train_model_full_pipeline(self):
        """Test complete model training pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test CSV file
            test_csv = Path(temp_dir) / "test_data.csv"
            test_data = pd.DataFrame({
                'source': ['test'] * 10,
                'tenant_id': ['tenant1'] * 10,
                'dataset_uuid': ['uuid1'] * 10,
                'dataset_name': ['dataset1'] * 10,
                'rule_code': ['R001'] * 10,
                'business_date': ['2024-01-01'] * 5 + ['2024-01-02'] * 5,
                'results': ['{"status": "Pass"}'] * 7 + ['{"status": "Fail"}'] * 3,
                'dataset_record_count': [1000] * 10,
                'filtered_record_count': [900] * 10,
                'level_of_execution': ['dataset'] * 10,
                'attribute_name': [None] * 10
            })
            test_data.to_csv(test_csv, index=False)
            
            # Create rule metadata
            rule_metadata = {'R001': Mock(rule_name='Test Rule')}
            
            # Test model path
            model_path = Path(temp_dir) / "test_model.pkl"
            
            pipeline = MLPipeline()
            
            # This should execute the full training pipeline
            result = pipeline.train_model(
                csv_file=str(test_csv),
                rule_metadata=rule_metadata,
                output_model_path=str(model_path)
            )
            
            assert result['success'] is True
            assert 'training_time' in result
            assert 'model_path' in result
            assert 'evaluation_metrics' in result
            assert model_path.exists()

    def test_train_model_with_invalid_csv(self):
        """Test training with invalid CSV file."""
        pipeline = MLPipeline()
        
        result = pipeline.train_model(
            csv_file="nonexistent.csv",
            rule_metadata={},
            output_model_path="model.pkl"
        )
        
        assert result['success'] is False
        assert 'error' in result

    def test_train_model_with_insufficient_data(self):
        """Test training with insufficient data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal test CSV file
            test_csv = Path(temp_dir) / "minimal_data.csv"
            test_data = pd.DataFrame({
                'source': ['test'],
                'tenant_id': ['tenant1'],
                'dataset_uuid': ['uuid1'],
                'dataset_name': ['dataset1'],
                'rule_code': ['R001'],
                'business_date': ['2024-01-01'],
                'results': ['{"status": "Pass"}'],
                'dataset_record_count': [1000],
                'filtered_record_count': [900],
                'level_of_execution': ['dataset'],
                'attribute_name': [None]
            })
            test_data.to_csv(test_csv, index=False)
            
            model_path = Path(temp_dir) / "test_model.pkl"
            pipeline = MLPipeline()
            
            result = pipeline.train_model(
                csv_file=str(test_csv),
                rule_metadata={'R001': Mock(rule_name='Test Rule')},
                output_model_path=str(model_path)
            )
            
            assert result['success'] is False
            assert 'insufficient data' in result['error'].lower()

    def test_load_model_configuration(self):
        """Test loading model configuration from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.json"
            config_data = {
                'chunk_size': 15000,
                'test_size': 0.25,
                'model_params': {'n_estimators': 200}
            }
            
            import json
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
            
            pipeline = MLPipeline()
            loaded_config = pipeline.load_config(str(config_file))
            
            assert loaded_config['chunk_size'] == 15000
            assert loaded_config['test_size'] == 0.25
            assert loaded_config['model_params']['n_estimators'] == 200

    def test_save_model_configuration(self):
        """Test saving model configuration to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.json"
            config_data = {
                'chunk_size': 20000,
                'test_size': 0.2,
                'model_params': {'max_depth': 10}
            }
            
            pipeline = MLPipeline(config=config_data)
            pipeline.save_config(str(config_file))
            
            assert config_file.exists()
            
            import json
            with open(config_file, 'r') as f:
                saved_config = json.load(f)
            
            assert saved_config['chunk_size'] == 20000
            assert saved_config['test_size'] == 0.2
            assert saved_config['model_params']['max_depth'] == 10

    def test_pipeline_progress_tracking(self):
        """Test that pipeline tracks progress correctly."""
        pipeline = MLPipeline()
        
        # Mock progress callback
        progress_callback = Mock()
        pipeline.set_progress_callback(progress_callback)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_csv = Path(temp_dir) / "test_data.csv"
            test_data = pd.DataFrame({
                'source': ['test'] * 20,
                'tenant_id': ['tenant1'] * 20,
                'dataset_uuid': ['uuid1'] * 10 + ['uuid2'] * 10,
                'dataset_name': ['dataset1'] * 10 + ['dataset2'] * 10,
                'rule_code': ['R001'] * 20,
                'business_date': ['2024-01-01'] * 10 + ['2024-01-02'] * 10,
                'results': ['{"status": "Pass"}'] * 14 + ['{"status": "Fail"}'] * 6,
                'dataset_record_count': [1000] * 20,
                'filtered_record_count': [900] * 20,
                'level_of_execution': ['dataset'] * 20,
                'attribute_name': [None] * 20
            })
            test_data.to_csv(test_csv, index=False)
            
            model_path = Path(temp_dir) / "test_model.pkl"
            rule_metadata = {'R001': Mock(rule_name='Test Rule')}
            
            result = pipeline.train_model(
                csv_file=str(test_csv),
                rule_metadata=rule_metadata,
                output_model_path=str(model_path)
            )
            
            # Verify progress callback was called
            assert progress_callback.called
            
    def test_pipeline_memory_monitoring(self):
        """Test that pipeline monitors memory usage."""
        pipeline = MLPipeline()
        
        assert hasattr(pipeline, 'get_memory_usage')
        memory_usage = pipeline.get_memory_usage()
        assert isinstance(memory_usage, (int, float))
        assert memory_usage >= 0

    def test_pipeline_error_handling(self):
        """Test comprehensive error handling in pipeline."""
        pipeline = MLPipeline()
        
        # Test with None inputs
        result = pipeline.train_model(
            csv_file=None,
            rule_metadata=None,
            output_model_path=None
        )
        
        assert result['success'] is False
        assert 'error' in result
        
    def test_pipeline_cleanup(self):
        """Test that pipeline cleans up resources properly."""
        pipeline = MLPipeline()
        
        # Should have cleanup method
        assert hasattr(pipeline, 'cleanup')
        
        # Cleanup should not raise errors
        pipeline.cleanup()