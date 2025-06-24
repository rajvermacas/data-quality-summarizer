"""
Tests for the UI backend integration module.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd

from src.data_quality_summarizer.ui.backend_integration import app, ProcessingResult


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing."""
    return """source,tenant_id,dataset_uuid,dataset_name,business_date,rule_code,results,level_of_execution,attribute_name
test_source,tenant1,uuid1,test_dataset,2024-01-01,R001,"{""status"":""pass""}",dataset,attr1
test_source,tenant1,uuid1,test_dataset,2024-01-02,R001,"{""status"":""fail""}",dataset,attr1
test_source,tenant1,uuid2,test_dataset2,2024-01-01,R002,"{""status"":""pass""}",dataset,attr2"""


@pytest.fixture
def sample_rules_content():
    """Sample rules JSON content for testing."""
    return {
        "R001": {
            "category": "Completeness",
            "description": "Check for missing values",
            "severity": "HIGH"
        },
        "R002": {
            "category": "Validity",
            "description": "Validate data formats", 
            "severity": "MEDIUM"
        }
    }


@pytest.fixture
def mock_processing_components():
    """Mock the core processing components."""
    with patch('src.data_quality_summarizer.ui.backend_integration.CSVIngestionEngine') as mock_ingestion, \
         patch('src.data_quality_summarizer.ui.backend_integration.RuleMetadataLoader') as mock_rule_loader, \
         patch('src.data_quality_summarizer.ui.backend_integration.StreamingDataAggregator') as mock_aggregator, \
         patch('src.data_quality_summarizer.ui.backend_integration.DataQualitySummarizer') as mock_summarizer:
        
        # Configure mocks
        mock_ingestion_instance = Mock()
        mock_ingestion.return_value = mock_ingestion_instance
        
        mock_rule_loader_instance = Mock()
        mock_rule_loader.return_value = mock_rule_loader_instance
        mock_rule_loader_instance.load_rule_metadata.return_value = {
            "R001": {"category": "Completeness", "description": "Test rule 1"},
            "R002": {"category": "Validity", "description": "Test rule 2"}
        }
        
        mock_aggregator_instance = Mock()
        mock_aggregator.return_value = mock_aggregator_instance
        
        mock_summarizer_instance = Mock()
        mock_summarizer.return_value = mock_summarizer_instance
        mock_summarizer_instance.generate_csv_summary.return_value = [
            {
                "dataset_uuid": "uuid1",
                "dataset_name": "test_dataset",
                "rule_code": "R001",
                "overall_fail_rate": 0.1,
                "latest_business_date": "2024-01-02"
            }
        ]
        mock_summarizer_instance.generate_natural_language_summary.return_value = [
            "Test summary sentence 1",
            "Test summary sentence 2"
        ]
        
        # Mock chunk reading
        sample_chunk = pd.DataFrame({
            'source': ['test_source'],
            'dataset_uuid': ['uuid1'],
            'rule_code': ['R001']
        })
        mock_ingestion_instance.read_csv_chunks.return_value = [sample_chunk]
        
        yield {
            'ingestion': mock_ingestion_instance,
            'rule_loader': mock_rule_loader_instance,
            'aggregator': mock_aggregator_instance,
            'summarizer': mock_summarizer_instance
        }


class TestHealthCheck:
    """Test the health check endpoint."""
    
    def test_health_check(self, client):
        """Test that health check returns success."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "Data Quality Summarizer API" in data["message"]


class TestFileProcessing:
    """Test file processing endpoints."""
    
    def test_process_files_success(self, client, sample_csv_content, sample_rules_content, mock_processing_components):
        """Test successful file processing."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as csv_file:
            csv_file.write(sample_csv_content)
            csv_file_path = csv_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as rules_file:
            json.dump(sample_rules_content, rules_file)
            rules_file_path = rules_file.name
        
        try:
            # Test the endpoint
            with open(csv_file_path, 'rb') as csv_f, open(rules_file_path, 'rb') as rules_f:
                response = client.post(
                    "/api/process",
                    files={
                        "csv_file": ("test.csv", csv_f, "text/csv"),
                        "rules_file": ("rules.json", rules_f, "application/json")
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "summary_data" in data
            assert "nl_summary" in data
            assert "total_rows_processed" in data
            assert "processing_time_seconds" in data
            assert "memory_usage_mb" in data
            assert "unique_datasets" in data
            assert "unique_rules" in data
            assert "time_range" in data
            
            # Verify mocks were called
            mock_processing_components['rule_loader'].load_rule_metadata.assert_called_once()
            mock_processing_components['ingestion'].read_csv_chunks.assert_called_once()
            mock_processing_components['aggregator'].process_chunk.assert_called()
            mock_processing_components['summarizer'].generate_csv_summary.assert_called_once()
            mock_processing_components['summarizer'].generate_natural_language_summary.assert_called_once()
            
        finally:
            # Cleanup
            Path(csv_file_path).unlink(missing_ok=True)
            Path(rules_file_path).unlink(missing_ok=True)
    
    def test_process_files_invalid_csv_extension(self, client, sample_rules_content):
        """Test that invalid CSV file extension is rejected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as csv_file:
            csv_file.write("test content")
            csv_file_path = csv_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as rules_file:
            json.dump(sample_rules_content, rules_file)
            rules_file_path = rules_file.name
        
        try:
            with open(csv_file_path, 'rb') as csv_f, open(rules_file_path, 'rb') as rules_f:
                response = client.post(
                    "/api/process",
                    files={
                        "csv_file": ("test.txt", csv_f, "text/plain"),
                        "rules_file": ("rules.json", rules_f, "application/json")
                    }
                )
            
            assert response.status_code == 400
            assert "CSV file must have .csv extension" in response.json()["detail"]
            
        finally:
            Path(csv_file_path).unlink(missing_ok=True)
            Path(rules_file_path).unlink(missing_ok=True)
    
    def test_process_files_invalid_rules_extension(self, client, sample_csv_content):
        """Test that invalid rules file extension is rejected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as csv_file:
            csv_file.write(sample_csv_content)
            csv_file_path = csv_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as rules_file:
            rules_file.write("test content")
            rules_file_path = rules_file.name
        
        try:
            with open(csv_file_path, 'rb') as csv_f, open(rules_file_path, 'rb') as rules_f:
                response = client.post(
                    "/api/process",
                    files={
                        "csv_file": ("test.csv", csv_f, "text/csv"),
                        "rules_file": ("rules.txt", rules_f, "text/plain")
                    }
                )
            
            assert response.status_code == 400
            assert "Rules file must have .json extension" in response.json()["detail"]
            
        finally:
            Path(csv_file_path).unlink(missing_ok=True)
            Path(rules_file_path).unlink(missing_ok=True)


class TestMLEndpoints:
    """Test ML pipeline endpoints."""
    
    @patch('src.data_quality_summarizer.ui.backend_integration.ModelTrainer')
    @patch('src.data_quality_summarizer.ui.backend_integration.Predictor')
    def test_train_model_success(self, mock_predictor_class, mock_trainer_class, client):
        """Test successful model training."""
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.train_model.return_value = {
            'training_score': 0.85,
            'validation_score': 0.82,
            'test_score': 0.80,
            'feature_importance': {'feature1': 0.6, 'feature2': 0.4}
        }
        
        # Mock predictor
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor
        
        # Sample training data
        sample_data = [
            {
                "dataset_uuid": "uuid1",
                "rule_code": "R001",
                "overall_fail_rate": 0.1
            }
        ]
        
        response = client.post(
            "/api/ml/train",
            data={"csv_data": json.dumps(sample_data)}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_id" in data
        assert data["model_type"] == "lightgbm"
        assert "training_score" in data
        assert "validation_score" in data
        assert "test_score" in data
        assert "feature_importance" in data
        assert "training_time_seconds" in data
        
        # Verify trainer was called
        mock_trainer.train_model.assert_called_once()
    
    def test_train_model_invalid_json(self, client):
        """Test model training with invalid JSON data."""
        response = client.post(
            "/api/ml/train",
            data={"csv_data": "invalid json"}
        )
        
        assert response.status_code == 400
        assert "Invalid JSON data" in response.json()["detail"]
    
    def test_train_model_empty_data(self, client):
        """Test model training with empty data."""
        response = client.post(
            "/api/ml/train",
            data={"csv_data": "[]"}
        )
        
        assert response.status_code == 400
        assert "No data provided for training" in response.json()["detail"]
    
    @patch('src.data_quality_summarizer.ui.backend_integration.current_model')
    def test_predict_no_model(self, mock_current_model, client):
        """Test prediction when no model is loaded."""
        mock_current_model = None
        
        prediction_request = {
            "dataset_uuid": "uuid1",
            "rule_code": "R001",
            "business_date": "2024-01-01"
        }
        
        response = client.post("/api/ml/predict", json=prediction_request)
        
        assert response.status_code == 400
        assert "No model available" in response.json()["detail"]
    
    def test_get_model_info_no_model(self, client):
        """Test model info when no model is loaded."""
        response = client.get("/api/ml/model-info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is False
        assert "No model currently loaded" in data["message"]


class TestUtilityFunctions:
    """Test utility functions and error handling."""
    
    def test_processing_result_model(self):
        """Test ProcessingResult model validation."""
        # Valid data
        valid_data = {
            "summary_data": [{"test": "data"}],
            "nl_summary": ["sentence 1"],
            "total_rows_processed": 100,
            "processing_time_seconds": 1.5,
            "memory_usage_mb": 50.0,
            "unique_datasets": 2,
            "unique_rules": 3,
            "time_range": {"start_date": "2024-01-01", "end_date": "2024-01-31"}
        }
        
        result = ProcessingResult(**valid_data)
        assert result.total_rows_processed == 100
        assert result.processing_time_seconds == 1.5
        assert len(result.summary_data) == 1
        assert len(result.nl_summary) == 1
    
    @patch('src.data_quality_summarizer.ui.backend_integration.run_server')
    def test_server_startup(self, mock_run_server):
        """Test server startup function."""
        from src.data_quality_summarizer.ui.backend_integration import run_server
        
        # Test default parameters
        run_server()
        mock_run_server.assert_called_with()
        
        # Test custom parameters
        run_server(host="0.0.0.0", port=9000, debug=False)
        mock_run_server.assert_called_with(host="0.0.0.0", port=9000, debug=False)