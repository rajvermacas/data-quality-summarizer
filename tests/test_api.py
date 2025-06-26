"""Tests for the FastAPI backend."""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json
import io
import pandas as pd
from unittest.mock import patch, MagicMock

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for testing."""
    data = {
        'source': ['db1', 'db2', 'db1'],
        'tenant_id': ['tenant1', 'tenant1', 'tenant2'],
        'dataset_uuid': ['uuid1', 'uuid2', 'uuid1'],
        'dataset_name': ['Dataset 1', 'Dataset 2', 'Dataset 1'],
        'business_date': ['2024-01-01', '2024-01-01', '2024-01-02'],
        'rule_code': ['R001', 'R002', 'R001'],
        'results': ['{"status": "pass"}', '{"status": "fail"}', '{"status": "pass"}'],
        'level_of_execution': ['row', 'table', 'row'],
        'attribute_name': ['attr1', 'attr2', 'attr1']
    }
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()


@pytest.fixture
def sample_rules():
    """Create sample rule metadata."""
    return {
        "R001": {
            "rule_category": "Completeness",
            "rule_name": "Not Null Check",
            "rule_description": "Checks if field is not null"
        },
        "R002": {
            "rule_category": "Accuracy",
            "rule_name": "Range Check",
            "rule_description": "Checks if value is within expected range"
        }
    }


class TestFileUploadEndpoint:
    """Test the file upload endpoint."""
    
    def test_upload_csv_success(self, client, sample_csv_file, sample_rules):
        """Test successful CSV file upload."""
        files = {
            'file': ('test.csv', sample_csv_file, 'text/csv'),
            'rules': ('rules.json', json.dumps(sample_rules), 'application/json')
        }
        
        with patch('src.api.main.process_data_quality') as mock_process:
            mock_process.return_value = {
                'processing_id': 'test-123',
                'status': 'processing',
                'message': 'File uploaded successfully'
            }
            
            response = client.post('/api/upload', files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data['processing_id'] == 'test-123'
            assert data['status'] == 'processing'
    
    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type."""
        files = {
            'file': ('test.txt', 'invalid content', 'text/plain'),
            'rules': ('rules.json', '{}', 'application/json')
        }
        
        response = client.post('/api/upload', files=files)
        
        assert response.status_code == 400
        assert 'Only CSV files are allowed' in response.json()['detail']
    
    def test_upload_missing_file(self, client):
        """Test upload with missing file."""
        response = client.post('/api/upload')
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_upload_large_file(self, client):
        """Test upload with file exceeding size limit."""
        # Create a large CSV content (over 100MB limit)
        large_content = 'col1,col2\n' + 'value1,value2\n' * 10_000_000
        
        files = {
            'file': ('large.csv', large_content, 'text/csv'),
            'rules': ('rules.json', '{}', 'application/json')
        }
        
        response = client.post('/api/upload', files=files)
        
        assert response.status_code == 413
        assert 'File too large' in response.json()['detail']


class TestProcessingStatusEndpoint:
    """Test the processing status endpoint."""
    
    def test_get_processing_status_success(self, client):
        """Test getting processing status."""
        processing_id = 'test-123'
        
        with patch('src.api.main.get_processing_status') as mock_status:
            mock_status.return_value = {
                'processing_id': processing_id,
                'status': 'completed',
                'progress': 100,
                'message': 'Processing completed successfully'
            }
            
            response = client.get(f'/api/processing/{processing_id}')
            
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'completed'
            assert data['progress'] == 100
    
    def test_get_processing_status_not_found(self, client):
        """Test getting status for non-existent processing."""
        response = client.get('/api/processing/non-existent')
        
        assert response.status_code == 404
        assert 'Processing not found' in response.json()['detail']


class TestResultsEndpoint:
    """Test the results endpoint."""
    
    def test_get_results_success(self, client):
        """Test getting processing results."""
        processing_id = 'test-123'
        
        with patch('src.api.main.get_processing_results') as mock_results:
            mock_results.return_value = {
                'summary': {
                    'total_checks': 1000,
                    'passed': 950,
                    'failed': 50,
                    'pass_rate': 0.95
                },
                'by_rule': [
                    {'rule_code': 'R001', 'pass_rate': 0.98},
                    {'rule_code': 'R002', 'pass_rate': 0.92}
                ],
                'trends': {
                    '1_month': {'pass_rate': 0.94},
                    '3_month': {'pass_rate': 0.93},
                    '12_month': {'pass_rate': 0.95}
                }
            }
            
            response = client.get(f'/api/results/{processing_id}')
            
            assert response.status_code == 200
            data = response.json()
            assert data['summary']['total_checks'] == 1000
            assert len(data['by_rule']) == 2
    
    def test_get_results_not_found(self, client):
        """Test getting results for non-existent processing."""
        response = client.get('/api/results/non-existent')
        
        assert response.status_code == 404


class TestExportEndpoint:
    """Test the export endpoint."""
    
    def test_export_csv_success(self, client):
        """Test exporting results as CSV."""
        processing_id = 'test-123'
        
        with patch('src.api.main.export_results_csv') as mock_export:
            mock_export.return_value = 'col1,col2\nval1,val2\n'
            
            response = client.get(f'/api/export/{processing_id}?format=csv')
            
            assert response.status_code == 200
            assert response.headers['content-type'] == 'text/csv; charset=utf-8'
            assert 'attachment; filename=' in response.headers['content-disposition']
    
    def test_export_invalid_format(self, client):
        """Test export with invalid format."""
        response = client.get('/api/export/test-123?format=invalid')
        
        assert response.status_code == 400
        assert 'Invalid export format' in response.json()['detail']