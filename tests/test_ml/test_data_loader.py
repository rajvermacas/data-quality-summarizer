"""
Tests for data_loader module.

Following TDD approach: Red -> Green -> Refactor
"""
import json
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import os

from src.data_quality_summarizer.ml.data_loader import (
    load_and_validate_csv,
    parse_results_column,
    create_binary_pass_column
)


class TestDataLoader:
    """Test suite for data loading functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data for testing
        self.sample_data = {
            'source': ['system_a', 'system_a', 'system_b'],
            'tenant_id': ['tenant1', 'tenant1', 'tenant2'],
            'dataset_uuid': ['uuid1', 'uuid1', 'uuid2'],
            'dataset_name': ['dataset1', 'dataset1', 'dataset2'],
            'business_date': ['2024-01-01', '2024-01-02', '2024-01-01'],
            'rule_code': ['R001', 'R001', 'R002'],
            'results': [
                '{"status": "Pass", "value": 100}',
                '{"status": "Fail", "value": 50}',
                '{"status": "Pass", "value": 95}'
            ],
            'level_of_execution': ['dataset', 'dataset', 'column'],
            'attribute_name': ['', '', 'col1']
        }
        self.sample_df = pd.DataFrame(self.sample_data)
    
    def test_load_and_validate_csv_success(self):
        """Test successful CSV loading with valid structure."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            result_df = load_and_validate_csv(temp_path)
            
            # Verify data loaded correctly
            assert len(result_df) == 3
            assert 'results' in result_df.columns
            assert 'business_date' in result_df.columns
            assert 'rule_code' in result_df.columns
            
        finally:
            os.unlink(temp_path)
    
    def test_load_and_validate_csv_missing_columns(self):
        """Test CSV loading fails with missing required columns."""
        # Create CSV with missing columns
        incomplete_data = {'source': ['system_a'], 'tenant_id': ['tenant1']}
        incomplete_df = pd.DataFrame(incomplete_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            incomplete_df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                load_and_validate_csv(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_parse_results_column_valid_json(self):
        """Test parsing valid JSON results column."""
        json_strings = [
            '{"status": "Pass", "value": 100}',
            '{"status": "Fail", "value": 50}',
            '{"status": "Pass", "value": 95}'
        ]
        
        result = parse_results_column(json_strings)
        
        assert len(result) == 3
        assert result[0]['status'] == 'Pass'
        assert result[1]['status'] == 'Fail'
        assert result[2]['status'] == 'Pass'
    
    def test_parse_results_column_invalid_json(self):
        """Test parsing handles malformed JSON gracefully."""
        json_strings = [
            '{"status": "Pass", "value": 100}',
            'invalid json string',
            '{"status": "Fail", "value": 50}'
        ]
        
        result = parse_results_column(json_strings)
        
        # Should return parsed results for valid JSON, None for invalid
        assert len(result) == 3
        assert result[0]['status'] == 'Pass'
        assert result[1] is None  # Invalid JSON should return None
        assert result[2]['status'] == 'Fail'
    
    def test_create_binary_pass_column_from_status(self):
        """Test binary pass column creation from parsed results."""
        parsed_results = [
            {'status': 'Pass', 'value': 100},
            {'status': 'Fail', 'value': 50},
            {'status': 'Pass', 'value': 95},
            None  # Malformed result
        ]
        
        binary_column = create_binary_pass_column(parsed_results)
        
        assert len(binary_column) == 4
        assert binary_column[0] == 1  # Pass -> 1
        assert binary_column[1] == 0  # Fail -> 0
        assert binary_column[2] == 1  # Pass -> 1
        assert binary_column[3] == 0  # None -> 0 (default to fail)
    
    def test_create_binary_pass_column_case_insensitive(self):
        """Test binary column handles case variations in status."""
        parsed_results = [
            {'status': 'pass', 'value': 100},
            {'status': 'FAIL', 'value': 50},
            {'status': 'Pass', 'value': 95}
        ]
        
        binary_column = create_binary_pass_column(parsed_results)
        
        assert binary_column[0] == 1  # pass -> 1
        assert binary_column[1] == 0  # FAIL -> 0
        assert binary_column[2] == 1  # Pass -> 1