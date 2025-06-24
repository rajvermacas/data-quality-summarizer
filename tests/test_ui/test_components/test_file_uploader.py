"""
Tests for file upload component.

Tests Stage 1 User Story 2: File upload interface with validation.
"""

import pytest
from unittest.mock import Mock, patch
import tempfile
import json
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))


class TestFileUploader:
    """Test file upload component functionality."""
    
    def test_file_uploader_module_can_be_created(self):
        """Test that file uploader module can be imported when created."""
        # This will fail initially - following TDD
        try:
            from data_quality_summarizer.ui.components import file_uploader
            assert hasattr(file_uploader, 'validate_csv_file')
            assert hasattr(file_uploader, 'validate_json_file')
            assert hasattr(file_uploader, 'create_file_uploader')
        except ImportError:
            pytest.fail("File uploader component should be importable")
    
    def test_csv_validation_function_exists(self):
        """Test that CSV validation function exists."""
        from data_quality_summarizer.ui.components import file_uploader
        
        # Required function for CSV validation
        assert callable(file_uploader.validate_csv_file)
    
    def test_json_validation_function_exists(self):
        """Test that JSON validation function exists."""
        from data_quality_summarizer.ui.components import file_uploader
        
        # Required function for JSON validation  
        assert callable(file_uploader.validate_json_file)
    
    def test_file_uploader_creation_function_exists(self):
        """Test that file uploader creation function exists."""
        from data_quality_summarizer.ui.components import file_uploader
        
        # Required function for creating uploader UI
        assert callable(file_uploader.create_file_uploader)


class TestFileValidation:
    """Test file validation logic."""
    
    def test_csv_validation_with_valid_file(self):
        """Test CSV validation accepts valid data file."""
        from data_quality_summarizer.ui.components.file_uploader import validate_csv_file
        
        # Create a valid test CSV
        valid_csv_content = """source,tenant_id,dataset_uuid,dataset_name,business_date,rule_code,results,level_of_execution,attribute_name
test_source,tenant1,uuid123,dataset1,2024-01-01,R001,"{""passed"": true}",dataset,attr1
test_source,tenant1,uuid123,dataset1,2024-01-02,R002,"{""passed"": false}",dataset,attr2"""
        
        # Test validation
        result = validate_csv_file(valid_csv_content.encode())
        
        assert result["is_valid"] is True
        assert "error" not in result or result["error"] is None
        assert "row_count" in result
        assert result["row_count"] >= 2
    
    def test_csv_validation_with_missing_columns(self):
        """Test CSV validation rejects files with missing required columns."""
        from data_quality_summarizer.ui.components.file_uploader import validate_csv_file
        
        # CSV missing required columns
        invalid_csv_content = """source,tenant_id,dataset_name
test_source,tenant1,dataset1"""
        
        result = validate_csv_file(invalid_csv_content.encode())
        
        assert result["is_valid"] is False
        assert "error" in result
        assert "missing required columns" in result["error"].lower()
    
    def test_json_validation_with_valid_rules(self):
        """Test JSON validation accepts valid rule metadata."""
        from data_quality_summarizer.ui.components.file_uploader import validate_json_file
        
        # Valid rule metadata
        valid_rules = {
            "R001": {
                "name": "Test Rule 1",
                "description": "Test description",
                "category": "data_quality"
            },
            "R002": {
                "name": "Test Rule 2", 
                "description": "Another test",
                "category": "completeness"
            }
        }
        
        result = validate_json_file(json.dumps(valid_rules).encode())
        
        assert result["is_valid"] is True
        assert "error" not in result or result["error"] is None
        assert "rule_count" in result
        assert result["rule_count"] == 2
    
    def test_json_validation_with_invalid_format(self):
        """Test JSON validation rejects malformed JSON."""
        from data_quality_summarizer.ui.components.file_uploader import validate_json_file
        
        # Invalid JSON
        invalid_json = "{ invalid json content"
        
        result = validate_json_file(invalid_json.encode())
        
        assert result["is_valid"] is False
        assert "error" in result
        assert "json" in result["error"].lower()


class TestFileUploaderUI:
    """Test file uploader UI component creation."""
    
    @patch('data_quality_summarizer.ui.components.file_uploader.st', create=True)
    def test_create_file_uploader_creates_ui_elements(self, mock_st):
        """Test that file uploader creates necessary UI elements."""
        from data_quality_summarizer.ui.components.file_uploader import create_file_uploader
        
        # Mock return values
        mock_st.columns.return_value = [Mock(), Mock()]
        mock_st.file_uploader.return_value = None
        
        # Call function
        uploader_state = create_file_uploader()
        
        # Verify UI elements were created
        assert mock_st.file_uploader.call_count >= 2  # CSV and JSON uploaders
        assert isinstance(uploader_state, dict)
        assert "csv_file" in uploader_state
        assert "json_file" in uploader_state