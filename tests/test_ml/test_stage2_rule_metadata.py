"""
Tests for Stage 2: Rule Metadata Format Standardization.

Tests the rule code format conversion functionality that handles
both string ('R001') and integer (1) formats seamlessly.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock

from src.data_quality_summarizer.rules import (
    load_rule_metadata, 
    validate_and_convert_rule_code,
    RuleMetadata
)


class TestRuleCodeConversion:
    """Test rule code format conversion functionality."""
    
    def test_convert_rule_code_r_prefix(self):
        """Test converting 'R001' format to integer 1."""
        # This test should fail initially - function doesn't exist yet
        result = validate_and_convert_rule_code('R001')
        assert result == 1
    
    def test_convert_rule_code_r_prefix_multiple_digits(self):
        """Test converting 'R123' format to integer 123."""
        result = validate_and_convert_rule_code('R123')
        assert result == 123
    
    def test_convert_rule_code_direct_string(self):
        """Test converting '001' format to integer 1."""
        result = validate_and_convert_rule_code('001')
        assert result == 1
    
    def test_convert_rule_code_integer_passthrough(self):
        """Test that integer rule codes are returned as-is."""
        result = validate_and_convert_rule_code(1)
        assert result == 1
    
    def test_convert_rule_code_invalid_format(self):
        """Test handling invalid rule codes gracefully."""
        result = validate_and_convert_rule_code('INVALID')
        assert result is None
    
    def test_convert_rule_code_invalid_type(self):
        """Test handling unexpected types gracefully."""
        result = validate_and_convert_rule_code(None)
        assert result is None
    
    def test_convert_rule_code_empty_string(self):
        """Test handling empty string gracefully."""
        result = validate_and_convert_rule_code('')
        assert result is None


class TestRuleMetadataLoading:
    """Test rule metadata loading with mixed formats."""
    
    def test_load_metadata_with_string_rule_codes(self):
        """Test loading rule metadata with 'R001' format keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_file = Path(temp_dir) / "test_rules.json"
            
            # Create test metadata with string rule codes
            test_metadata = {
                "R001": {
                    "rule_name": "Completeness Check",
                    "rule_type": "DATASET",
                    "dimension": "Completeness",
                    "rule_description": "Check data completeness",
                    "category": "C1"
                },
                "R002": {
                    "rule_name": "Format Validation",
                    "rule_type": "ATTRIBUTE",
                    "dimension": "Validity",
                    "rule_description": "Validate data format",
                    "category": "C2"
                }
            }
            
            with open(rules_file, 'w') as f:
                json.dump(test_metadata, f)
            
            # This should succeed after implementing conversion
            result = load_rule_metadata(str(rules_file))
            
            # Should have converted string keys to integer keys
            assert 1 in result
            assert 2 in result
            assert result[1].rule_name == "Completeness Check"
            assert result[2].rule_name == "Format Validation"
    
    def test_load_metadata_mixed_format_rule_codes(self):
        """Test loading rule metadata with mixed string/int formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_file = Path(temp_dir) / "test_rules.json"
            
            # Create test metadata with mixed formats
            test_metadata = {
                "1": {  # String number
                    "rule_name": "Rule 1",
                    "rule_type": "DATASET",
                    "dimension": "Completeness",
                    "rule_description": "Description 1",
                    "category": "C1"
                },
                "R002": {  # R-prefix format
                    "rule_name": "Rule 2",
                    "rule_type": "ATTRIBUTE",
                    "dimension": "Validity", 
                    "rule_description": "Description 2",
                    "category": "C2"
                }
            }
            
            with open(rules_file, 'w') as f:
                json.dump(test_metadata, f)
            
            result = load_rule_metadata(str(rules_file))
            
            # Both should be converted to integer keys
            assert 1 in result
            assert 2 in result
    
    def test_load_metadata_invalid_rule_codes_skipped(self):
        """Test that invalid rule codes are skipped with warnings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_file = Path(temp_dir) / "test_rules.json"
            
            test_metadata = {
                "R001": {
                    "rule_name": "Valid Rule",
                    "rule_type": "DATASET",
                    "dimension": "Completeness",
                    "rule_description": "Valid description",
                    "category": "C1"
                },
                "INVALID": {
                    "rule_name": "Invalid Rule",
                    "rule_type": "DATASET",
                    "dimension": "Completeness",
                    "rule_description": "Invalid rule code",
                    "category": "C1"
                }
            }
            
            with open(rules_file, 'w') as f:
                json.dump(test_metadata, f)
            
            with patch('src.data_quality_summarizer.rules.logger') as mock_logger:
                result = load_rule_metadata(str(rules_file))
                
                # Should have only the valid rule
                assert len(result) == 1
                assert 1 in result
                
                # Should have logged warning for invalid rule code
                mock_logger.warning.assert_called_with(
                    "Invalid rule code format (cannot convert to int): INVALID"
                )


class TestDataLoaderRuleCodeNormalization:
    """Test rule code normalization in data loader."""
    
    def test_normalize_rule_codes_function_exists(self):
        """Test that normalize_rule_codes function exists."""
        from src.data_quality_summarizer.ml.data_loader import normalize_rule_codes
        import pandas as pd
        
        # Create test dataframe with string rule codes
        df = pd.DataFrame({
            'rule_code': ['R001', 'R002', 'R003'],
            'other_col': [1, 2, 3]
        })
        
        # This should convert rule codes to integers
        result = normalize_rule_codes(df)
        
        assert result['rule_code'].dtype == 'int64'
        assert list(result['rule_code']) == [1, 2, 3]
    
    def test_normalize_rule_codes_mixed_formats(self):
        """Test normalization handles mixed rule code formats."""
        from src.data_quality_summarizer.ml.data_loader import normalize_rule_codes
        import pandas as pd
        
        df = pd.DataFrame({
            'rule_code': ['R001', '2', 3, 'R004'],
            'other_col': [1, 2, 3, 4]
        })
        
        result = normalize_rule_codes(df)
        
        assert result['rule_code'].dtype == 'int64'
        assert list(result['rule_code']) == [1, 2, 3, 4]
    
    def test_normalize_rule_codes_invalid_codes_removed(self):
        """Test that invalid rule codes are removed from dataframe."""
        from src.data_quality_summarizer.ml.data_loader import normalize_rule_codes
        import pandas as pd
        
        df = pd.DataFrame({
            'rule_code': ['R001', 'INVALID', 'R003'],
            'other_col': [1, 2, 3]
        })
        
        result = normalize_rule_codes(df)
        
        # Should have only 2 rows (invalid one removed)
        assert len(result) == 2
        assert list(result['rule_code']) == [1, 3]
        assert list(result['other_col']) == [1, 3]