"""
Test module for ML input validation functionality.

This module tests the input validation and sanitization for the prediction service,
ensuring proper parameter validation and error handling.
"""

import pytest
from datetime import datetime, date
from unittest.mock import Mock, patch
import pandas as pd

from src.data_quality_summarizer.ml.validator import InputValidator


class TestInputValidator:
    """Test the InputValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()
    
    def test_validator_initialization(self):
        """Test InputValidator initializes correctly."""
        validator = InputValidator()
        assert validator is not None
        assert hasattr(validator, 'validate_dataset_uuid')
        assert hasattr(validator, 'validate_rule_code')
        assert hasattr(validator, 'validate_business_date')
        assert hasattr(validator, 'validate_all_inputs')
    
    def test_validate_dataset_uuid_valid_string(self):
        """Test dataset_uuid validation with valid string."""
        valid_uuid = "abc123-def456"
        result = self.validator.validate_dataset_uuid(valid_uuid)
        assert result == valid_uuid
    
    def test_validate_dataset_uuid_empty_string(self):
        """Test dataset_uuid validation with empty string."""
        with pytest.raises(ValueError, match="dataset_uuid cannot be empty"):
            self.validator.validate_dataset_uuid("")
    
    def test_validate_dataset_uuid_none_value(self):
        """Test dataset_uuid validation with None value."""
        with pytest.raises(ValueError, match="dataset_uuid must be a string"):
            self.validator.validate_dataset_uuid(None)
    
    def test_validate_dataset_uuid_non_string(self):
        """Test dataset_uuid validation with non-string value."""
        with pytest.raises(ValueError, match="dataset_uuid must be a string"):
            self.validator.validate_dataset_uuid(123)
    
    def test_validate_rule_code_valid_string(self):
        """Test rule_code validation with valid string."""
        valid_code = "R001"
        result = self.validator.validate_rule_code(valid_code)
        assert result == valid_code
    
    def test_validate_rule_code_valid_integer(self):
        """Test rule_code validation with valid integer."""
        valid_code = 1001
        result = self.validator.validate_rule_code(valid_code)
        assert result == "1001"  # Should convert to string
    
    def test_validate_rule_code_empty_string(self):
        """Test rule_code validation with empty string."""
        with pytest.raises(ValueError, match="rule_code cannot be empty"):
            self.validator.validate_rule_code("")
    
    def test_validate_rule_code_none_value(self):
        """Test rule_code validation with None value."""
        with pytest.raises(ValueError, match="rule_code must be a string or integer"):
            self.validator.validate_rule_code(None)
    
    def test_validate_business_date_valid_string(self):
        """Test business_date validation with valid string."""
        valid_date = "2024-01-15"
        result = self.validator.validate_business_date(valid_date)
        assert result == datetime(2024, 1, 15).date()
    
    def test_validate_business_date_valid_datetime(self):
        """Test business_date validation with datetime object."""
        valid_date = datetime(2024, 1, 15)
        result = self.validator.validate_business_date(valid_date)
        assert result == valid_date.date()
    
    def test_validate_business_date_valid_date(self):
        """Test business_date validation with date object."""
        valid_date = date(2024, 1, 15)
        result = self.validator.validate_business_date(valid_date)
        assert result == valid_date
    
    def test_validate_business_date_invalid_format(self):
        """Test business_date validation with invalid format."""
        with pytest.raises(ValueError, match="Invalid date format"):
            self.validator.validate_business_date("01/15/2024")
    
    def test_validate_business_date_invalid_date(self):
        """Test business_date validation with invalid date."""
        with pytest.raises(ValueError, match="Invalid date format"):
            self.validator.validate_business_date("2024-13-45")
    
    def test_validate_business_date_none_value(self):
        """Test business_date validation with None value."""
        with pytest.raises(ValueError, match="business_date cannot be None"):
            self.validator.validate_business_date(None)
    
    def test_validate_all_inputs_valid(self):
        """Test complete input validation with valid inputs."""
        result = self.validator.validate_all_inputs("abc123", "R001", "2024-01-15")
        
        assert result["dataset_uuid"] == "abc123"
        assert result["rule_code"] == "R001"
        assert result["business_date"] == date(2024, 1, 15)
    
    def test_validate_all_inputs_with_integer_rule(self):
        """Test complete input validation with integer rule code."""
        result = self.validator.validate_all_inputs("abc123", 1001, "2024-01-15")
        
        assert result["dataset_uuid"] == "abc123"
        assert result["rule_code"] == "1001"
        assert result["business_date"] == date(2024, 1, 15)
    
    def test_validate_all_inputs_invalid_dataset(self):
        """Test complete input validation with invalid dataset_uuid."""
        with pytest.raises(ValueError, match="dataset_uuid cannot be empty"):
            self.validator.validate_all_inputs("", "R001", "2024-01-15")
    
    def test_validate_all_inputs_invalid_rule(self):
        """Test complete input validation with invalid rule_code."""
        with pytest.raises(ValueError, match="rule_code cannot be empty"):
            self.validator.validate_all_inputs("abc123", "", "2024-01-15")
    
    def test_validate_all_inputs_invalid_date(self):
        """Test complete input validation with invalid business_date."""
        with pytest.raises(ValueError, match="Invalid date format"):
            self.validator.validate_all_inputs("abc123", "R001", "invalid-date")


class TestInputValidatorEdgeCases:
    """Test edge cases for input validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()
    
    def test_validate_dataset_uuid_whitespace_only(self):
        """Test dataset_uuid validation with whitespace only."""
        with pytest.raises(ValueError, match="dataset_uuid cannot be empty"):
            self.validator.validate_dataset_uuid("   ")
    
    def test_validate_rule_code_zero_integer(self):
        """Test rule_code validation with zero integer."""
        result = self.validator.validate_rule_code(0)
        assert result == "0"
    
    def test_validate_rule_code_negative_integer(self):
        """Test rule_code validation with negative integer."""
        result = self.validator.validate_rule_code(-1)
        assert result == "-1"
    
    def test_validate_business_date_future_date(self):
        """Test business_date validation with future date."""
        future_date = "2030-12-31"
        result = self.validator.validate_business_date(future_date)
        assert result == date(2030, 12, 31)
    
    def test_validate_business_date_very_old_date(self):
        """Test business_date validation with very old date."""
        old_date = "1900-01-01"
        result = self.validator.validate_business_date(old_date)
        assert result == date(1900, 1, 1)
    
    def test_validate_all_inputs_strips_whitespace(self):
        """Test that validation strips whitespace from string inputs."""
        result = self.validator.validate_all_inputs("  abc123  ", "  R001  ", "2024-01-15")
        
        assert result["dataset_uuid"] == "abc123"
        assert result["rule_code"] == "R001"
        assert result["business_date"] == date(2024, 1, 15)