"""Test module for data validation functionality."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.data_quality_summarizer.ml.data_validator import (
    DataValidator,
    ValidationReport,
    DataQualityException
)


class TestDataValidator:
    """Test cases for DataValidator class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data for testing."""
        # Create data with 25 samples per group to meet minimum requirements
        data = []
        for uuid in ['uuid1', 'uuid2']:
            for rule in ['R001', 'R002']:
                for i in range(25):  # 25 samples per group
                    data.append({
                        'pass_percentage': np.random.uniform(0, 100),
                        'dataset_uuid': uuid,
                        'rule_code': rule,
                        'feature1': i + np.random.uniform(-0.1, 0.1),  # Slightly varied to avoid perfect correlation
                        'feature2': (i * 0.1) + np.random.uniform(-0.01, 0.01)  # Slightly varied
                    })
        return pd.DataFrame(data)
    
    @pytest.fixture
    def zero_heavy_data(self):
        """Create data with 95% zero values in target."""
        target_values = [0.0] * 95 + [50.0] * 5
        return pd.DataFrame({
            'pass_percentage': target_values,
            'dataset_uuid': ['uuid1'] * 100,
            'rule_code': ['R001'] * 100,
            'feature1': range(100),
            'feature2': np.random.rand(100)
        })
    
    @pytest.fixture
    def low_variance_data(self):
        """Create data with very low target variance (std < 0.1)."""
        target_values = [50.0, 50.01, 49.99, 50.02, 49.98] * 20
        return pd.DataFrame({
            'pass_percentage': target_values,
            'dataset_uuid': ['uuid1'] * 100,
            'rule_code': ['R001'] * 100,
            'feature1': range(100),
            'feature2': np.random.rand(100)
        })
    
    def test_validate_target_distribution_normal_case(self, sample_data):
        """Test target distribution validation with normal data."""
        validator = DataValidator()
        report = validator.validate_target_distribution(sample_data, 'pass_percentage')
        
        assert isinstance(report, ValidationReport)
        assert report.passed is True
        assert 'target_stats' in report.details
        assert report.details['target_stats']['count'] == 100
        assert report.details['target_stats']['std'] > 0.1  # Should have good variance
        
    def test_validate_target_distribution_zero_heavy_warning(self, zero_heavy_data):
        """Test warning generation when >90% of target values are zero."""
        validator = DataValidator()
        
        with patch('src.data_quality_summarizer.ml.data_validator.logger') as mock_logger:
            report = validator.validate_target_distribution(zero_heavy_data, 'pass_percentage')
            
            # Should still pass but generate warning
            assert report.passed is True
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0]
            assert "High zero percentage in target" in str(call_args)
    
    def test_validate_target_distribution_low_variance_halt(self, low_variance_data):
        """Test training halt when standard deviation < 0.1."""
        validator = DataValidator()
        
        with pytest.raises(DataQualityException, match="Insufficient target variable variance"):
            validator.validate_target_distribution(low_variance_data, 'pass_percentage')
    
    def test_check_sample_sizes_sufficient(self, sample_data):
        """Test sample size checking with sufficient data."""
        # Create validator with lower threshold for testing
        validator = DataValidator(min_samples_per_group=20)
        result = validator.check_sample_sizes(
            sample_data, 
            ['dataset_uuid', 'rule_code']
        )
        
        # Each (uuid, rule) combination should have samples
        assert isinstance(result, dict)
        assert len(result) > 0
        for group_key, count in result.items():
            assert count >= 20
    
    def test_check_sample_sizes_insufficient(self):
        """Test sample size checking with insufficient data per group."""
        # Create data with <50 samples per group
        data = pd.DataFrame({
            'pass_percentage': [50.0] * 30,
            'dataset_uuid': ['uuid1'] * 15 + ['uuid2'] * 15,
            'rule_code': ['R001'] * 30,
            'feature1': range(30)
        })
        
        validator = DataValidator()
        
        with pytest.raises(DataQualityException, match="Insufficient sample size"):
            validator.check_sample_sizes(data, ['dataset_uuid', 'rule_code'])
    
    def test_validate_feature_matrix_rank_good(self, sample_data):
        """Test feature matrix rank validation with good rank."""
        validator = DataValidator()
        feature_data = sample_data[['feature1', 'feature2']]
        
        rank_ratio = validator.validate_feature_matrix_rank(feature_data)
        
        assert rank_ratio > 0.8  # Should have good rank
        assert isinstance(rank_ratio, float)
    
    def test_validate_feature_matrix_rank_poor(self):
        """Test feature matrix rank validation with poor rank (collinear features)."""
        # Create perfectly correlated features
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],  # Perfectly correlated with feature1
            'feature3': [3, 6, 9, 12, 15]  # Also perfectly correlated
        })
        
        validator = DataValidator()
        
        with pytest.raises(DataQualityException, match="Poor feature matrix rank"):
            validator.validate_feature_matrix_rank(data)
    
    def test_generate_quality_report(self, sample_data):
        """Test comprehensive quality report generation."""
        # Use validator with appropriate thresholds for test data
        validator = DataValidator(min_samples_per_group=20)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "quality_report.json"
            
            report = validator.generate_quality_report(sample_data, report_path)
            
            assert isinstance(report, dict)
            assert 'target_distribution' in report
            assert 'sample_sizes' in report
            assert 'feature_matrix_rank' in report
            assert 'timestamp' in report
            
            # Verify file was created
            assert report_path.exists()
            
            # Verify file contents (compare JSON-serialized versions)
            with open(report_path) as f:
                saved_report = json.load(f)
                json_report = validator._convert_numpy_types(report)
                assert saved_report == json_report


class TestValidationReport:
    """Test cases for ValidationReport class."""
    
    def test_validation_report_creation(self):
        """Test ValidationReport object creation."""
        details = {'test': 'data'}
        report = ValidationReport(passed=True, message="Test passed", details=details)
        
        assert report.passed is True
        assert report.message == "Test passed"
        assert report.details == details
    
    def test_validation_report_failure(self):
        """Test ValidationReport for failure case."""
        report = ValidationReport(passed=False, message="Test failed")
        
        assert report.passed is False
        assert report.message == "Test failed"
        assert report.details == {}


class TestDataQualityException:
    """Test cases for DataQualityException."""
    
    def test_exception_creation(self):
        """Test DataQualityException creation."""
        with pytest.raises(DataQualityException, match="Test error"):
            raise DataQualityException("Test error")