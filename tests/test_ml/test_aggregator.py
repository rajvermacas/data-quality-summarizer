"""
Tests for ML aggregator module.

Following TDD approach: Red -> Green -> Refactor
"""
import pandas as pd
import pytest
from datetime import datetime

from src.data_quality_summarizer.ml.aggregator import (
    aggregate_pass_percentages,
    calculate_group_pass_percentage,
    handle_empty_groups
)


class TestMLAggregator:
    """Test suite for ML data aggregation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample processed data with is_pass column
        self.sample_data = {
            'source': ['system_a', 'system_a', 'system_a', 'system_b', 'system_b'],
            'tenant_id': ['tenant1', 'tenant1', 'tenant1', 'tenant2', 'tenant2'],
            'dataset_uuid': ['uuid1', 'uuid1', 'uuid1', 'uuid2', 'uuid2'],
            'dataset_name': ['dataset1', 'dataset1', 'dataset1', 'dataset2', 'dataset2'],
            'business_date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-01', '2024-01-01'],
            'rule_code': ['R001', 'R001', 'R001', 'R002', 'R002'],
            'is_pass': [1, 0, 1, 1, 1],  # 50% pass rate for R001 on 2024-01-01, 100% on 2024-01-02
            'level_of_execution': ['dataset', 'dataset', 'dataset', 'column', 'column'],
            'attribute_name': ['', '', '', 'col1', 'col2']
        }
        self.sample_df = pd.DataFrame(self.sample_data)
    
    def test_aggregate_pass_percentages_basic(self):
        """Test basic aggregation by (dataset_uuid, rule_code, business_date)."""
        result_df = aggregate_pass_percentages(self.sample_df)
        
        # Should have 3 groups: 
        # (uuid1, R001, 2024-01-01) -> 50% pass rate
        # (uuid1, R001, 2024-01-02) -> 100% pass rate  
        # (uuid2, R002, 2024-01-01) -> 100% pass rate
        assert len(result_df) == 3
        assert 'pass_percentage' in result_df.columns
        
        # Check specific pass percentages
        group1 = result_df[
            (result_df['dataset_uuid'] == 'uuid1') & 
            (result_df['rule_code'] == 'R001') & 
            (result_df['business_date'] == '2024-01-01')
        ]
        assert len(group1) == 1
        assert group1.iloc[0]['pass_percentage'] == 50.0
        
        group2 = result_df[
            (result_df['dataset_uuid'] == 'uuid1') & 
            (result_df['rule_code'] == 'R001') & 
            (result_df['business_date'] == '2024-01-02')
        ]
        assert len(group2) == 1
        assert group2.iloc[0]['pass_percentage'] == 100.0
    
    def test_aggregate_pass_percentages_preserves_metadata(self):
        """Test that aggregation preserves required metadata columns."""
        required_columns = [
            'source', 'tenant_id', 'dataset_uuid', 'dataset_name',
            'business_date', 'rule_code', 'pass_percentage'
        ]
        
        result_df = aggregate_pass_percentages(self.sample_df)
        
        for col in required_columns:
            assert col in result_df.columns, f"Missing column: {col}"
    
    def test_calculate_group_pass_percentage_all_pass(self):
        """Test pass percentage calculation when all results pass."""
        is_pass_values = [1, 1, 1, 1]
        
        result = calculate_group_pass_percentage(is_pass_values)
        
        assert result == 100.0
    
    def test_calculate_group_pass_percentage_all_fail(self):
        """Test pass percentage calculation when all results fail."""
        is_pass_values = [0, 0, 0, 0]
        
        result = calculate_group_pass_percentage(is_pass_values)
        
        assert result == 0.0
    
    def test_calculate_group_pass_percentage_mixed(self):
        """Test pass percentage calculation with mixed results."""
        is_pass_values = [1, 0, 1, 0, 1]  # 3 out of 5 pass
        
        result = calculate_group_pass_percentage(is_pass_values)
        
        assert result == 60.0
    
    def test_calculate_group_pass_percentage_single_value(self):
        """Test pass percentage calculation with single value."""
        is_pass_values = [1]
        
        result = calculate_group_pass_percentage(is_pass_values)
        
        assert result == 100.0
    
    def test_handle_empty_groups_empty_list(self):
        """Test handling of empty groups."""
        empty_values = []
        
        result = handle_empty_groups(empty_values)
        
        assert result == 0.0  # Default to 0% for empty groups
    
    def test_aggregate_pass_percentages_empty_dataframe(self):
        """Test aggregation with empty input DataFrame."""
        empty_df = pd.DataFrame(columns=self.sample_df.columns)
        
        result_df = aggregate_pass_percentages(empty_df)
        
        assert len(result_df) == 0
        assert 'pass_percentage' in result_df.columns
    
    def test_aggregate_pass_percentages_single_row(self):
        """Test aggregation with single row input."""
        single_row = self.sample_df.iloc[:1].copy()
        
        result_df = aggregate_pass_percentages(single_row)
        
        assert len(result_df) == 1
        expected_percentage = 100.0 if single_row.iloc[0]['is_pass'] == 1 else 0.0
        assert result_df.iloc[0]['pass_percentage'] == expected_percentage