"""
Test suite for ML Data Splitter module.

This module tests the chronological train/test splitting functionality
for the predictive model pipeline.
"""
import unittest
import pandas as pd
from datetime import datetime, timedelta
from src.data_quality_summarizer.ml.data_splitter import (
    split_data_chronologically,
    validate_temporal_ordering,
    determine_optimal_cutoff_date
)


class TestDataSplitter(unittest.TestCase):
    """Test chronological data splitting functionality."""

    def setUp(self):
        """Create test data for splitting tests."""
        # Create sample data spanning 30 days
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(30)]
        
        self.sample_data = pd.DataFrame({
            'dataset_uuid': ['dataset1'] * 30,
            'rule_code': ['R001'] * 30,
            'business_date': dates,
            'pass_percentage': [85.0 + i for i in range(30)],  # Ascending values
            'day_of_week': [d.weekday() for d in dates]
        })

    def test_split_data_chronologically_basic(self):
        """Test basic chronological split functionality."""
        cutoff_date = datetime(2024, 1, 20)
        
        train_data, test_data = split_data_chronologically(
            self.sample_data, cutoff_date
        )
        
        # Should have data for first 19 days in train, remaining in test
        self.assertEqual(len(train_data), 19)
        self.assertEqual(len(test_data), 11)
        
        # Check temporal ordering is maintained
        self.assertTrue(all(train_data['business_date'] < cutoff_date))
        self.assertTrue(all(test_data['business_date'] >= cutoff_date))

    def test_split_data_chronologically_edge_case_cutoff(self):
        """Test edge cases for cutoff date selection."""
        # Cutoff before all data
        early_cutoff = datetime(2023, 12, 31)
        train_data, test_data = split_data_chronologically(
            self.sample_data, early_cutoff
        )
        self.assertEqual(len(train_data), 0)
        self.assertEqual(len(test_data), 30)
        
        # Cutoff after all data
        late_cutoff = datetime(2024, 2, 1)
        train_data, test_data = split_data_chronologically(
            self.sample_data, late_cutoff
        )
        self.assertEqual(len(train_data), 30)
        self.assertEqual(len(test_data), 0)

    def test_validate_temporal_ordering(self):
        """Test temporal ordering validation."""
        # Valid ordering should pass
        self.assertTrue(validate_temporal_ordering(self.sample_data))
        
        # Invalid ordering should fail
        shuffled_data = self.sample_data.sample(frac=1).reset_index(drop=True)
        self.assertFalse(validate_temporal_ordering(shuffled_data))
        
        # Empty data should pass
        empty_data = pd.DataFrame(columns=self.sample_data.columns)
        self.assertTrue(validate_temporal_ordering(empty_data))

    def test_determine_optimal_cutoff_date_default_split(self):
        """Test optimal cutoff date determination with default 80/20 split."""
        cutoff_date = determine_optimal_cutoff_date(self.sample_data)
        
        # Should split approximately 80/20
        train_data, test_data = split_data_chronologically(
            self.sample_data, cutoff_date
        )
        
        train_ratio = len(train_data) / len(self.sample_data)
        self.assertAlmostEqual(train_ratio, 0.8, delta=0.1)

    def test_determine_optimal_cutoff_date_custom_split(self):
        """Test optimal cutoff date with custom train ratio."""
        cutoff_date = determine_optimal_cutoff_date(
            self.sample_data, train_ratio=0.7
        )
        
        train_data, test_data = split_data_chronologically(
            self.sample_data, cutoff_date
        )
        
        train_ratio = len(train_data) / len(self.sample_data)
        self.assertAlmostEqual(train_ratio, 0.7, delta=0.1)

    def test_split_preserves_all_columns(self):
        """Test that splitting preserves all original columns."""
        cutoff_date = datetime(2024, 1, 20)
        train_data, test_data = split_data_chronologically(
            self.sample_data, cutoff_date
        )
        
        # All columns should be preserved
        self.assertEqual(list(train_data.columns), list(self.sample_data.columns))
        self.assertEqual(list(test_data.columns), list(self.sample_data.columns))

    def test_split_with_multiple_dataset_groups(self):
        """Test splitting works correctly with multiple dataset/rule combinations."""
        # Create data with multiple groups
        multi_group_data = pd.concat([
            self.sample_data,
            self.sample_data.assign(dataset_uuid='dataset2'),
            self.sample_data.assign(rule_code='R002')
        ]).reset_index(drop=True)
        
        cutoff_date = datetime(2024, 1, 20)
        train_data, test_data = split_data_chronologically(
            multi_group_data, cutoff_date
        )
        
        # Each group should be split consistently
        unique_groups_train = train_data.groupby(['dataset_uuid', 'rule_code']).size()
        unique_groups_test = test_data.groupby(['dataset_uuid', 'rule_code']).size()
        
        # Should have same number of unique groups
        self.assertEqual(len(unique_groups_train), len(unique_groups_test))


class TestDataSplitterEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for data splitter."""

    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        empty_data = pd.DataFrame(columns=[
            'dataset_uuid', 'rule_code', 'business_date', 'pass_percentage'
        ])
        
        cutoff_date = datetime(2024, 1, 15)
        train_data, test_data = split_data_chronologically(
            empty_data, cutoff_date
        )
        
        self.assertEqual(len(train_data), 0)
        self.assertEqual(len(test_data), 0)

    def test_single_row_dataframe(self):
        """Test handling of single-row dataframes."""
        single_row = pd.DataFrame({
            'dataset_uuid': ['dataset1'],
            'rule_code': ['R001'],
            'business_date': [datetime(2024, 1, 15)],
            'pass_percentage': [85.0]
        })
        
        # Cutoff before the row
        cutoff_date = datetime(2024, 1, 10)
        train_data, test_data = split_data_chronologically(
            single_row, cutoff_date
        )
        self.assertEqual(len(train_data), 0)
        self.assertEqual(len(test_data), 1)
        
        # Cutoff after the row
        cutoff_date = datetime(2024, 1, 20)
        train_data, test_data = split_data_chronologically(
            single_row, cutoff_date
        )
        self.assertEqual(len(train_data), 1)
        self.assertEqual(len(test_data), 0)

    def test_invalid_cutoff_date_type(self):
        """Test handling of invalid cutoff date types."""
        sample_data = pd.DataFrame({
            'business_date': [datetime(2024, 1, 1)],
            'pass_percentage': [85.0]
        })
        
        with self.assertRaises(TypeError):
            split_data_chronologically(sample_data, "2024-01-15")
        
        with self.assertRaises(TypeError):
            split_data_chronologically(sample_data, None)


if __name__ == '__main__':
    unittest.main()