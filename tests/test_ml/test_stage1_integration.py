"""
Integration tests for Stage 1: Data Preparation & Feature Engineering

Tests the complete pipeline from raw CSV to engineered features.
"""
import pandas as pd
import pytest
import tempfile
import os

from src.data_quality_summarizer.ml.data_loader import (
    load_and_validate_csv, parse_results_column, create_binary_pass_column
)
from src.data_quality_summarizer.ml.aggregator import aggregate_pass_percentages
from src.data_quality_summarizer.ml.feature_engineer import engineer_all_features


class TestStage1Integration:
    """Integration tests for complete Stage 1 pipeline."""
    
    def setup_method(self):
        """Set up test fixtures with realistic data."""
        # Create realistic test data similar to actual input
        self.sample_raw_data = {
            'source': ['system_a', 'system_a', 'system_a', 'system_a', 'system_b', 'system_b'],
            'tenant_id': ['tenant1', 'tenant1', 'tenant1', 'tenant1', 'tenant2', 'tenant2'],
            'dataset_uuid': ['uuid1', 'uuid1', 'uuid1', 'uuid1', 'uuid2', 'uuid2'],
            'dataset_name': ['dataset1', 'dataset1', 'dataset1', 'dataset1', 'dataset2', 'dataset2'],
            'business_date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-01', '2024-01-02'],
            'rule_code': ['R001', 'R001', 'R001', 'R001', 'R002', 'R002'],
            'results': [
                '{"status": "Pass", "value": 100}',
                '{"status": "Fail", "value": 50}',
                '{"status": "Pass", "value": 95}',
                '{"status": "Pass", "value": 85}',
                '{"status": "Pass", "value": 90}',
                '{"status": "Fail", "value": 40}'
            ],
            'level_of_execution': ['dataset', 'dataset', 'dataset', 'dataset', 'column', 'column'],
            'attribute_name': ['', '', '', '', 'col1', 'col2']
        }
    
    def test_complete_stage1_pipeline(self):
        """Test complete pipeline from raw CSV to engineered features."""
        # Create temporary CSV file
        input_df = pd.DataFrame(self.sample_raw_data)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            input_df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Step 1: Load and validate CSV
            raw_df = load_and_validate_csv(temp_path)
            assert len(raw_df) == 6
            
            # Step 2: Parse results column
            parsed_results = parse_results_column(raw_df['results'].tolist())
            assert len(parsed_results) == 6
            assert parsed_results[0]['status'] == 'Pass'
            assert parsed_results[1]['status'] == 'Fail'
            
            # Step 3: Create binary pass column
            raw_df['is_pass'] = create_binary_pass_column(parsed_results)
            assert raw_df['is_pass'].tolist() == [1, 0, 1, 1, 1, 0]
            
            # Step 4: Aggregate pass percentages
            aggregated_df = aggregate_pass_percentages(raw_df)
            
            # Should have 4 groups:
            # (uuid1, R001, 2024-01-01) -> 1 pass, 1 fail -> 50%
            # (uuid1, R001, 2024-01-02) -> 1 pass -> 100%
            # (uuid1, R001, 2024-01-03) -> 1 pass -> 100%
            # (uuid2, R002, 2024-01-01) -> 1 pass -> 100%
            # (uuid2, R002, 2024-01-02) -> 0 pass, 1 fail -> 0%
            assert len(aggregated_df) == 5
            
            # Check specific pass percentages
            jan_1_uuid1 = aggregated_df[
                (aggregated_df['dataset_uuid'] == 'uuid1') & 
                (aggregated_df['rule_code'] == 'R001') & 
                (aggregated_df['business_date'] == '2024-01-01')
            ]
            assert len(jan_1_uuid1) == 1
            assert jan_1_uuid1.iloc[0]['pass_percentage'] == 50.0
            
            # Step 5: Engineer features
            final_df = engineer_all_features(aggregated_df)
            
            # Check that all feature types are present
            expected_features = [
                'day_of_week', 'day_of_month', 'week_of_year', 'month',  # Time features
                'lag_1_day', 'lag_2_day', 'lag_7_day',  # Lag features
                'ma_3_day', 'ma_7_day'  # Moving averages
            ]
            
            for feature in expected_features:
                assert feature in final_df.columns, f"Missing feature: {feature}"
            
            # Verify data integrity
            assert len(final_df) == len(aggregated_df)
            assert 'pass_percentage' in final_df.columns
            
            # Check time features for first row (2024-01-01)
            first_row = final_df.iloc[0]
            assert first_row['day_of_week'] == 0  # Monday
            assert first_row['month'] == 1  # January
            
        finally:
            os.unlink(temp_path)
    
    def test_pipeline_with_malformed_data(self):
        """Test pipeline handles malformed JSON gracefully."""
        # Add malformed JSON to test data
        malformed_data = self.sample_raw_data.copy()
        malformed_data['results'][2] = 'invalid json'
        
        input_df = pd.DataFrame(malformed_data)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            input_df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Run complete pipeline
            raw_df = load_and_validate_csv(temp_path)
            parsed_results = parse_results_column(raw_df['results'].tolist())
            raw_df['is_pass'] = create_binary_pass_column(parsed_results)
            aggregated_df = aggregate_pass_percentages(raw_df)
            final_df = engineer_all_features(aggregated_df)
            
            # Pipeline should complete successfully despite malformed data
            assert len(final_df) > 0
            assert 'pass_percentage' in final_df.columns
            
            # Malformed JSON should be treated as fail (0)
            assert parsed_results[2] is None
            assert raw_df.iloc[2]['is_pass'] == 0
            
        finally:
            os.unlink(temp_path)
    
    def test_pipeline_performance_requirements(self):
        """Test that pipeline meets performance requirements."""
        import time
        import psutil
        import os
        
        # Create larger dataset for performance testing
        larger_data = []
        for i in range(1000):  # 1000 rows
            row = {
                'source': f'system_{i % 10}',
                'tenant_id': f'tenant_{i % 5}',
                'dataset_uuid': f'uuid_{i % 100}',
                'dataset_name': f'dataset_{i % 100}',
                'business_date': f'2024-01-{(i % 30) + 1:02d}',
                'rule_code': f'R{i % 50:03d}',
                'results': '{"status": "Pass", "value": 95}' if i % 3 == 0 else '{"status": "Fail", "value": 45}',
                'level_of_execution': 'dataset',
                'attribute_name': ''
            }
            larger_data.append(row)
        
        input_df = pd.DataFrame(larger_data)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            input_df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.time()
            
            # Run complete pipeline
            raw_df = load_and_validate_csv(temp_path)
            parsed_results = parse_results_column(raw_df['results'].tolist())
            raw_df['is_pass'] = create_binary_pass_column(parsed_results)
            aggregated_df = aggregate_pass_percentages(raw_df)
            final_df = engineer_all_features(aggregated_df)
            
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Performance assertions
            processing_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Should complete in reasonable time for 1000 rows
            assert processing_time < 30, f"Processing took {processing_time:.2f}s, expected <30s"
            
            # Memory usage should be reasonable
            assert memory_usage < 100, f"Memory usage {memory_usage:.2f}MB, expected <100MB"
            
            # Verify output quality
            assert len(final_df) > 0
            assert len(final_df) < len(raw_df)  # Should be aggregated
            
        finally:
            os.unlink(temp_path)