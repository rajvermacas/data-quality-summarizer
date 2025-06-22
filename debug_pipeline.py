#!/usr/bin/env python3
"""Debug script to see actual pipeline errors."""

import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock

from src.data_quality_summarizer.ml.pipeline import MLPipeline

def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test CSV file
        test_csv = Path(temp_dir) / "test_data.csv"
        test_data = pd.DataFrame({
            'source': ['test'] * 10,
            'tenant_id': ['tenant1'] * 10,
            'dataset_uuid': ['uuid1'] * 10,
            'dataset_name': ['dataset1'] * 10,
            'rule_code': ['R001'] * 10,
            'business_date': ['2024-01-01'] * 5 + ['2024-01-02'] * 5,
            'results': ['{"status": "Pass"}'] * 7 + ['{"status": "Fail"}'] * 3,
            'dataset_record_count': [1000] * 10,
            'filtered_record_count': [900] * 10,
            'level_of_execution': ['dataset'] * 10,
            'attribute_name': [None] * 10
        })
        test_data.to_csv(test_csv, index=False)
        
        # Create rule metadata
        rule_metadata = {'R001': Mock(rule_name='Test Rule')}
        
        # Test model path
        model_path = Path(temp_dir) / "test_model.pkl"
        
        pipeline = MLPipeline()
        
        # This should execute the full training pipeline
        result = pipeline.train_model(
            csv_file=str(test_csv),
            rule_metadata=rule_metadata,
            output_model_path=str(model_path)
        )
        
        print("Pipeline result:")
        print(result)

if __name__ == "__main__":
    main()