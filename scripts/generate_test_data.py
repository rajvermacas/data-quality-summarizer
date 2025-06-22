#!/usr/bin/env python3
"""
Generate test data for CI/CD pipeline testing.

This script creates realistic test data that matches the expected schema
for the ML pipeline testing in automated environments.
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path


def generate_test_data(size: int, output_path: str):
    """
    Generate test data CSV file.
    
    Args:
        size: Number of records to generate
        output_path: Path to save the CSV file
    """
    np.random.seed(42)  # Reproducible data for CI
    
    # Generate realistic data patterns
    datasets = ['dataset-001', 'dataset-002', 'dataset-003']
    rule_codes = [1, 2, 3, 4, 5]
    
    data = []
    dates = pd.date_range('2024-01-01', periods=size//50, freq='D')
    
    for i, date in enumerate(dates):
        for dataset_uuid in datasets:
            for rule_code in rule_codes:
                # Generate multiple executions per day/dataset/rule
                num_executions = np.random.randint(2, 4)
                
                for _ in range(num_executions):
                    # Simulate realistic pass/fail patterns
                    base_pass_rate = 0.8 + np.random.normal(0, 0.1)
                    base_pass_rate = max(0.1, min(0.99, base_pass_rate))  # Clamp to valid range
                    
                    is_pass = np.random.random() < base_pass_rate
                    
                    data.append({
                        'source': 'ci_test',
                        'tenant_id': 'test_tenant',
                        'dataset_uuid': dataset_uuid,
                        'dataset_name': f'Test Dataset {dataset_uuid[-1]}',
                        'business_date': date.strftime('%Y-%m-%d'),
                        'rule_code': rule_code,
                        'results': json.dumps({'status': 'Pass' if is_pass else 'Fail'}),
                        'level_of_execution': 'dataset',
                        'attribute_name': None,
                        'dataset_record_count': np.random.randint(1000, 5000),
                        'filtered_record_count': np.random.randint(900, 4500)
                    })
                    
                    if len(data) >= size:
                        break
                
                if len(data) >= size:
                    break
            
            if len(data) >= size:
                break
        
        if len(data) >= size:
            break
    
    # Create DataFrame and save
    df = pd.DataFrame(data[:size])
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} test records and saved to {output_path}")
    
    # Print data summary for verification
    print(f"Data summary:")
    print(f"  Date range: {df['business_date'].min()} to {df['business_date'].max()}")
    print(f"  Datasets: {df['dataset_uuid'].nunique()}")
    print(f"  Rule codes: {sorted(df['rule_code'].unique())}")
    print(f"  Pass rate: {df['results'].str.contains('Pass').mean():.1%}")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Generate test data for CI/CD pipeline')
    parser.add_argument('--size', type=int, default=1000, help='Number of records to generate')
    parser.add_argument('--output', type=str, default='test_data.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    generate_test_data(args.size, args.output)


if __name__ == '__main__':
    main()