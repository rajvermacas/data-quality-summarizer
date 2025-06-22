#!/usr/bin/env python3
"""
Generate larger test dataset for ML pipeline demonstration.
Creates synthetic data quality check results with realistic patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random

# Configuration
NUM_SAMPLES = 1000
DATE_RANGE_DAYS = 365
RULE_CODES = ['101', '102', '103', '104']
DATASETS = [
    ('uuid-123', 'Customer_Data'),
    ('uuid-456', 'Order_Data'), 
    ('uuid-789', 'Product_Data'),
    ('uuid-abc', 'Sales_Data'),
    ('uuid-def', 'Inventory_Data')
]

def generate_test_data():
    """Generate synthetic test data with realistic patterns."""
    
    # Base date for generation
    base_date = datetime(2024, 1, 1)
    
    data = []
    
    for i in range(NUM_SAMPLES):
        # Random date within range
        days_offset = random.randint(0, DATE_RANGE_DAYS)
        business_date = (base_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')
        
        # Random dataset
        dataset_uuid, dataset_name = random.choice(DATASETS)
        
        # Random rule code
        rule_code = random.choice(RULE_CODES)
        
        # Generate pass/fail based on some patterns
        # Rule 101 (ROW_COUNT_CHECK) - generally high pass rate
        # Rule 102 (NULL_VALUE_CHECK) - medium pass rate with some trends
        # Rule 103 (DATA_FRESHNESS) - high pass rate
        # Rule 104 (RANGE_VALIDATION) - variable pass rate
        
        if rule_code == '101':
            pass_prob = 0.85 + 0.1 * np.sin(days_offset / 30)  # Seasonal pattern
        elif rule_code == '102':
            pass_prob = 0.7 + 0.2 * np.cos(days_offset / 60)  # Longer cycle
        elif rule_code == '103':
            pass_prob = 0.9 - 0.1 * (days_offset / DATE_RANGE_DAYS)  # Declining trend
        else:  # rule_code == '104'
            pass_prob = 0.6 + 0.3 * random.random()  # Random variation
        
        # Ensure pass_prob is between 0 and 1
        pass_prob = max(0.05, min(0.95, pass_prob))
        
        # Generate pass/fail result
        result = "Pass" if random.random() < pass_prob else "Fail"
        
        # Create row data
        row = {
            'source': 'TestSystem',
            'tenant_id': f'tenant{random.randint(1, 3)}',
            'dataset_uuid': dataset_uuid,
            'dataset_name': dataset_name,
            'business_date': business_date,
            'dataset_record_count': random.randint(1000, 50000),
            'rule_code': rule_code,
            'level_of_execution': 'DATASET' if rule_code in ['101', '103'] else 'ATTRIBUTE',
            'attribute_name': '' if rule_code in ['101', '103'] else random.choice(['customer_id', 'order_date', 'amount', 'status']),
            'results': json.dumps({"result": result}),
            'context_id': f'ctx{i+1}',
            'filtered_record_count': random.randint(800, 45000)
        }
        
        data.append(row)
    
    return pd.DataFrame(data)

def main():
    """Generate and save test data."""
    print(f"Generating {NUM_SAMPLES} samples...")
    
    # Generate data
    df = generate_test_data()
    
    # Save to CSV
    output_file = 'large_test_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Generated {len(df)} samples")
    print(f"✓ Date range: {df['business_date'].min()} to {df['business_date'].max()}")
    print(f"✓ Rule codes: {sorted(df['rule_code'].unique())}")
    print(f"✓ Datasets: {len(df['dataset_uuid'].unique())}")
    print(f"✓ Pass rate distribution:")
    
    for rule_code in RULE_CODES:
        rule_data = df[df['rule_code'] == rule_code]
        passes = sum(json.loads(r)['result'] == 'Pass' for r in rule_data['results'])
        total = len(rule_data)
        print(f"   Rule {rule_code}: {passes}/{total} ({passes/total*100:.1f}%)")
    
    print(f"✓ Saved to: {output_file}")

if __name__ == '__main__':
    main()