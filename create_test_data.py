import pandas as pd
import json
from datetime import datetime, timedelta

# Create test CSV data
data = []
base_date = datetime(2024, 1, 1)
for i in range(100):
    status = 'Pass' if i % 4 != 0 else 'Fail'
    data.append({
        'source': 'test_source',
        'tenant_id': 'test_tenant', 
        'dataset_uuid': f'dataset-{i%3 + 1:03d}',
        'dataset_name': f'Test Dataset {i%3 + 1}',
        'business_date': (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
        'rule_code': (i % 2) + 1,
        'results': json.dumps({'status': status}),
        'level_of_execution': 'dataset',
        'attribute_name': None,
        'dataset_record_count': 1000,
        'filtered_record_count': 950
    })

df = pd.DataFrame(data)
df.to_csv('test_ml_data.csv', index=False)
print(f'Created test_ml_data.csv with {len(df)} rows')

# Create test rules metadata
rules = {
    1: {
        'rule_name': 'Completeness Check',
        'rule_type': 'Completeness',
        'dimension': 'Completeness',
        'rule_description': 'Check for missing values',
        'category': 'C1'
    },
    2: {
        'rule_name': 'Format Validation',
        'rule_type': 'Validity',
        'dimension': 'Validity',
        'rule_description': 'Validate data format',
        'category': 'V1'
    }
}

with open('test_rules.json', 'w') as f:
    json.dump(rules, f, indent=2)
print('Created test_rules.json')