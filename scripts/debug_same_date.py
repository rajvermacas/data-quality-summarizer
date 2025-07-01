#!/usr/bin/env python3
"""Debug script to test behavior with multiple rows having the same business_date"""

import pandas as pd
from datetime import date
import sys
sys.path.append('.')

from src.data_quality_summarizer.aggregator import StreamingAggregator

# Create test data where multiple rows have the same business_date
# This should reveal if the first or last row's values are kept
test_data_same_date = [
    {
        "source": "test_system",
        "tenant_id": "tenant_123",
        "dataset_uuid": "uuid-456",
        "dataset_name": "Test Dataset",
        "rule_code": 101,
        "business_date": "2024-01-15",  # Same date
        "results": '{"result": "Pass"}',
        "dataset_record_count": 1000,  # First value
        "filtered_record_count": 900,
        "level_of_execution": "DATASET",
    },
    {
        "source": "test_system",
        "tenant_id": "tenant_123",
        "dataset_uuid": "uuid-456",
        "dataset_name": "Test Dataset",
        "rule_code": 101,
        "business_date": "2024-01-15",  # Same date
        "results": '{"result": "Fail"}',
        "dataset_record_count": 2000,  # Different value
        "filtered_record_count": 1800,
        "level_of_execution": "ATTRIBUTE",
    },
    {
        "source": "test_system",
        "tenant_id": "tenant_123",
        "dataset_uuid": "uuid-456",
        "dataset_name": "Test Dataset",
        "rule_code": 101,
        "business_date": "2024-01-15",  # Same date
        "results": '{"result": "Pass"}',
        "dataset_record_count": 3000,  # Last value
        "filtered_record_count": 2700,
        "level_of_execution": "COLUMN",
    },
]

print("Testing with multiple rows having the same business_date:")
print("="*60)

aggregator = StreamingAggregator(weeks=1)

print("\nProcessing rows:")
for i, row_data in enumerate(test_data_same_date):
    print(f"  Row {i+1}: date={row_data['business_date']}, " +
          f"dataset_record_count={row_data['dataset_record_count']}, " +
          f"level={row_data['level_of_execution']}")
    row = pd.Series(row_data)
    aggregator.process_row(row)

print("\nResult:")
for key, metrics in aggregator.accumulator.items():
    print(f"  Business date latest: {metrics.business_date_latest}")
    print(f"  Dataset record count latest: {metrics.dataset_record_count_latest}")
    print(f"  Last execution level: {metrics.last_execution_level}")
    print(f"  Pass count: {metrics.pass_count}, Fail count: {metrics.fail_count}")

print("\nExpected behavior:")
print("  - If using >= comparison: Should keep FIRST row's values (1000, DATASET)")
print("  - If using > comparison: Should keep LAST row's values (3000, COLUMN)")

# Now test with mixed dates to see the pattern
print("\n" + "="*60)
print("Testing with mixed dates:")
print("="*60)

mixed_data = [
    {"business_date": "2024-01-14", "dataset_record_count": 100},
    {"business_date": "2024-01-15", "dataset_record_count": 200},
    {"business_date": "2024-01-15", "dataset_record_count": 300},  # Same as previous
    {"business_date": "2024-01-16", "dataset_record_count": 400},
    {"business_date": "2024-01-15", "dataset_record_count": 500},  # Earlier than previous
]

aggregator2 = StreamingAggregator(weeks=1)

print("\nProcessing rows:")
for i, data in enumerate(mixed_data):
    row_data = {
        "source": "test_system",
        "tenant_id": "tenant_123",
        "dataset_uuid": "uuid-456",
        "dataset_name": "Test Dataset",
        "rule_code": 101,
        "business_date": data['business_date'],
        "results": '{"result": "Pass"}',
        "dataset_record_count": data['dataset_record_count'],
        "filtered_record_count": data['dataset_record_count'] - 10,
        "level_of_execution": "DATASET",
    }
    print(f"  Row {i+1}: date={data['business_date']}, count={data['dataset_record_count']}")
    row = pd.Series(row_data)
    aggregator2.process_row(row)

print("\nResult:")
for key, metrics in aggregator2.accumulator.items():
    print(f"  Business date latest: {metrics.business_date_latest}")
    print(f"  Dataset record count latest: {metrics.dataset_record_count_latest}")
    
print("\nIssue: When the last row has an earlier date (2024-01-15) than the")
print("previously processed row (2024-01-16), it won't update even though")
print("it's the most recently processed row for that date.")