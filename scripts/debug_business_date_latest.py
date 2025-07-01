#!/usr/bin/env python3
"""Debug script to test business_date_latest handling"""

import pandas as pd
from datetime import date
import sys
sys.path.append('.')

from src.data_quality_summarizer.aggregator import StreamingAggregator

# Create test data with multiple dates in random order
test_data = [
    {
        "source": "test_system",
        "tenant_id": "tenant_123",
        "dataset_uuid": "uuid-456",
        "dataset_name": "Test Dataset",
        "rule_code": 101,
        "business_date": "2024-01-15",  # Middle date
        "results": '{"result": "Pass"}',
        "dataset_record_count": 1000,
        "filtered_record_count": 950,
        "level_of_execution": "DATASET",
    },
    {
        "source": "test_system",
        "tenant_id": "tenant_123",
        "dataset_uuid": "uuid-456",
        "dataset_name": "Test Dataset",
        "rule_code": 101,
        "business_date": "2024-01-10",  # Earlier date
        "results": '{"result": "Fail"}',
        "dataset_record_count": 2000,
        "filtered_record_count": 1900,
        "level_of_execution": "DATASET",
    },
    {
        "source": "test_system",
        "tenant_id": "tenant_123",
        "dataset_uuid": "uuid-456",
        "dataset_name": "Test Dataset",
        "rule_code": 101,
        "business_date": "2024-01-20",  # Latest date
        "results": '{"result": "Pass"}',
        "dataset_record_count": 3000,
        "filtered_record_count": 2900,
        "level_of_execution": "DATASET",
    },
    {
        "source": "test_system",
        "tenant_id": "tenant_123",
        "dataset_uuid": "uuid-456",
        "dataset_name": "Test Dataset",
        "rule_code": 101,
        "business_date": "2024-01-17",  # Another middle date
        "results": '{"result": "Pass"}',
        "dataset_record_count": 4000,
        "filtered_record_count": 3900,
        "level_of_execution": "DATASET",
    },
]

# Test with 1-week grouping
aggregator = StreamingAggregator(weeks=1)

print("Processing rows in order:")
for i, row_data in enumerate(test_data):
    print(f"  Row {i+1}: date={row_data['business_date']}, dataset_record_count={row_data['dataset_record_count']}")
    row = pd.Series(row_data)
    aggregator.process_row(row)

# Check the accumulator
print("\nAccumulator contents:")
for key, metrics in aggregator.accumulator.items():
    print(f"\nKey: {key}")
    print(f"  Week group: {metrics.week_group}")
    print(f"  Business date latest: {metrics.business_date_latest}")
    print(f"  Dataset record count latest: {metrics.dataset_record_count_latest}")
    print(f"  Pass count: {metrics.pass_count}")
    print(f"  Fail count: {metrics.fail_count}")

print(f"\nGlobal latest business date: {aggregator.latest_business_date}")

# Now test with 2-week grouping to see if multiple groups are created
print("\n" + "="*60)
print("Testing with 2-week grouping:")

aggregator2 = StreamingAggregator(weeks=2)

# Add more dates to span multiple week groups
extended_data = test_data + [
    {
        "source": "test_system",
        "tenant_id": "tenant_123",
        "dataset_uuid": "uuid-456",
        "dataset_name": "Test Dataset",
        "rule_code": 101,
        "business_date": "2024-01-29",  # Much later date (3rd week)
        "results": '{"result": "Pass"}',
        "dataset_record_count": 5000,
        "filtered_record_count": 4900,
        "level_of_execution": "DATASET",
    },
]

for row_data in extended_data:
    row = pd.Series(row_data)
    aggregator2.process_row(row)

print("\nAccumulator contents with 2-week grouping:")
for key, metrics in aggregator2.accumulator.items():
    print(f"\nKey: {key}")
    print(f"  Week group: {metrics.week_group}")
    print(f"  Week boundaries: {metrics.business_week_start_date} to {metrics.business_week_end_date}")
    print(f"  Business date latest: {metrics.business_date_latest}")
    print(f"  Dataset record count latest: {metrics.dataset_record_count_latest}")
    print(f"  Pass count: {metrics.pass_count}")
    print(f"  Fail count: {metrics.fail_count}")