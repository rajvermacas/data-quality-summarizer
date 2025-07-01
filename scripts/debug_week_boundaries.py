#!/usr/bin/env python3
"""Debug script to understand week boundary calculations"""

import pandas as pd
from datetime import date, timedelta
import sys
sys.path.append('.')

from src.data_quality_summarizer.aggregator import StreamingAggregator

# Create test data that demonstrates the issue
# When rows are processed in different order, business_date_latest shows different values
test_data_order1 = [
    {"business_date": "2024-01-15", "record_count": 1000},  # Monday, Week 3
    {"business_date": "2024-01-17", "record_count": 2000},  # Wednesday, Week 3
    {"business_date": "2024-01-19", "record_count": 3000},  # Friday, Week 3
]

test_data_order2 = [
    {"business_date": "2024-01-19", "record_count": 3000},  # Friday first
    {"business_date": "2024-01-15", "record_count": 1000},  # Monday second
    {"business_date": "2024-01-17", "record_count": 2000},  # Wednesday last
]

def process_data(data_order, order_name):
    print(f"\n{'='*60}")
    print(f"Processing data in {order_name}:")
    print('='*60)
    
    aggregator = StreamingAggregator(weeks=1)
    
    print("\nProcessing order:")
    for i, data in enumerate(data_order):
        print(f"  {i+1}. {data['business_date']} (record_count={data['record_count']})")
        
        row_data = {
            "source": "test_system",
            "tenant_id": "tenant_123",
            "dataset_uuid": "uuid-456",
            "dataset_name": "Test Dataset",
            "rule_code": 101,
            "business_date": data['business_date'],
            "results": '{"result": "Pass"}',
            "dataset_record_count": data['record_count'],
            "filtered_record_count": data['record_count'] - 50,
            "level_of_execution": "DATASET",
        }
        
        row = pd.Series(row_data)
        aggregator.process_row(row)
    
    print(f"\nEarliest business date: {aggregator.earliest_business_date}")
    print(f"Latest business date: {aggregator.latest_business_date}")
    
    print("\nAccumulator contents:")
    for key, metrics in aggregator.accumulator.items():
        print(f"\n  Key (week_group={key[5]}):")
        print(f"    Week boundaries: {metrics.business_week_start_date} to {metrics.business_week_end_date}")
        print(f"    Business date latest: {metrics.business_date_latest}")
        print(f"    Dataset record count latest: {metrics.dataset_record_count_latest}")
        print(f"    Row data processed:")
        for row in metrics.row_data:
            print(f"      - {row['business_date']}: record_count={row['dataset_record_count']}")

# Test both orderings
process_data(test_data_order1, "Order 1: Chronological")
process_data(test_data_order2, "Order 2: Reverse")

# Now test with dates spanning multiple weeks
print(f"\n{'='*60}")
print("Testing with dates spanning multiple weeks:")
print('='*60)

multi_week_data = [
    {"business_date": "2024-01-08", "record_count": 100},   # Week 2, Monday
    {"business_date": "2024-01-15", "record_count": 200},   # Week 3, Monday
    {"business_date": "2024-01-10", "record_count": 300},   # Week 2, Wednesday
    {"business_date": "2024-01-17", "record_count": 400},   # Week 3, Wednesday
    {"business_date": "2024-01-22", "record_count": 500},   # Week 4, Monday
]

aggregator = StreamingAggregator(weeks=1)

print("\nProcessing dates:")
for data in multi_week_data:
    print(f"  {data['business_date']} (record_count={data['record_count']})")
    
    row_data = {
        "source": "test_system",
        "tenant_id": "tenant_123",
        "dataset_uuid": "uuid-456",
        "dataset_name": "Test Dataset",
        "rule_code": 101,
        "business_date": data['business_date'],
        "results": '{"result": "Pass"}',
        "dataset_record_count": data['record_count'],
        "filtered_record_count": data['record_count'] - 10,
        "level_of_execution": "DATASET",
    }
    
    row = pd.Series(row_data)
    aggregator.process_row(row)

print(f"\nEarliest: {aggregator.earliest_business_date}, Latest: {aggregator.latest_business_date}")

print("\nWeek groups created:")
week_groups = {}
for key, metrics in aggregator.accumulator.items():
    wg = key[5]
    if wg not in week_groups:
        week_groups[wg] = []
    week_groups[wg].append(metrics)

for wg in sorted(week_groups.keys()):
    metrics = week_groups[wg][0]  # All metrics in same group have same boundaries
    print(f"\n  Week group {wg}:")
    print(f"    Boundaries: {metrics.business_week_start_date} to {metrics.business_week_end_date}")
    print(f"    Business date latest: {metrics.business_date_latest}")
    print(f"    Dataset record count latest: {metrics.dataset_record_count_latest}")
    print(f"    Dates in this group:")
    for row in metrics.row_data:
        print(f"      - {row['business_date']}: record_count={row['dataset_record_count']}")