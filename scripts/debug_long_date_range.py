#!/usr/bin/env python3
"""Debug script to test behavior with 2+ years of data"""

import pandas as pd
from datetime import date, timedelta
import sys
sys.path.append('.')

from src.data_quality_summarizer.aggregator import StreamingAggregator

print("Testing with 2+ years of data starting from future date:")
print("="*60)

# Create test data spanning 2+ years
start_date = date(2025, 6, 20)
aggregator = StreamingAggregator(weeks=1)

# Generate data for every week for 2 years
test_rows = []
for week in range(0, 104, 4):  # Every 4 weeks for brevity (26 data points)
    business_date = start_date + timedelta(weeks=week)
    row_data = {
        "source": "test_system",
        "tenant_id": "tenant_123",
        "dataset_uuid": "uuid-456",
        "dataset_name": "Test Dataset",
        "rule_code": 101,
        "business_date": str(business_date),
        "results": '{"result": "Pass"}',
        "dataset_record_count": 1000 + week * 100,
        "filtered_record_count": 900 + week * 100,
        "level_of_execution": "DATASET",
    }
    test_rows.append((business_date, week))
    row = pd.Series(row_data)
    aggregator.process_row(row)

print(f"\nProcessed {len(test_rows)} rows spanning from {start_date} to {test_rows[-1][0]}")
print(f"Total week groups created: {len(aggregator.accumulator)}")

# Show a sample of week groups
print("\nSample of week groups created:")
sample_keys = sorted(list(aggregator.accumulator.keys()), key=lambda x: x[5])[:5]  # First 5
for key in sample_keys:
    metrics = aggregator.accumulator[key]
    print(f"\n  Week group {key[5]}:")
    print(f"    Boundaries: {metrics.business_week_start_date} to {metrics.business_week_end_date}")
    print(f"    Business date latest: {metrics.business_date_latest}")
    print(f"    Number of rows in this group: {len(metrics.row_data)}")

print("\n...")

# Show last few week groups
last_keys = sorted(list(aggregator.accumulator.keys()), key=lambda x: x[5])[-3:]
for key in last_keys:
    metrics = aggregator.accumulator[key]
    print(f"\n  Week group {key[5]}:")
    print(f"    Boundaries: {metrics.business_week_start_date} to {metrics.business_week_end_date}")
    print(f"    Business date latest: {metrics.business_date_latest}")
    print(f"    Number of rows in this group: {len(metrics.row_data)}")

print("\n" + "="*60)
print("ISSUE IDENTIFIED:")
print("="*60)
print("With 2+ years of data and weekly grouping:")
print("- Each week becomes its own group (104+ groups for 2 years)")
print("- Each group only contains data from that specific week")
print("- business_date_latest for each group is just the date from that week")
print("- If each week only has one data point, business_date_latest equals that date")
print("\nThis explains why the user sees 'always the first one' - each week group")
print("only has one or few entries, so business_date_latest is always the same as")
print("the actual business_date for that group!")

# Test with larger week grouping to show the difference
print("\n" + "="*60)
print("Testing with 4-week grouping (monthly):")
print("="*60)

aggregator_monthly = StreamingAggregator(weeks=4)

# Process same data
for business_date, week in test_rows[:12]:  # First 12 entries
    row_data = {
        "source": "test_system",
        "tenant_id": "tenant_123",
        "dataset_uuid": "uuid-456",
        "dataset_name": "Test Dataset",
        "rule_code": 101,
        "business_date": str(business_date),
        "results": '{"result": "Pass"}',
        "dataset_record_count": 1000 + week * 100,
        "filtered_record_count": 900 + week * 100,
        "level_of_execution": "DATASET",
    }
    row = pd.Series(row_data)
    aggregator_monthly.process_row(row)

print(f"\nWith 4-week grouping, total groups: {len(aggregator_monthly.accumulator)}")
for key, metrics in aggregator_monthly.accumulator.items():
    print(f"\n  Week group {key[5]}:")
    print(f"    Boundaries: {metrics.business_week_start_date} to {metrics.business_week_end_date}")
    print(f"    Business date latest: {metrics.business_date_latest}")
    print(f"    Number of rows in this group: {len(metrics.row_data)}")
    print(f"    Dates in group: {[str(r['business_date']) for r in metrics.row_data]}")