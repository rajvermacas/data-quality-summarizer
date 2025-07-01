#!/usr/bin/env python3
"""Debug script to show the week calculation bug"""

from datetime import date, timedelta

# Simulate the current week calculation logic
def calculate_week_group_current(business_date, earliest_date, weeks=1):
    days_since_start = (business_date - earliest_date).days
    week_number = days_since_start // 7
    week_group = week_number // weeks
    return week_group

# Test with the actual dates from sample data
earliest_date = date(2024, 1, 15)
test_dates = [
    date(2024, 1, 15),
    date(2024, 1, 16),
    date(2024, 1, 17),
    date(2025, 1, 15),
    date(2025, 1, 16),
    date(2026, 1, 15),
    date(2024, 2, 1),
]

print("Current week calculation (from earliest date forward):")
print("="*60)
print(f"Earliest date: {earliest_date}")
print("\nDate calculations:")

for test_date in test_dates:
    days_since = (test_date - earliest_date).days
    week_num = days_since // 7
    week_group = calculate_week_group_current(test_date, earliest_date)
    print(f"{test_date}: {days_since:4d} days → week {week_num:3d} → group {week_group:3d}")

print("\nPROBLEM: Dates from different years get different week groups,")
print("but the week boundaries might overlap or be confusing!")

# Show what happens with the boundary calculation
def get_week_boundaries_current(week_group, earliest_date, weeks=1):
    start_week = week_group * weeks
    start_date = earliest_date + timedelta(days=start_week * 7)
    days_since_monday = start_date.weekday()
    start_date = start_date - timedelta(days=days_since_monday)
    end_date = start_date + timedelta(days=(weeks * 7) - 1)
    return start_date, end_date

print("\n" + "="*60)
print("Week boundaries for each group:")
for wg in [0, 52, 104, 2]:
    start, end = get_week_boundaries_current(wg, earliest_date)
    print(f"Week group {wg:3d}: {start} to {end}")

# Proposed fix: Calculate from latest date backward
latest_date = date(2026, 1, 15)

print("\n" + "="*60)
print("Proposed fix: Calculate from latest date backward:")
print(f"Latest date: {latest_date}")

def calculate_week_group_fixed(business_date, latest_date, weeks=1):
    # Calculate how many days BEFORE the latest date
    days_before_latest = (latest_date - business_date).days
    # Ensure we're calculating weeks backward
    week_number = days_before_latest // 7
    week_group = week_number // weeks
    return week_group

print("\nDate calculations (backward from latest):")
for test_date in sorted(test_dates, reverse=True):
    days_before = (latest_date - test_date).days
    week_num = days_before // 7
    week_group = calculate_week_group_fixed(test_date, latest_date)
    print(f"{test_date}: {days_before:4d} days before → week {week_num:3d} → group {week_group:3d}")

print("\nWith this approach:")
print("- Most recent dates get the lowest week group numbers")
print("- Historical data gets higher week group numbers")
print("- Dates are properly grouped by actual time periods")