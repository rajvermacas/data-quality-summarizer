"""
Tests for streaming aggregation engine - Stage 3 TDD

Following TDD RED phase: writing failing tests first to drive implementation.
Tests cover:
- Accumulator with composite keys
- Rolling time window calculations
- Pass/fail tracking from JSON results
- Trend computation
"""

from datetime import date, timedelta
import pandas as pd
import pytest

# Import will fail initially (RED phase) - driving implementation
from data_quality_summarizer.aggregator import StreamingAggregator, AggregationMetrics


class TestStreamingAggregator:
    """Test suite for StreamingAggregator class"""

    def test_aggregator_initialization(self):
        """Test aggregator initializes with empty accumulator"""
        aggregator = StreamingAggregator()
        assert aggregator.accumulator == {}
        assert aggregator.latest_business_date is None
        assert aggregator.earliest_business_date is None
        assert aggregator.weeks == 1  # Default value

    def test_composite_key_generation(self):
        """Test composite key creation from row data"""
        aggregator = StreamingAggregator()

        row_data = {
            "source": "test_system",
            "tenant_id": "tenant_123",
            "dataset_uuid": "uuid-456",
            "dataset_name": "Test Dataset",
            "rule_code": 101,
        }

        week_group = 0
        expected_key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 0)
        actual_key = aggregator._create_composite_key(row_data, week_group)
        assert actual_key == expected_key

    def test_process_single_row_creates_entry(self):
        """Test processing first row creates accumulator entry"""
        aggregator = StreamingAggregator()

        row = pd.Series(
            {
                "source": "test_system",
                "tenant_id": "tenant_123",
                "dataset_uuid": "uuid-456",
                "dataset_name": "Test Dataset",
                "rule_code": 101,
                "business_date": "2024-01-15",
                "results": '{"result": "Pass"}',
                "dataset_record_count": 1000,
                "filtered_record_count": 950,
                "level_of_execution": "DATASET",
            }
        )

        aggregator.process_row(row)

        # With new weekly structure, key includes week_group
        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 0)
        assert key in aggregator.accumulator

        metrics = aggregator.accumulator[key]
        assert metrics.pass_count == 1
        assert metrics.fail_count == 0
        assert metrics.warning_count == 0
        assert metrics.week_group == 0

    def test_process_row_with_fail_result(self):
        """Test processing row with fail result"""
        aggregator = StreamingAggregator()

        row = pd.Series(
            {
                "source": "test_system",
                "tenant_id": "tenant_123",
                "dataset_uuid": "uuid-456",
                "dataset_name": "Test Dataset",
                "rule_code": 101,
                "business_date": "2024-01-15",
                "results": '{"result": "Fail"}',
                "dataset_record_count": 1000,
                "filtered_record_count": 950,
                "level_of_execution": "DATASET",
            }
        )

        aggregator.process_row(row)

        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 0)
        metrics = aggregator.accumulator[key]
        assert metrics.pass_count == 0
        assert metrics.fail_count == 1
        assert metrics.warning_count == 0

    def test_process_row_with_warning_result(self):
        """Test processing row with warning result"""
        aggregator = StreamingAggregator()

        row = pd.Series(
            {
                "source": "test_system",
                "tenant_id": "tenant_123",
                "dataset_uuid": "uuid-456",
                "dataset_name": "Test Dataset",
                "rule_code": 101,
                "business_date": "2024-01-15",
                "results": '{"result": "Warning"}',
                "dataset_record_count": 1000,
                "filtered_record_count": 950,
                "level_of_execution": "DATASET",
            }
        )

        aggregator.process_row(row)

        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 0)
        metrics = aggregator.accumulator[key]
        assert metrics.pass_count == 0
        assert metrics.fail_count == 0
        assert metrics.warning_count == 1

    def test_malformed_json_results_handling(self):
        """Test graceful handling of malformed JSON in results"""
        aggregator = StreamingAggregator()

        row = pd.Series(
            {
                "source": "test_system",
                "tenant_id": "tenant_123",
                "dataset_uuid": "uuid-456",
                "dataset_name": "Test Dataset",
                "rule_code": 101,
                "business_date": "2024-01-15",
                "results": '{"result": invalid_json}',  # Malformed JSON
                "dataset_record_count": 1000,
                "filtered_record_count": 950,
                "level_of_execution": "DATASET",
            }
        )

        # Should not raise exception, should log warning
        aggregator.process_row(row)

        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 0)
        # Should still create entry but with neutral counts
        assert key in aggregator.accumulator

    def test_multiple_rows_same_key_accumulation(self):
        """Test multiple rows with same key accumulate correctly"""
        aggregator = StreamingAggregator()

        base_row = {
            "source": "test_system",
            "tenant_id": "tenant_123",
            "dataset_uuid": "uuid-456",
            "dataset_name": "Test Dataset",
            "rule_code": 101,
            "dataset_record_count": 1000,
            "filtered_record_count": 950,
            "level_of_execution": "DATASET",
        }

        # Add three rows: Pass, Fail, Pass
        rows = [
            {
                **base_row,
                "business_date": "2024-01-15",
                "results": '{"result": "Pass"}',
            },
            {
                **base_row,
                "business_date": "2024-01-16",
                "results": '{"result": "Fail"}',
            },
            {
                **base_row,
                "business_date": "2024-01-17",
                "results": '{"result": "Pass"}',
            },
        ]

        for row_data in rows:
            row = pd.Series(row_data)
            aggregator.process_row(row)

        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 0)
        metrics = aggregator.accumulator[key]
        assert metrics.pass_count == 2
        assert metrics.fail_count == 1

    def test_latest_business_date_tracking(self):
        """Test latest business_date is tracked correctly"""
        aggregator = StreamingAggregator()

        dates = ["2024-01-15", "2024-01-10", "2024-01-20", "2024-01-17"]
        base_row = {
            "source": "test_system",
            "tenant_id": "tenant_123",
            "dataset_uuid": "uuid-456",
            "dataset_name": "Test Dataset",
            "rule_code": 101,
            "results": '{"result": "Pass"}',
            "dataset_record_count": 1000,
            "filtered_record_count": 950,
            "level_of_execution": "DATASET",
        }

        for business_date in dates:
            row = pd.Series({**base_row, "business_date": business_date})
            aggregator.process_row(row)

        # Latest date should be 2024-01-20
        assert aggregator.latest_business_date == date(2024, 1, 20)

    @pytest.mark.skip(reason="Obsolete functionality - system now uses weekly grouping instead of rolling windows")
    def test_rolling_window_calculations(self):
        """Test rolling window calculations for 1m, 3m, 12m"""
        aggregator = StreamingAggregator()

        # Set latest date to 2024-01-31
        latest_date = date(2024, 1, 31)

        # Create test data spanning different time windows
        test_dates_and_results = [
            ("2024-01-31", "Pass"),  # Today (latest)
            ("2024-01-15", "Fail"),  # 16 days ago (within 1m)
            ("2024-01-01", "Pass"),  # 30 days ago (within 1m)
            ("2023-12-15", "Fail"),  # ~47 days ago (within 3m, outside 1m)
            ("2023-11-01", "Pass"),  # ~91 days ago (within 12m, outside 3m)
            ("2023-01-01", "Fail"),  # 365 days ago (outside 12m)
        ]

        base_row = {
            "source": "test_system",
            "tenant_id": "tenant_123",
            "dataset_uuid": "uuid-456",
            "dataset_name": "Test Dataset",
            "rule_code": 101,
            "dataset_record_count": 1000,
            "filtered_record_count": 950,
            "level_of_execution": "DATASET",
        }

        for business_date, result in test_dates_and_results:
            row = pd.Series(
                {
                    **base_row,
                    "business_date": business_date,
                    "results": f'{{"result": "{result}"}}',
                }
            )
            aggregator.process_row(row)

        # Calculate rolling windows with latest_date = 2024-01-31
        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101)
        metrics = aggregator.accumulator[key]

        # Force calculation of rolling windows
        aggregator._calculate_rolling_windows(key, latest_date)

        # 1-month window (30 days): 2024-01-31, 2024-01-15, 2024-01-01
        assert metrics.pass_count_1m == 2  # 2024-01-31, 2024-01-01
        assert metrics.fail_count_1m == 1  # 2024-01-15

        # 3-month window (90 days): includes 2023-12-15 as well
        assert metrics.pass_count_3m == 2  # Same as 1m
        assert metrics.fail_count_3m == 2  # 2024-01-15, 2023-12-15

        # 12-month window (365 days): includes 2023-11-01 as well
        assert metrics.pass_count_12m == 3  # Previous + 2023-11-01
        assert metrics.fail_count_12m == 2  # Same as 3m

    def test_fail_rate_calculations(self):
        """Test fail rate calculation as percentage"""
        aggregator = StreamingAggregator()

        # Create metrics with known counts
        metrics = AggregationMetrics()

        # Test case 1: Basic fail rate calculation
        metrics.pass_count = 7
        metrics.fail_count = 3

        # Calculate fail rate
        aggregator._calculate_fail_rate(metrics)

        # Fail rate: 3/(7+3) = 30%
        assert abs(metrics.fail_rate - 30.0) < 0.001

        # Test case 2: Zero total should give 0% fail rate
        metrics2 = AggregationMetrics()
        metrics2.pass_count = 0
        metrics2.fail_count = 0
        aggregator._calculate_fail_rate(metrics2)
        assert metrics2.fail_rate == 0.0

        # Test case 3: All failures should give 100% fail rate
        metrics3 = AggregationMetrics()
        metrics3.pass_count = 0
        metrics3.fail_count = 5
        aggregator._calculate_fail_rate(metrics3)
        assert abs(metrics3.fail_rate - 100.0) < 0.001

        # Test case 4: With warnings in total - fail rate should be lower
        metrics4 = AggregationMetrics()
        metrics4.pass_count = 7
        metrics4.fail_count = 3
        metrics4.warning_count = 2
        aggregator._calculate_fail_rate(metrics4)
        # Fail rate: 3/(7+3+2) = 25%
        assert abs(metrics4.fail_rate - 25.0) < 0.001

        # Test case 5: Only warnings should give 0% fail rate
        metrics5 = AggregationMetrics()
        metrics5.pass_count = 0
        metrics5.fail_count = 0
        metrics5.warning_count = 5
        aggregator._calculate_fail_rate(metrics5)
        assert metrics5.fail_rate == 0.0

        # Test case 6: Mix with high warning count
        metrics6 = AggregationMetrics()
        metrics6.pass_count = 10
        metrics6.fail_count = 2
        metrics6.warning_count = 8
        aggregator._calculate_fail_rate(metrics6)
        # Fail rate: 2/(10+2+8) = 10%
        assert abs(metrics6.fail_rate - 10.0) < 0.001

    def test_trend_flag_calculation(self):
        """Test trend flag calculation based on current vs previous period fail rates"""
        aggregator = StreamingAggregator()

        # Test case 1: Improving trend (down)
        metrics1 = AggregationMetrics()
        metrics1.fail_rate = 20.0
        metrics1.previous_period_fail_rate = 40.0
        aggregator._calculate_trend_flag(metrics1)
        assert metrics1.trend_flag == "down"

        # Test case 2: Degrading trend (up)
        metrics2 = AggregationMetrics()
        metrics2.fail_rate = 50.0
        metrics2.previous_period_fail_rate = 20.0
        aggregator._calculate_trend_flag(metrics2)
        assert metrics2.trend_flag == "up"

        # Test case 3: Stable trend (equal)
        metrics3 = AggregationMetrics()
        metrics3.fail_rate = 30.0
        metrics3.previous_period_fail_rate = 31.0  # Within epsilon threshold (5%)
        aggregator._calculate_trend_flag(metrics3)
        assert metrics3.trend_flag == "equal"

        # Test case 4: No previous period data
        metrics4 = AggregationMetrics()
        metrics4.fail_rate = 30.0
        metrics4.previous_period_fail_rate = None
        aggregator._calculate_trend_flag(metrics4)
        assert metrics4.trend_flag == "equal"

    @pytest.mark.skip(reason="Obsolete functionality - system now uses weekly grouping instead of rolling windows")
    def test_finalize_aggregation(self):
        """Test finalization of aggregation with all calculations"""
        aggregator = StreamingAggregator()

        # Add some test data
        test_data = [
            ("2024-01-31", "Pass"),
            ("2024-01-15", "Fail"),
            ("2024-01-01", "Pass"),
            ("2023-12-01", "Fail"),
        ]

        base_row = {
            "source": "test_system",
            "tenant_id": "tenant_123",
            "dataset_uuid": "uuid-456",
            "dataset_name": "Test Dataset",
            "rule_code": 101,
            "dataset_record_count": 1000,
            "filtered_record_count": 950,
            "level_of_execution": "DATASET",
        }

        for business_date, result in test_data:
            row = pd.Series(
                {
                    **base_row,
                    "business_date": business_date,
                    "results": f'{{"result": "{result}"}}',
                }
            )
            aggregator.process_row(row)

        # Finalize aggregation
        aggregator.finalize_aggregation()

        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101)
        metrics = aggregator.accumulator[key]

        # Check that all calculations are completed
        assert metrics.fail_rate_total is not None
        assert metrics.fail_rate_1m is not None
        assert metrics.fail_rate_3m is not None
        assert metrics.fail_rate_12m is not None
        assert metrics.trend_flag in ["up", "down", "equal"]
        assert metrics.business_date_latest is not None


class TestAggregationMetrics:
    """Test suite for AggregationMetrics data class"""

    def test_metrics_initialization(self):
        """Test metrics object initializes with correct defaults"""
        metrics = AggregationMetrics()

        # Count fields should start at 0
        assert metrics.pass_count == 0
        assert metrics.fail_count == 0
        assert metrics.warning_count == 0

        # Calculated fields should start as None
        assert metrics.fail_rate is None
        assert metrics.trend_flag is None
        assert metrics.business_date_latest is None
        assert metrics.dataset_record_count_latest is None
        assert metrics.filtered_record_count_latest is None
        assert metrics.last_execution_level is None
        
        # New aggregated total fields should start at 0
        assert metrics.dataset_record_count_total == 0
        assert metrics.filtered_record_count_total == 0

    def test_metrics_update_latest_values(self):
        """Test updating latest values in metrics"""
        metrics = AggregationMetrics()

        test_date = date(2024, 1, 15)
        metrics.update_latest_values(
            business_date=test_date,
            dataset_record_count=1000,
            filtered_record_count=950,
            level_of_execution="DATASET",
        )

        assert metrics.business_date_latest == test_date
        assert metrics.dataset_record_count_latest == 1000
        assert metrics.filtered_record_count_latest == 950
        assert metrics.last_execution_level == "DATASET"


# Performance and memory tests
class TestStreamingAggregatorPerformance:
    """Performance-focused tests for streaming aggregator"""

    def test_memory_usage_stays_under_limit(self):
        """Test memory usage stays under 50MB for accumulator"""
        import sys

        aggregator = StreamingAggregator()

        # Simulate processing many rows to test memory usage
        # ~100 unique keys with ~1000 entries each should stay under 50MB
        for source_idx in range(10):  # 10 sources
            for tenant_idx in range(10):  # 10 tenants per source
                base_row = {
                    "source": f"source_{source_idx}",
                    "tenant_id": f"tenant_{tenant_idx}",
                    "dataset_uuid": f"uuid_{source_idx}_{tenant_idx}",
                    "dataset_name": f"Dataset {source_idx}-{tenant_idx}",
                    "rule_code": 101,
                    "business_date": "2024-01-15",
                    "results": '{"result": "Pass"}',
                    "dataset_record_count": 1000,
                    "filtered_record_count": 950,
                    "level_of_execution": "DATASET",
                }

                row = pd.Series(base_row)
                aggregator.process_row(row)

        # Should have 100 keys total (10 sources × 10 tenants)
        assert len(aggregator.accumulator) == 100

        # Rough memory check - accumulator should be reasonable size
        accumulator_size = sys.getsizeof(aggregator.accumulator)
        assert accumulator_size < 50 * 1024 * 1024  # 50MB


class TestWeeklyGrouping:
    """Test suite for weekly grouping functionality"""

    def test_single_week_grouping(self):
        """Test 1-week grouping (default behavior)"""
        aggregator = StreamingAggregator(weeks=1)
        
        # Create data for consecutive days in the same week
        base_date = date(2024, 1, 15)  # Monday
        for day_offset in range(3):  # Mon, Tue, Wed
            row = pd.Series({
                "source": "test_system",
                "tenant_id": "tenant_123",
                "dataset_uuid": "uuid-456",
                "dataset_name": "Test Dataset",
                "rule_code": 101,
                "business_date": str(base_date + timedelta(days=day_offset)),
                "results": '{"result": "Pass"}',
                "dataset_record_count": 1000,
                "filtered_record_count": 950,
                "level_of_execution": "DATASET",
            })
            aggregator.process_row(row)
        
        # Should have 1 key for week group 0
        assert len(aggregator.accumulator) == 1
        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 0)
        assert key in aggregator.accumulator
        
        metrics = aggregator.accumulator[key]
        assert metrics.pass_count == 3
        assert metrics.week_group == 0

    def test_two_week_grouping(self):
        """Test 2-week grouping functionality"""
        aggregator = StreamingAggregator(weeks=2)
        
        # Create data spanning 3 weeks
        base_date = date(2024, 1, 15)  # Monday of week 1
        for week in range(3):
            for day in range(2):  # 2 days per week
                row = pd.Series({
                    "source": "test_system",
                    "tenant_id": "tenant_123", 
                    "dataset_uuid": "uuid-456",
                    "dataset_name": "Test Dataset",
                    "rule_code": 101,
                    "business_date": str(base_date + timedelta(weeks=week, days=day)),
                    "results": '{"result": "Pass"}',
                    "dataset_record_count": 1000,
                    "filtered_record_count": 950,
                    "level_of_execution": "DATASET",
                })
                aggregator.process_row(row)
        
        # Should have 2 keys: 
        # - Week group 0 (weeks 0-1): 4 passes
        # - Week group 1 (week 2): 2 passes
        assert len(aggregator.accumulator) == 2
        
        key_group_0 = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 0)
        key_group_1 = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 1)
        
        assert key_group_0 in aggregator.accumulator
        assert key_group_1 in aggregator.accumulator
        
        assert aggregator.accumulator[key_group_0].pass_count == 4
        assert aggregator.accumulator[key_group_1].pass_count == 2

    def test_trend_calculation_between_periods(self):
        """Test trend calculation between weekly periods"""
        aggregator = StreamingAggregator(weeks=1)
        
        # Week 1: 8 passes, 2 fails (20% fail rate)
        base_date = date(2024, 1, 15)
        for i in range(10):
            result = "Fail" if i < 2 else "Pass"
            row = pd.Series({
                "source": "test_system",
                "tenant_id": "tenant_123",
                "dataset_uuid": "uuid-456", 
                "dataset_name": "Test Dataset",
                "rule_code": 101,
                "business_date": str(base_date),
                "results": f'{{"result": "{result}"}}',
                "dataset_record_count": 1000,
                "filtered_record_count": 950,
                "level_of_execution": "DATASET",
            })
            aggregator.process_row(row)
        
        # Week 2: 9 passes, 1 fail (10% fail rate) - improving trend
        week2_date = base_date + timedelta(weeks=1)
        for i in range(10):
            result = "Fail" if i < 1 else "Pass"
            row = pd.Series({
                "source": "test_system",
                "tenant_id": "tenant_123",
                "dataset_uuid": "uuid-456",
                "dataset_name": "Test Dataset", 
                "rule_code": 101,
                "business_date": str(week2_date),
                "results": f'{{"result": "{result}"}}',
                "dataset_record_count": 1000,
                "filtered_record_count": 950,
                "level_of_execution": "DATASET",
            })
            aggregator.process_row(row)
        
        # Finalize to calculate trends
        aggregator.finalize_aggregation()
        
        # Check week 1 and week 2 metrics
        key_week_0 = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 0)
        key_week_1 = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 1)
        
        week_0_metrics = aggregator.accumulator[key_week_0]
        week_1_metrics = aggregator.accumulator[key_week_1]
        
        # Week 0: 20% fail rate, no previous period
        assert abs(week_0_metrics.fail_rate - 20.0) < 0.01
        assert week_0_metrics.previous_period_fail_rate is None
        assert week_0_metrics.trend_flag == "equal"
        
        # Week 1: 10% fail rate, previous was 20%, so trend should be "down" (improving)
        assert abs(week_1_metrics.fail_rate - 10.0) < 0.01
        assert abs(week_1_metrics.previous_period_fail_rate - 20.0) < 0.01
        assert week_1_metrics.trend_flag == "down"

    def test_multi_dataset_weekly_grouping(self):
        """Test weekly grouping with multiple datasets"""
        aggregator = StreamingAggregator(weeks=1)
        
        base_date = date(2024, 1, 15)
        
        # Create data for 2 different datasets across 2 weeks
        for week in range(2):
            for dataset_idx in range(2):
                row = pd.Series({
                    "source": "test_system",
                    "tenant_id": "tenant_123",
                    "dataset_uuid": f"uuid-{dataset_idx}",
                    "dataset_name": f"Dataset {dataset_idx}",
                    "rule_code": 101,
                    "business_date": str(base_date + timedelta(weeks=week)),
                    "results": '{"result": "Pass"}',
                    "dataset_record_count": 1000,
                    "filtered_record_count": 950,
                    "level_of_execution": "DATASET",
                })
                aggregator.process_row(row)
        
        # Should have 4 keys total (2 datasets × 2 weeks)
        assert len(aggregator.accumulator) == 4
        
        # Check that each dataset-week combination exists
        for week in range(2):
            for dataset_idx in range(2):
                key = ("test_system", "tenant_123", f"uuid-{dataset_idx}", f"Dataset {dataset_idx}", 101, week)
                assert key in aggregator.accumulator
                assert aggregator.accumulator[key].pass_count == 1

    def test_week_boundary_durations(self):
        """Test that week group boundaries have correct durations regardless of group number"""
        from datetime import timedelta
        
        # Test different week groupings
        for weeks in [1, 2, 4]:
            aggregator = StreamingAggregator(weeks=weeks)
            
            # Create data spanning multiple week groups
            base_date = date(2024, 1, 15)  # Monday
            
            # Add data across 6 weeks to create multiple groups
            for week_offset in range(6):
                for day in range(2):  # 2 entries per week
                    row = pd.Series({
                        "source": "test_system",
                        "tenant_id": "tenant_123",
                        "dataset_uuid": "uuid-456",
                        "dataset_name": "Test Dataset",
                        "rule_code": 101,
                        "business_date": str(base_date + timedelta(weeks=week_offset, days=day)),
                        "results": '{"result": "Pass"}',
                        "dataset_record_count": 1000,
                        "filtered_record_count": 950,
                        "level_of_execution": "DATASET",
                    })
                    aggregator.process_row(row)
            
            # Verify week boundaries for each group
            expected_groups = 6 // weeks if 6 % weeks == 0 else (6 // weeks) + 1
            
            for week_group in range(expected_groups):
                start_date, end_date = aggregator._get_week_boundaries(week_group)
                
                # Calculate actual duration
                duration_days = (end_date - start_date).days + 1  # +1 for inclusive
                expected_duration = weeks * 7
                
                assert duration_days == expected_duration, (
                    f"Week group {week_group} with {weeks}-week grouping has "
                    f"{duration_days} days, expected {expected_duration} days. "
                    f"Start: {start_date}, End: {end_date}"
                )
                
                # Verify start date is always Monday
                assert start_date.weekday() == 0, f"Start date {start_date} is not Monday"
                
                # Verify end date is always Sunday
                assert end_date.weekday() == 6, f"End date {end_date} is not Sunday"


class TestRecordCountAggregation:
    """Test suite for aggregated record count functionality"""

    def test_dataset_record_count_aggregation(self):
        """Test that dataset_record_count values are aggregated correctly"""
        aggregator = StreamingAggregator()
        
        # Process multiple rows with different record counts
        base_row = {
            "source": "test_system",
            "tenant_id": "tenant_123",
            "dataset_uuid": "uuid-456",
            "dataset_name": "Test Dataset",
            "rule_code": 101,
            "business_date": "2024-01-15",
            "results": '{"result": "Pass"}',
            "level_of_execution": "DATASET",
        }
        
        # Process 3 rows with different dataset_record_count values
        record_counts = [1000, 1500, 2000]
        for count in record_counts:
            row = pd.Series({
                **base_row,
                "dataset_record_count": count,
                "filtered_record_count": count - 50,  # Always 50 less
            })
            aggregator.process_row(row)
        
        # Verify aggregation
        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 0)
        metrics = aggregator.accumulator[key]
        
        # Total should be sum of all record counts
        expected_total = sum(record_counts)  # 1000 + 1500 + 2000 = 4500
        assert metrics.dataset_record_count_total == expected_total
        
        # Latest should be the last processed value
        assert metrics.dataset_record_count_latest == 2000
        
        # Filtered record count should also be aggregated
        expected_filtered_total = sum(count - 50 for count in record_counts)  # 950 + 1450 + 1950 = 4350
        assert metrics.filtered_record_count_total == expected_filtered_total
        assert metrics.filtered_record_count_latest == 1950

    def test_record_count_aggregation_across_multiple_weeks(self):
        """Test record count aggregation with weekly grouping"""
        aggregator = StreamingAggregator(weeks=1)
        
        base_row = {
            "source": "test_system", 
            "tenant_id": "tenant_123",
            "dataset_uuid": "uuid-456",
            "dataset_name": "Test Dataset",
            "rule_code": 101,
            "results": '{"result": "Pass"}',
            "level_of_execution": "DATASET",
        }
        
        # Week 1: 2 rows with different counts
        week1_counts = [1000, 1200]
        for count in week1_counts:
            row = pd.Series({
                **base_row,
                "business_date": "2024-01-15",  # Week 1
                "dataset_record_count": count,
                "filtered_record_count": count - 100,
            })
            aggregator.process_row(row)
        
        # Week 2: 3 rows with different counts  
        week2_counts = [1500, 1800, 2000]
        for count in week2_counts:
            row = pd.Series({
                **base_row,
                "business_date": "2024-01-22",  # Week 2
                "dataset_record_count": count,
                "filtered_record_count": count - 100,
            })
            aggregator.process_row(row)
        
        # Verify week 1 aggregation
        key_week1 = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 0)
        metrics_week1 = aggregator.accumulator[key_week1]
        
        assert metrics_week1.dataset_record_count_total == sum(week1_counts)  # 2200
        assert metrics_week1.filtered_record_count_total == sum(count - 100 for count in week1_counts)  # 2000
        assert metrics_week1.dataset_record_count_latest == 1200  # Last value in week 1
        
        # Verify week 2 aggregation
        key_week2 = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 1)
        metrics_week2 = aggregator.accumulator[key_week2]
        
        assert metrics_week2.dataset_record_count_total == sum(week2_counts)  # 5300
        assert metrics_week2.filtered_record_count_total == sum(count - 100 for count in week2_counts)  # 5000
        assert metrics_week2.dataset_record_count_latest == 2000  # Last value in week 2

    def test_zero_record_counts_handled_correctly(self):
        """Test that zero record counts are handled correctly in aggregation"""
        aggregator = StreamingAggregator()
        
        # Process rows with zero and non-zero counts
        test_counts = [0, 1000, 0, 500]
        
        for i, count in enumerate(test_counts):
            row = pd.Series({
                "source": "test_system",
                "tenant_id": "tenant_123", 
                "dataset_uuid": "uuid-456",
                "dataset_name": "Test Dataset",
                "rule_code": 101,
                "business_date": f"2024-01-{15 + i}",  # Different dates
                "results": '{"result": "Pass"}',
                "dataset_record_count": count,
                "filtered_record_count": count,
                "level_of_execution": "DATASET",
            })
            aggregator.process_row(row)
        
        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101, 0)
        metrics = aggregator.accumulator[key]
        
        # Total should include zeros: 0 + 1000 + 0 + 500 = 1500
        assert metrics.dataset_record_count_total == 1500
        assert metrics.filtered_record_count_total == 1500
        
        # Latest should be the last processed value (500)
        assert metrics.dataset_record_count_latest == 500
        assert metrics.filtered_record_count_latest == 500
