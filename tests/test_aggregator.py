"""
Tests for streaming aggregation engine - Stage 3 TDD

Following TDD RED phase: writing failing tests first to drive implementation.
Tests cover:
- Accumulator with composite keys
- Rolling time window calculations
- Pass/fail tracking from JSON results
- Trend computation
"""

from datetime import date
import pandas as pd

# Import will fail initially (RED phase) - driving implementation
from data_quality_summarizer.aggregator import StreamingAggregator, AggregationMetrics


class TestStreamingAggregator:
    """Test suite for StreamingAggregator class"""

    def test_aggregator_initialization(self):
        """Test aggregator initializes with empty accumulator"""
        aggregator = StreamingAggregator()
        assert aggregator.accumulator == {}
        assert aggregator.latest_business_date is None

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

        expected_key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101)
        actual_key = aggregator._create_composite_key(row_data)
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

        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101)
        assert key in aggregator.accumulator

        metrics = aggregator.accumulator[key]
        assert metrics.pass_count_total == 1
        assert metrics.fail_count_total == 0

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

        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101)
        metrics = aggregator.accumulator[key]
        assert metrics.pass_count_total == 0
        assert metrics.fail_count_total == 1

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

        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101)
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

        key = ("test_system", "tenant_123", "uuid-456", "Test Dataset", 101)
        metrics = aggregator.accumulator[key]
        assert metrics.pass_count_total == 2
        assert metrics.fail_count_total == 1

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
        """Test fail rate calculations for all time periods"""
        aggregator = StreamingAggregator()

        # Create metrics with known counts
        metrics = AggregationMetrics()

        # Set test values
        metrics.pass_count_total = 7
        metrics.fail_count_total = 3
        metrics.pass_count_1m = 2
        metrics.fail_count_1m = 1
        metrics.pass_count_3m = 4
        metrics.fail_count_3m = 2
        metrics.pass_count_12m = 6
        metrics.fail_count_12m = 3

        # Calculate fail rates
        aggregator._calculate_fail_rates(metrics)

        # Total fail rate: 3/(7+3) = 0.3
        assert abs(metrics.fail_rate_total - 0.3) < 0.001

        # 1m fail rate: 1/(2+1) = 0.333...
        assert abs(metrics.fail_rate_1m - (1 / 3)) < 0.001

        # 3m fail rate: 2/(4+2) = 0.333...
        assert abs(metrics.fail_rate_3m - (2 / 6)) < 0.001

        # 12m fail rate: 3/(6+3) = 0.333...
        assert abs(metrics.fail_rate_12m - (3 / 9)) < 0.001

    def test_trend_flag_calculation(self):
        """Test trend flag calculation based on 1m vs 3m fail rates"""
        aggregator = StreamingAggregator()

        # Test case 1: Improving trend (down)
        metrics1 = AggregationMetrics()
        metrics1.fail_rate_1m = 0.2
        metrics1.fail_rate_3m = 0.4
        aggregator._calculate_trend_flag(metrics1)
        assert metrics1.trend_flag == "down"

        # Test case 2: Degrading trend (up)
        metrics2 = AggregationMetrics()
        metrics2.fail_rate_1m = 0.5
        metrics2.fail_rate_3m = 0.2
        aggregator._calculate_trend_flag(metrics2)
        assert metrics2.trend_flag == "up"

        # Test case 3: Stable trend (equal)
        metrics3 = AggregationMetrics()
        metrics3.fail_rate_1m = 0.3
        metrics3.fail_rate_3m = 0.31  # Within epsilon threshold
        aggregator._calculate_trend_flag(metrics3)
        assert metrics3.trend_flag == "equal"

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
        assert metrics.pass_count_total == 0
        assert metrics.fail_count_total == 0
        assert metrics.pass_count_1m == 0
        assert metrics.fail_count_1m == 0
        assert metrics.pass_count_3m == 0
        assert metrics.fail_count_3m == 0
        assert metrics.pass_count_12m == 0
        assert metrics.fail_count_12m == 0

        # Calculated fields should start as None
        assert metrics.fail_rate_total is None
        assert metrics.fail_rate_1m is None
        assert metrics.fail_rate_3m is None
        assert metrics.fail_rate_12m is None
        assert metrics.trend_flag is None
        assert metrics.business_date_latest is None
        assert metrics.dataset_record_count_latest is None
        assert metrics.filtered_record_count_latest is None
        assert metrics.last_execution_level is None

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
