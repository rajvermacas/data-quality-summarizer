"""
Streaming Aggregation Engine - Stage 3

This module implements the core streaming aggregation logic for data quality metrics.
It processes CSV rows in chunks and maintains an accumulator with composite keys for
efficient memory usage and performance.

Key Features:
- Streaming aggregation with composite keys
- Rolling time window calculations (1m, 3m, 12m)
- Pass/fail tracking from JSON results
- Trend analysis and fail rate calculations
- Memory-efficient accumulator design
"""

import json
import logging
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import pandas as pd

# Set up structured logging
logger = logging.getLogger(__name__)


@dataclass
class AggregationMetrics:
    """
    Data class to store aggregated metrics for a single composite key.

    Tracks pass/fail counts across different time windows and calculated
    metrics like fail rates and trend flags.
    """

    # Count fields - raw pass/fail counts
    pass_count_total: int = 0
    fail_count_total: int = 0
    pass_count_1m: int = 0
    fail_count_1m: int = 0
    pass_count_3m: int = 0
    fail_count_3m: int = 0
    pass_count_12m: int = 0
    fail_count_12m: int = 0

    # Calculated fields - derived metrics
    fail_rate_total: Optional[float] = None
    fail_rate_1m: Optional[float] = None
    fail_rate_3m: Optional[float] = None
    fail_rate_12m: Optional[float] = None
    trend_flag: Optional[str] = None  # up, down, or equal

    # Latest values from most recent row
    business_date_latest: Optional[date] = None
    dataset_record_count_latest: Optional[int] = None
    filtered_record_count_latest: Optional[int] = None
    last_execution_level: Optional[str] = None

    # Store individual row data for time window calculations
    row_data: List[Dict] = field(default_factory=list)

    def update_latest_values(
        self,
        business_date: date,
        dataset_record_count: int,
        filtered_record_count: int,
        level_of_execution: str,
    ):
        """Update latest values from a processed row"""
        self.business_date_latest = business_date
        self.dataset_record_count_latest = dataset_record_count
        self.filtered_record_count_latest = filtered_record_count
        self.last_execution_level = level_of_execution


class StreamingAggregator:
    """
    Streaming aggregation engine for data quality metrics.

    Processes CSV rows in chunks and maintains an accumulator with composite keys:
    (source, tenant_id, dataset_uuid, dataset_name, rule_code)

    Calculates rolling time windows (1m, 3m, 12m) and trend analysis based on
    the latest business_date found in the data.
    """

    def __init__(self):
        """Initialize streaming aggregator with empty accumulator"""
        self.accumulator: Dict[Tuple[str, str, str, str, int], AggregationMetrics] = {}
        self.latest_business_date: Optional[date] = None
        self.epsilon = 0.05  # Threshold for trend calculations

        logger.info("StreamingAggregator initialized with empty accumulator")

    def _create_composite_key(self, row_data: Dict) -> Tuple[str, str, str, str, int]:
        """
        Create composite key from row data.

        Args:
            row_data: Dictionary containing row data with required fields

        Returns:
            Tuple representing composite key for accumulator
        """
        return (
            str(row_data["source"]),
            str(row_data["tenant_id"]),
            str(row_data["dataset_uuid"]),
            str(row_data["dataset_name"]),
            int(row_data["rule_code"]),
        )

    def _parse_results_json(self, results_str: str) -> Optional[str]:
        """
        Parse JSON results string to extract pass/fail status.

        Args:
            results_str: JSON string containing results data

        Returns:
            'Pass' or 'Fail' if successfully parsed, None otherwise
        """
        try:
            results_dict = json.loads(results_str)
            return results_dict.get("result")
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse results JSON: {results_str}. Error: {e}")
            return None

    def _parse_business_date(self, date_input) -> Optional[date]:
        """
        Parse business_date input to date object.

        Args:
            date_input: ISO date string (YYYY-MM-DD) or pandas Timestamp

        Returns:
            date object if successfully parsed, None otherwise
        """
        try:
            # Handle pandas Timestamp objects
            if hasattr(date_input, "date"):
                return date_input.date()

            # Handle string dates
            if isinstance(date_input, str):
                return datetime.strptime(date_input, "%Y-%m-%d").date()

            # Handle datetime objects
            if isinstance(date_input, datetime):
                return date_input.date()

            # Unsupported type
            logger.warning(
                f"Unsupported business_date type: {type(date_input)} - {date_input}"
            )
            return None

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to parse business_date: {date_input}. Error: {e}")
            return None

    def process_row(self, row: pd.Series):
        """
        Process a single row from the CSV and update accumulator.

        Args:
            row: pandas Series containing all required columns
        """
        # Create composite key
        key = self._create_composite_key(row.to_dict())

        # Parse business date
        business_date = self._parse_business_date(row["business_date"])
        if business_date is None:
            logger.warning(
                f"Skipping row with invalid business_date: {row['business_date']}"
            )
            return

        # Track latest business date globally
        if (
            self.latest_business_date is None
            or business_date > self.latest_business_date
        ):
            self.latest_business_date = business_date

        # Parse results to determine pass/fail
        result_status = self._parse_results_json(row["results"])

        # Initialize metrics if key doesn't exist
        if key not in self.accumulator:
            self.accumulator[key] = AggregationMetrics()
            logger.debug(f"Created new accumulator entry for key: {key}")

        metrics = self.accumulator[key]

        # Update pass/fail counts
        if result_status == "Pass":
            metrics.pass_count_total += 1
        elif result_status == "Fail":
            metrics.fail_count_total += 1
        else:
            logger.warning(f"Unknown result status: {result_status} for key: {key}")

        # Update latest values
        metrics.update_latest_values(
            business_date=business_date,
            dataset_record_count=int(row["dataset_record_count"]),
            filtered_record_count=int(row["filtered_record_count"]),
            level_of_execution=str(row["level_of_execution"]),
        )

        # Store row data for time window calculations
        row_entry = {
            "business_date": business_date,
            "result_status": result_status,
            "dataset_record_count": int(row["dataset_record_count"]),
            "filtered_record_count": int(row["filtered_record_count"]),
            "level_of_execution": str(row["level_of_execution"]),
        }
        metrics.row_data.append(row_entry)

        logger.debug(f"Processed row for key {key}: {result_status} on {business_date}")

    def _calculate_rolling_windows(
        self, key: Tuple[str, str, str, str, int], reference_date: date
    ):
        """
        Calculate rolling window metrics for a specific key.

        Args:
            key: Composite key for the metrics
            reference_date: Latest business date to calculate windows from
        """
        metrics = self.accumulator[key]

        # Calculate cutoff dates for different windows
        cutoff_1m = reference_date - timedelta(days=30)
        cutoff_3m = reference_date - timedelta(days=90)
        cutoff_12m = reference_date - timedelta(days=365)

        # Reset rolling window counts
        metrics.pass_count_1m = 0
        metrics.fail_count_1m = 0
        metrics.pass_count_3m = 0
        metrics.fail_count_3m = 0
        metrics.pass_count_12m = 0
        metrics.fail_count_12m = 0

        # Count entries within each time window
        for row_entry in metrics.row_data:
            entry_date = row_entry["business_date"]
            result_status = row_entry["result_status"]

            # 12-month window (most inclusive)
            if entry_date >= cutoff_12m:
                if result_status == "Pass":
                    metrics.pass_count_12m += 1
                elif result_status == "Fail":
                    metrics.fail_count_12m += 1

            # 3-month window
            if entry_date >= cutoff_3m:
                if result_status == "Pass":
                    metrics.pass_count_3m += 1
                elif result_status == "Fail":
                    metrics.fail_count_3m += 1

            # 1-month window (most restrictive)
            if entry_date >= cutoff_1m:
                if result_status == "Pass":
                    metrics.pass_count_1m += 1
                elif result_status == "Fail":
                    metrics.fail_count_1m += 1

        logger.debug(
            f"Calculated rolling windows for key {key}: "
            f"1m({metrics.pass_count_1m}P/{metrics.fail_count_1m}F), "
            f"3m({metrics.pass_count_3m}P/{metrics.fail_count_3m}F), "
            f"12m({metrics.pass_count_12m}P/{metrics.fail_count_12m}F)"
        )

    def _calculate_fail_rates(self, metrics: AggregationMetrics):
        """
        Calculate fail rates for all time periods.

        Args:
            metrics: AggregationMetrics object to update with fail rates
        """

        def safe_fail_rate(fail_count: int, pass_count: int) -> float:
            """Calculate fail rate with division by zero protection"""
            total = pass_count + fail_count
            return fail_count / total if total > 0 else 0.0

        metrics.fail_rate_total = safe_fail_rate(
            metrics.fail_count_total, metrics.pass_count_total
        )
        metrics.fail_rate_1m = safe_fail_rate(
            metrics.fail_count_1m, metrics.pass_count_1m
        )
        metrics.fail_rate_3m = safe_fail_rate(
            metrics.fail_count_3m, metrics.pass_count_3m
        )
        metrics.fail_rate_12m = safe_fail_rate(
            metrics.fail_count_12m, metrics.pass_count_12m
        )

    def _calculate_trend_flag(self, metrics: AggregationMetrics):
        """
        Calculate trend flag based on 1m vs 3m fail rate comparison.

        Args:
            metrics: AggregationMetrics object to update with trend flag
        """
        if metrics.fail_rate_1m is None or metrics.fail_rate_3m is None:
            metrics.trend_flag = "equal"
            return

        diff = metrics.fail_rate_1m - metrics.fail_rate_3m

        if diff > self.epsilon:
            metrics.trend_flag = "up"  # Degrading (higher fail rate)
        elif diff < -self.epsilon:
            metrics.trend_flag = "down"  # Improving (lower fail rate)
        else:
            metrics.trend_flag = "equal"  # Stable

    def finalize_aggregation(self):
        """
        Finalize aggregation by calculating all derived metrics.

        This should be called after all rows have been processed to compute
        rolling windows, fail rates, and trend flags.
        """
        if self.latest_business_date is None:
            logger.warning("No valid business dates found, cannot finalize aggregation")
            return

        logger.info(
            "Finalizing aggregation with latest_business_date: "
            f"{self.latest_business_date}"
        )

        # Calculate rolling windows and derived metrics for each key
        for key in self.accumulator:
            self._calculate_rolling_windows(key, self.latest_business_date)
            self._calculate_fail_rates(self.accumulator[key])
            self._calculate_trend_flag(self.accumulator[key])

        logger.info(f"Aggregation finalized for {len(self.accumulator)} keys")

    def get_accumulator_summary(self) -> Dict:
        """
        Get summary statistics about the accumulator.

        Returns:
            Dictionary with accumulator statistics
        """
        return {
            "total_keys": len(self.accumulator),
            "latest_business_date": (
                str(self.latest_business_date) if self.latest_business_date else None
            ),
            "total_entries": sum(
                len(metrics.row_data) for metrics in self.accumulator.values()
            ),
        }
