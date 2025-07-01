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
    Data class to store aggregated metrics for a weekly period.

    Tracks pass/fail counts for a specific N-week period and calculated
    metrics like fail rates and trend flags.
    """

    # Count fields - raw pass/fail/warning counts for this period
    pass_count: int = 0
    fail_count: int = 0
    warning_count: int = 0

    # Calculated fields - derived metrics
    fail_rate: Optional[float] = None
    trend_flag: Optional[str] = None  # up, down, or equal

    # Period information
    week_group: int = 0  # Which N-week group this represents
    business_week_start_date: Optional[date] = None
    business_week_end_date: Optional[date] = None
    business_date_latest: Optional[date] = None

    # Latest values from most recent row in this period
    dataset_record_count_latest: Optional[int] = None
    filtered_record_count_latest: Optional[int] = None
    last_execution_level: Optional[str] = None

    # Aggregated totals for this period
    dataset_record_count_total: int = 0
    filtered_record_count_total: int = 0

    # Previous period fail rate for trend calculation
    previous_period_fail_rate: Optional[float] = None

    # Store individual row data for this period
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
    Streaming aggregation engine for data quality metrics with weekly grouping.

    Processes CSV rows in chunks and maintains an accumulator with composite keys:
    (source, tenant_id, dataset_uuid, dataset_name, rule_code, week_group)

    Groups data into N-week periods and calculates metrics for each period.
    """

    def __init__(self, weeks: int = 1):
        """Initialize streaming aggregator with empty accumulator"""
        self.accumulator: Dict[Tuple[str, str, str, str, int, int], AggregationMetrics] = {}
        self.weeks = weeks  # Number of weeks to group together
        self.epsilon = 5.0  # Threshold for trend calculations (5% difference for percentage values)
        self.earliest_business_date: Optional[date] = None
        self.latest_business_date: Optional[date] = None

        logger.info(f"StreamingAggregator initialized with {weeks}-week grouping")

    def _create_composite_key(self, row_data: Dict, week_group: int) -> Tuple[str, str, str, str, int, int]:
        """
        Create composite key from row data including week group.

        Args:
            row_data: Dictionary containing row data with required fields
            week_group: Week group identifier for this row

        Returns:
            Tuple representing composite key for accumulator
        """
        return (
            str(row_data["source"]),
            str(row_data["tenant_id"]),
            str(row_data["dataset_uuid"]),
            str(row_data["dataset_name"]),
            int(row_data["rule_code"]),
            week_group,
        )

    def _calculate_week_group(self, business_date: date) -> int:
        """
        Calculate which N-week group a date belongs to.

        Args:
            business_date: Date to calculate week group for

        Returns:
            Week group number (0-based)
        """
        if self.earliest_business_date is None:
            return 0

        # Calculate total days between dates
        days_since_start = (business_date - self.earliest_business_date).days
        
        # Calculate week number (0-based) 
        week_number = days_since_start // 7
        
        # Group weeks into N-week periods
        week_group = week_number // self.weeks
        
        return week_group

    def _get_week_boundaries(self, week_group: int) -> Tuple[date, date]:
        """
        Get start and end dates for a week group.

        Args:
            week_group: Week group number

        Returns:
            Tuple of (start_date, end_date) for the week group
        """
        if self.earliest_business_date is None:
            raise ValueError("Cannot calculate week boundaries without earliest date")

        # Calculate start week for this group
        start_week = week_group * self.weeks
        
        # Calculate start date (Monday of start week)
        start_date = self.earliest_business_date + timedelta(days=start_week * 7)
        
        # Ensure start_date is a Monday
        days_since_monday = start_date.weekday()
        start_date = start_date - timedelta(days=days_since_monday)
        
        # Calculate end date (Sunday of end week)
        end_date = start_date + timedelta(days=(self.weeks * 7) - 1)
        
        return start_date, end_date

    def _parse_results_json(self, results_str: str) -> Optional[str]:
        """
        Parse JSON results string to extract pass/fail/warning status.

        Args:
            results_str: JSON string containing results data

        Returns:
            'Pass', 'Fail', or 'Warning' if successfully parsed, None otherwise
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
        # Parse business date
        business_date = self._parse_business_date(row["business_date"])
        if business_date is None:
            logger.warning(
                f"Skipping row with invalid business_date: {row['business_date']}"
            )
            return

        # Track earliest and latest business dates globally
        if self.earliest_business_date is None or business_date < self.earliest_business_date:
            self.earliest_business_date = business_date
        if self.latest_business_date is None or business_date > self.latest_business_date:
            self.latest_business_date = business_date

        # Calculate week group for this date
        week_group = self._calculate_week_group(business_date)
        
        # Create composite key with week group
        key = self._create_composite_key(row.to_dict(), week_group)

        # Parse results to determine pass/fail
        result_status = self._parse_results_json(row["results"])

        # Initialize metrics if key doesn't exist
        if key not in self.accumulator:
            start_date, end_date = self._get_week_boundaries(week_group)
            self.accumulator[key] = AggregationMetrics(
                week_group=week_group,
                business_week_start_date=start_date,
                business_week_end_date=end_date
            )
            logger.debug(f"Created new accumulator entry for key: {key}")

        metrics = self.accumulator[key]

        # Update pass/fail/warning counts
        if result_status == "Pass":
            metrics.pass_count += 1
        elif result_status == "Fail":
            metrics.fail_count += 1
        elif result_status == "Warning":
            metrics.warning_count += 1
        else:
            logger.warning(f"Unknown result status: {result_status} for key: {key}")

        # Update latest values for this period
        if metrics.business_date_latest is None or business_date >= metrics.business_date_latest:
            metrics.update_latest_values(
                business_date=business_date,
                dataset_record_count=int(row["dataset_record_count"]),
                filtered_record_count=int(row["filtered_record_count"]),
                level_of_execution=str(row["level_of_execution"]),
            )

        # Accumulate record counts for aggregated totals
        metrics.dataset_record_count_total += int(row["dataset_record_count"])
        metrics.filtered_record_count_total += int(row["filtered_record_count"])

        # Store row data for this period
        row_entry = {
            "business_date": business_date,
            "result_status": result_status,
            "dataset_record_count": int(row["dataset_record_count"]),
            "filtered_record_count": int(row["filtered_record_count"]),
            "level_of_execution": str(row["level_of_execution"]),
        }
        metrics.row_data.append(row_entry)

        logger.debug(f"Processed row for key {key}: {result_status} on {business_date}")

    def _calculate_fail_rate(self, metrics: AggregationMetrics):
        """
        Calculate fail rate for a weekly period as percentage (0-100).

        Args:
            metrics: AggregationMetrics object to update with fail rate
        """
        def safe_fail_rate(fail_count: int, pass_count: int, warning_count: int) -> float:
            """Calculate fail rate as percentage with division by zero protection"""
            total = pass_count + fail_count + warning_count
            return (fail_count / total * 100) if total > 0 else 0.0

        metrics.fail_rate = safe_fail_rate(metrics.fail_count, metrics.pass_count, metrics.warning_count)

    def _calculate_trend_flag(self, metrics: AggregationMetrics):
        """
        Calculate trend flag based on current vs previous period fail rate comparison.

        Args:
            metrics: AggregationMetrics object to update with trend flag
        """
        if metrics.fail_rate is None or metrics.previous_period_fail_rate is None:
            metrics.trend_flag = "equal"
            return

        diff = metrics.fail_rate - metrics.previous_period_fail_rate

        if diff > self.epsilon:
            metrics.trend_flag = "up"  # Degrading (higher fail rate)
        elif diff < -self.epsilon:
            metrics.trend_flag = "down"  # Improving (lower fail rate)
        else:
            metrics.trend_flag = "equal"  # Stable

    def _calculate_week_group_from_latest(self, business_date: date) -> int:
        """
        Calculate week group counting backward from latest date.
        
        Args:
            business_date: Date to calculate week group for
            
        Returns:
            Week group number (0-based, with 0 being most recent)
        """
        if self.latest_business_date is None:
            return 0
            
        # Calculate days before latest date
        days_before_latest = (self.latest_business_date - business_date).days
        
        # Calculate week number (0-based)
        week_number = days_before_latest // 7
        
        # Group weeks into N-week periods
        week_group = week_number // self.weeks
        
        return week_group

    def _recalculate_week_groups(self):
        """
        Recalculate all week groups based on latest date (counting backward).
        This ensures most recent data has lowest week group numbers.
        """
        if not self.accumulator:
            return
            
        logger.info("Recalculating week groups based on latest date")
        
        # Create new accumulator with corrected week groups
        new_accumulator = {}
        
        # Process each existing entry
        for old_key, metrics in self.accumulator.items():
            source, tenant_id, dataset_uuid, dataset_name, rule_code, old_week_group = old_key
            
            # For each row in this group, recalculate its week group
            for row_data in metrics.row_data:
                business_date = row_data["business_date"]
                
                # Calculate new week group from latest date
                new_week_group = self._calculate_week_group_from_latest(business_date)
                
                # Create new key with updated week group
                new_key = (source, tenant_id, dataset_uuid, dataset_name, rule_code, new_week_group)
                
                # Initialize new metrics if needed
                if new_key not in new_accumulator:
                    start_date, end_date = self._get_week_boundaries_from_latest(new_week_group)
                    new_accumulator[new_key] = AggregationMetrics(
                        week_group=new_week_group,
                        business_week_start_date=start_date,
                        business_week_end_date=end_date
                    )
                
                # Update metrics with row data
                new_metrics = new_accumulator[new_key]
                
                # Update counts
                if row_data["result_status"] == "Pass":
                    new_metrics.pass_count += 1
                elif row_data["result_status"] == "Fail":
                    new_metrics.fail_count += 1
                elif row_data["result_status"] == "Warning":
                    new_metrics.warning_count += 1
                
                # Update latest values
                if new_metrics.business_date_latest is None or business_date >= new_metrics.business_date_latest:
                    new_metrics.business_date_latest = business_date
                    new_metrics.dataset_record_count_latest = row_data["dataset_record_count"]
                    new_metrics.filtered_record_count_latest = row_data["filtered_record_count"]
                    new_metrics.last_execution_level = row_data["level_of_execution"]
                
                # Accumulate totals
                new_metrics.dataset_record_count_total += row_data["dataset_record_count"]
                new_metrics.filtered_record_count_total += row_data["filtered_record_count"]
                
                # Store row data
                new_metrics.row_data.append(row_data)
        
        # Replace old accumulator with new one
        self.accumulator = new_accumulator
        logger.info(f"Recalculated {len(self.accumulator)} week groups")

    def _get_week_boundaries_from_latest(self, week_group: int) -> Tuple[date, date]:
        """
        Get start and end dates for a week group counting from latest date.
        
        Args:
            week_group: Week group number (0 = most recent)
            
        Returns:
            Tuple of (start_date, end_date) for the week group
        """
        if self.latest_business_date is None:
            raise ValueError("Cannot calculate week boundaries without latest date")
            
        # Calculate end date (working backward from latest)
        # Week group 0 ends on the Sunday containing or after latest date
        latest_weekday = self.latest_business_date.weekday()
        days_to_sunday = (6 - latest_weekday) % 7
        
        # End date of week group 0
        week_0_end_date = self.latest_business_date + timedelta(days=days_to_sunday)
        
        # Calculate this group's end date
        weeks_back = week_group * self.weeks
        end_date = week_0_end_date - timedelta(weeks=weeks_back)
        
        # Start date is (N*7 - 1) days before end date
        start_date = end_date - timedelta(days=(self.weeks * 7) - 1)
        
        return start_date, end_date

    def _set_previous_period_fail_rates(self):
        """
        Set previous period fail rates for trend calculation.
        
        Groups keys by base composite key (without week_group) and sorts by week_group
        to calculate trends between consecutive periods.
        """
        # Group keys by base composite key (without week_group)
        base_key_groups = {}
        for key in self.accumulator.keys():
            source, tenant_id, dataset_uuid, dataset_name, rule_code, week_group = key
            base_key = (source, tenant_id, dataset_uuid, dataset_name, rule_code)
            
            if base_key not in base_key_groups:
                base_key_groups[base_key] = []
            base_key_groups[base_key].append(key)
        
        # For each base key group, sort by week_group and set previous fail rates
        for base_key, keys in base_key_groups.items():
            # Sort keys by week_group (ascending: 0 is most recent, higher numbers are older)
            sorted_keys = sorted(keys, key=lambda k: k[5])  # k[5] is week_group
            
            # Set previous period fail rates
            # With backward counting: lower week_group = more recent, so previous period has higher week_group
            for i in range(len(sorted_keys)):
                if i < len(sorted_keys) - 1:  # Not the last (oldest) period
                    current_key = sorted_keys[i]
                    # Previous period is the next one in sorted order (higher week_group = older)
                    prev_key = sorted_keys[i+1]
                    prev_metrics = self.accumulator[prev_key]
                    current_metrics = self.accumulator[current_key]
                    current_metrics.previous_period_fail_rate = prev_metrics.fail_rate

    def finalize_aggregation(self):
        """
        Finalize aggregation by calculating all derived metrics.

        This should be called after all rows have been processed to compute
        fail rates and trend flags for weekly periods.
        """
        if self.latest_business_date is None:
            logger.warning("No valid business dates found, cannot finalize aggregation")
            return

        logger.info(
            f"Finalizing aggregation with date range: {self.earliest_business_date} to {self.latest_business_date}"
        )

        # Recalculate week groups based on latest date (counting backward)
        self._recalculate_week_groups()

        # Calculate fail rates for each period
        for key, metrics in self.accumulator.items():
            self._calculate_fail_rate(metrics)

        # Set previous period fail rates for trend calculation
        self._set_previous_period_fail_rates()

        # Calculate trend flags
        for key, metrics in self.accumulator.items():
            self._calculate_trend_flag(metrics)

        logger.info(f"Aggregation finalized for {len(self.accumulator)} weekly periods")

    def get_accumulator_summary(self) -> Dict:
        """
        Get summary statistics about the accumulator.

        Returns:
            Dictionary with accumulator statistics
        """
        return {
            "total_keys": len(self.accumulator),
            "weeks_grouping": self.weeks,
            "earliest_business_date": (
                str(self.earliest_business_date) if self.earliest_business_date else None
            ),
            "latest_business_date": (
                str(self.latest_business_date) if self.latest_business_date else None
            ),
            "total_entries": sum(
                len(metrics.row_data) for metrics in self.accumulator.values()
            ),
        }
