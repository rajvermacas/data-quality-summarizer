"""
Summary generation module for data quality summarizer - Stage 4.

This module provides functionality to generate CSV summaries and natural language
artifacts from aggregated data quality results.

Key Features:
- CSV export with exact 27-column schema
- Natural language sentence generation following template
- File output to specified directory
- UTF-8 encoding support
- Memory-efficient processing
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import date


logger = logging.getLogger(__name__)


class SummaryGenerator:
    """
    Generator for summary artifacts from aggregated data quality results.

    Handles both CSV and natural language output generation following
    the exact specifications in the requirements.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the summary generator.

        Args:
            output_dir: Directory path for output files
        """
        self.output_dir = Path(output_dir)
        logger.info(f"Initialized SummaryGenerator with output_dir: {self.output_dir}")

    def _ensure_output_directory(self) -> None:
        """Ensure output directory exists, create if needed."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Output directory ensured: {self.output_dir}")
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            raise

    def generate_csv(
        self, aggregated_data: Dict[Tuple[str, str, str, str, int, int], Dict[str, Any]]
    ) -> Path:
        """
        Generate full summary CSV with exact 27-column schema.

        Args:
            aggregated_data: Dictionary with composite keys and aggregated metrics

        Returns:
            Path to generated CSV file
        """
        self._ensure_output_directory()

        # Define weekly aggregation schema
        columns = [
            "source",
            "tenant_id",
            "dataset_uuid",
            "dataset_name",
            "rule_code",
            "rule_name",
            "rule_type",
            "dimension",
            "rule_description",
            "category",
            "week_group",
            "week_start_date",
            "week_end_date",
            "business_date_latest",
            "dataset_record_count_latest",
            "filtered_record_count_latest",
            "pass_count",
            "fail_count",
            "warn_count",
            "fail_rate",
            "previous_period_fail_rate",
            "trend_flag",
            "last_execution_level",
        ]

        # Build rows from aggregated data
        rows = []
        for key, metrics in aggregated_data.items():
            source, tenant_id, dataset_uuid, dataset_name, rule_code, week_group = key

            row = {
                "source": source,
                "tenant_id": tenant_id,
                "dataset_uuid": dataset_uuid,
                "dataset_name": dataset_name,
                "rule_code": rule_code,
                "rule_name": metrics.get("rule_name", ""),
                "rule_type": metrics.get("rule_type", ""),
                "dimension": metrics.get("dimension", ""),
                "rule_description": metrics.get("rule_description", ""),
                "category": metrics.get("category", ""),
                "week_group": metrics.get("week_group", 0),
                "week_start_date": metrics.get("week_start_date", ""),
                "week_end_date": metrics.get("week_end_date", ""),
                "business_date_latest": metrics.get("business_date_latest", ""),
                "dataset_record_count_latest": metrics.get(
                    "dataset_record_count_latest", 0
                ),
                "filtered_record_count_latest": metrics.get(
                    "filtered_record_count_latest", 0
                ),
                "pass_count": metrics.get("pass_count", 0),
                "fail_count": metrics.get("fail_count", 0),
                "warn_count": metrics.get("warn_count", 0),
                "fail_rate": metrics.get("fail_rate", 0.0),
                "previous_period_fail_rate": metrics.get("previous_period_fail_rate", None),
                "trend_flag": metrics.get("trend_flag", "equal"),
                "last_execution_level": metrics.get("last_execution_level", ""),
            }
            rows.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(rows, columns=columns)
        csv_path = self.output_dir / "full_summary.csv"

        try:
            df.to_csv(csv_path, index=False, encoding="utf-8", float_format="%.2f")
            logger.info(f"Generated CSV summary: {csv_path} ({len(df)} rows)")
            return csv_path
        except Exception as e:
            logger.error(f"Failed to write CSV file {csv_path}: {e}")
            raise

    def generate_nl_sentences(
        self, aggregated_data: Dict[Tuple[str, str, str, str, int, int], Dict[str, Any]]
    ) -> Path:
        """
        Generate natural language sentences following exact template.

        Args:
            aggregated_data: Dictionary with composite keys and aggregated metrics

        Returns:
            Path to generated NL text file
        """
        self._ensure_output_directory()

        sentences = []
        for key, metrics in aggregated_data.items():
            source, tenant_id, dataset_uuid, dataset_name, rule_code, week_group = key

            # Extract required fields
            business_date_latest = metrics.get("business_date_latest", date.today())
            week_start_date = metrics.get("week_start_date", "")
            week_end_date = metrics.get("week_end_date", "")
            rule_name = metrics.get("rule_name", "")
            fail_count = metrics.get("fail_count", 0)
            warn_count = metrics.get("warn_count", 0)
            pass_count = metrics.get("pass_count", 0)
            fail_rate = metrics.get("fail_rate", 0.0)
            previous_fail_rate = metrics.get("previous_period_fail_rate", None)
            trend_flag = metrics.get("trend_flag", "equal")

            # Generate sentence following weekly template
            period_desc = f"week group {week_group} ({week_start_date} to {week_end_date})"
            trend_desc = f"trend {trend_flag}"
            if previous_fail_rate is not None:
                trend_desc += f" (vs previous period: {previous_fail_rate:.2%})"
            
            sentence = (
                f'• For {period_desc}, dataset "{dataset_name}" '
                f"(source: {source}, tenant: {tenant_id}, UUID: {dataset_uuid}) "
                f'under rule "{rule_name}" [{rule_code}] '
                f"recorded {fail_count} failures, {warn_count} warnings, and {pass_count} passes "
                f"(fail-rate {fail_rate:.2%}) "
                f"— {trend_desc}. Latest business date: {business_date_latest}."
            )
            sentences.append(sentence)

        # Write to file
        nl_path = self.output_dir / "nl_all_rows.txt"

        try:
            with open(nl_path, "w", encoding="utf-8") as f:
                if sentences:
                    for sentence in sentences:
                        f.write(sentence + "\n")
                # For empty data, write empty file (no newline)

            logger.info(
                f"Generated NL sentences: {nl_path} ({len(sentences)} sentences)"
            )
            return nl_path
        except Exception as e:
            logger.error(f"Failed to write NL file {nl_path}: {e}")
            raise


def generate_full_summary_csv(
    aggregated_data: Dict[Tuple[str, str, str, str, int, int], Dict[str, Any]],
    output_dir: str,
) -> Path:
    """
    Standalone function to generate full summary CSV.

    Args:
        aggregated_data: Dictionary with composite keys and aggregated metrics
        output_dir: Directory path for output files

    Returns:
        Path to generated CSV file
    """
    generator = SummaryGenerator(output_dir)
    return generator.generate_csv(aggregated_data)


def generate_nl_sentences(
    aggregated_data: Dict[Tuple[str, str, str, str, int, int], Dict[str, Any]],
    output_dir: str,
) -> Path:
    """
    Standalone function to generate natural language sentences.

    Args:
        aggregated_data: Dictionary with composite keys and aggregated metrics
        output_dir: Directory path for output files

    Returns:
        Path to generated NL text file
    """
    generator = SummaryGenerator(output_dir)
    return generator.generate_nl_sentences(aggregated_data)
