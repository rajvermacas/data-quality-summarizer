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
        self, aggregated_data: Dict[Tuple[str, str, str, str, int], Dict[str, Any]]
    ) -> Path:
        """
        Generate full summary CSV with exact 27-column schema.

        Args:
            aggregated_data: Dictionary with composite keys and aggregated metrics

        Returns:
            Path to generated CSV file
        """
        self._ensure_output_directory()

        # Define exact 27-column schema
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
            "business_date_latest",
            "dataset_record_count_latest",
            "filtered_record_count_latest",
            "pass_count_total",
            "fail_count_total",
            "pass_count_1m",
            "fail_count_1m",
            "pass_count_3m",
            "fail_count_3m",
            "pass_count_12m",
            "fail_count_12m",
            "fail_rate_total",
            "fail_rate_1m",
            "fail_rate_3m",
            "fail_rate_12m",
            "trend_flag",
            "last_execution_level",
        ]

        # Build rows from aggregated data
        rows = []
        for key, metrics in aggregated_data.items():
            source, tenant_id, dataset_uuid, dataset_name, rule_code = key

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
                "business_date_latest": metrics.get("business_date_latest", ""),
                "dataset_record_count_latest": metrics.get(
                    "dataset_record_count_latest", 0
                ),
                "filtered_record_count_latest": metrics.get(
                    "filtered_record_count_latest", 0
                ),
                "pass_count_total": metrics.get("pass_count_total", 0),
                "fail_count_total": metrics.get("fail_count_total", 0),
                "pass_count_1m": metrics.get("pass_count_1m", 0),
                "fail_count_1m": metrics.get("fail_count_1m", 0),
                "pass_count_3m": metrics.get("pass_count_3m", 0),
                "fail_count_3m": metrics.get("fail_count_3m", 0),
                "pass_count_12m": metrics.get("pass_count_12m", 0),
                "fail_count_12m": metrics.get("fail_count_12m", 0),
                "fail_rate_total": metrics.get("fail_rate_total", 0.0),
                "fail_rate_1m": metrics.get("fail_rate_1m", 0.0),
                "fail_rate_3m": metrics.get("fail_rate_3m", 0.0),
                "fail_rate_12m": metrics.get("fail_rate_12m", 0.0),
                "trend_flag": metrics.get("trend_flag", "="),
                "last_execution_level": metrics.get("last_execution_level", ""),
            }
            rows.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(rows, columns=columns)
        csv_path = self.output_dir / "full_summary.csv"

        try:
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"Generated CSV summary: {csv_path} ({len(df)} rows)")
            return csv_path
        except Exception as e:
            logger.error(f"Failed to write CSV file {csv_path}: {e}")
            raise

    def generate_nl_sentences(
        self, aggregated_data: Dict[Tuple[str, str, str, str, int], Dict[str, Any]]
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
            source, tenant_id, dataset_uuid, dataset_name, rule_code = key

            # Extract required fields
            business_date_latest = metrics.get("business_date_latest", date.today())
            rule_name = metrics.get("rule_name", "")
            fail_count_total = metrics.get("fail_count_total", 0)
            pass_count_total = metrics.get("pass_count_total", 0)
            fail_rate_total = metrics.get("fail_rate_total", 0.0)
            fail_rate_1m = metrics.get("fail_rate_1m", 0.0)
            fail_rate_3m = metrics.get("fail_rate_3m", 0.0)
            fail_rate_12m = metrics.get("fail_rate_12m", 0.0)
            trend_flag = metrics.get("trend_flag", "=")

            # Generate sentence following exact template
            sentence = (
                f'• On {business_date_latest}, dataset "{dataset_name}" '
                f"(source: {source}, tenant: {tenant_id}, UUID: {dataset_uuid}) "
                f'under rule "{rule_name}" [{rule_code}] '
                f"recorded {fail_count_total} failures and {pass_count_total} passes "
                f"overall "
                f"(fail-rate {fail_rate_total:.2%}; 1-month {fail_rate_1m:.2%}, "
                f"3-month {fail_rate_3m:.2%}, 12-month {fail_rate_12m:.2%}) "
                f"— trend {trend_flag}."
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
    aggregated_data: Dict[Tuple[str, str, str, str, int], Dict[str, Any]],
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
    aggregated_data: Dict[Tuple[str, str, str, str, int], Dict[str, Any]],
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
