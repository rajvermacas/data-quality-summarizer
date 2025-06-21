"""
Test suite for the summarizer module - Stage 4 TDD.

This module tests the CSV and natural language artifact generation functionality.
Following TDD: RED → GREEN → REFACTOR cycle.
"""

import pytest
import pandas as pd
import tempfile
from datetime import date
from pathlib import Path

# Import will fail initially - this is expected for RED phase
from data_quality_summarizer.summarizer import (
    SummaryGenerator,
    generate_full_summary_csv,
    generate_nl_sentences,
)


class TestSummaryGenerator:
    """Test cases for the SummaryGenerator class."""

    @pytest.fixture
    def sample_aggregated_data(self):
        """Sample aggregated data for testing."""
        return {
            ("SRC1", "tenant1", "uuid1", "Dataset A", 101): {
                "rule_name": "ROW_COUNT",
                "rule_type": "DATASET",
                "dimension": "Correctness",
                "rule_description": "Row count validation",
                "category": 1,
                "business_date_latest": date(2024, 1, 15),
                "dataset_record_count_latest": 50000,
                "filtered_record_count_latest": 48000,
                "pass_count_total": 100,
                "fail_count_total": 5,
                "pass_count_1m": 30,
                "fail_count_1m": 2,
                "pass_count_3m": 80,
                "fail_count_3m": 4,
                "pass_count_12m": 100,
                "fail_count_12m": 5,
                "fail_rate_total": 0.0476,
                "fail_rate_1m": 0.0625,
                "fail_rate_3m": 0.0476,
                "fail_rate_12m": 0.0476,
                "trend_flag": "up",
                "last_execution_level": "DATASET",
            }
        }

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_summary_generator_initialization(self):
        """Test SummaryGenerator initialization."""
        generator = SummaryGenerator(output_dir="/tmp/test")
        assert generator.output_dir == Path("/tmp/test")

    def test_create_output_directory(self, temp_output_dir):
        """Test output directory creation."""
        test_dir = Path(temp_output_dir) / "new_artifacts"
        generator = SummaryGenerator(output_dir=str(test_dir))
        generator._ensure_output_directory()
        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_generate_csv_with_exact_schema(
        self, sample_aggregated_data, temp_output_dir
    ):
        """Test CSV generation with exact 27-column schema."""
        generator = SummaryGenerator(output_dir=temp_output_dir)
        csv_path = generator.generate_csv(sample_aggregated_data)

        # Verify file creation
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"

        # Load and verify schema
        df = pd.read_csv(csv_path)
        expected_columns = [
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

        assert len(df.columns) == 27, f"Expected 27 columns, got {len(df.columns)}"
        assert list(df.columns) == expected_columns
        assert len(df) == 1  # One row from sample data

    def test_csv_data_accuracy(self, sample_aggregated_data, temp_output_dir):
        """Test CSV data accuracy and types."""
        generator = SummaryGenerator(output_dir=temp_output_dir)
        csv_path = generator.generate_csv(sample_aggregated_data)

        df = pd.read_csv(csv_path)
        row = df.iloc[0]

        # Verify key data
        assert row["source"] == "SRC1"
        assert row["tenant_id"] == "tenant1"
        assert row["dataset_uuid"] == "uuid1"
        assert row["dataset_name"] == "Dataset A"
        assert row["rule_code"] == 101
        assert row["rule_name"] == "ROW_COUNT"
        assert row["fail_count_total"] == 5
        assert row["pass_count_total"] == 100
        assert abs(row["fail_rate_total"] - 0.0476) < 0.0001
        assert row["trend_flag"] == "up"

    def test_generate_nl_sentences(self, sample_aggregated_data, temp_output_dir):
        """Test natural language sentence generation."""
        generator = SummaryGenerator(output_dir=temp_output_dir)
        nl_path = generator.generate_nl_sentences(sample_aggregated_data)

        # Verify file creation
        assert nl_path.exists()
        assert nl_path.suffix == ".txt"

        # Verify content
        with open(nl_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Check template structure
        assert content.startswith("•")
        assert "Dataset A" in content
        assert "ROW_COUNT" in content
        assert "[101]" in content
        assert "5 failures" in content
        assert "100 passes" in content
        assert "fail-rate 4.76%" in content
        assert "trend up" in content

    def test_nl_sentence_template_exact_format(
        self, sample_aggregated_data, temp_output_dir
    ):
        """Test NL sentence follows exact template format."""
        generator = SummaryGenerator(output_dir=temp_output_dir)
        nl_path = generator.generate_nl_sentences(sample_aggregated_data)

        with open(nl_path, "r", encoding="utf-8") as f:
            sentence = f.read().strip()

        # Expected template format
        expected_parts = [
            "• On 2024-01-15",
            'dataset "Dataset A"',
            "(source: SRC1, tenant: tenant1, UUID: uuid1)",
            'under rule "ROW_COUNT" [101]',
            "recorded 5 failures and 100 passes overall",
            "(fail-rate 4.76%; 1-month 6.25%, 3-month 4.76%, 12-month 4.76%)",
            "— trend up",
        ]

        for part in expected_parts:
            assert part in sentence, f"Missing part: {part}"

    def test_multiple_rows_handling(self, temp_output_dir):
        """Test handling multiple aggregated rows."""
        data = {
            ("SRC1", "tenant1", "uuid1", "Dataset A", 101): {
                "rule_name": "ROW_COUNT",
                "rule_type": "DATASET",
                "dimension": "Correctness",
                "rule_description": "Row count validation",
                "category": 1,
                "business_date_latest": date(2024, 1, 15),
                "dataset_record_count_latest": 50000,
                "filtered_record_count_latest": 48000,
                "pass_count_total": 100,
                "fail_count_total": 5,
                "pass_count_1m": 30,
                "fail_count_1m": 2,
                "pass_count_3m": 80,
                "fail_count_3m": 4,
                "pass_count_12m": 100,
                "fail_count_12m": 5,
                "fail_rate_total": 0.0476,
                "fail_rate_1m": 0.0625,
                "fail_rate_3m": 0.0476,
                "fail_rate_12m": 0.0476,
                "trend_flag": "up",
                "last_execution_level": "DATASET",
            },
            ("SRC2", "tenant2", "uuid2", "Dataset B", 102): {
                "rule_name": "NULL_CHECK",
                "rule_type": "ATTRIBUTE",
                "dimension": "Completeness",
                "rule_description": "Null value validation",
                "category": 2,
                "business_date_latest": date(2024, 1, 16),
                "dataset_record_count_latest": 75000,
                "filtered_record_count_latest": 73000,
                "pass_count_total": 200,
                "fail_count_total": 0,
                "pass_count_1m": 60,
                "fail_count_1m": 0,
                "pass_count_3m": 150,
                "fail_count_3m": 0,
                "pass_count_12m": 200,
                "fail_count_12m": 0,
                "fail_rate_total": 0.0000,
                "fail_rate_1m": 0.0000,
                "fail_rate_3m": 0.0000,
                "fail_rate_12m": 0.0000,
                "trend_flag": "=",
                "last_execution_level": "ATTRIBUTE",
            },
        }

        generator = SummaryGenerator(output_dir=temp_output_dir)

        # Test CSV
        csv_path = generator.generate_csv(data)
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert df.iloc[0]["dataset_name"] == "Dataset A"
        assert df.iloc[1]["dataset_name"] == "Dataset B"

        # Test NL
        nl_path = generator.generate_nl_sentences(data)
        with open(nl_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 2
        assert "Dataset A" in lines[0]
        assert "Dataset B" in lines[1]

    def test_utf8_encoding_handling(self, temp_output_dir):
        """Test proper UTF-8 encoding for special characters."""
        data = {
            ("SRC1", "tenant1", "uuid1", "Datäset Ñame", 101): {
                "rule_name": "SPÉCIÅL_CHECK",
                "rule_type": "DATASET",
                "dimension": "Correctness",
                "rule_description": "Special character tëst",
                "category": 1,
                "business_date_latest": date(2024, 1, 15),
                "dataset_record_count_latest": 50000,
                "filtered_record_count_latest": 48000,
                "pass_count_total": 100,
                "fail_count_total": 5,
                "pass_count_1m": 30,
                "fail_count_1m": 2,
                "pass_count_3m": 80,
                "fail_count_3m": 4,
                "pass_count_12m": 100,
                "fail_count_12m": 5,
                "fail_rate_total": 0.0476,
                "fail_rate_1m": 0.0625,
                "fail_rate_3m": 0.0476,
                "fail_rate_12m": 0.0476,
                "trend_flag": "up",
                "last_execution_level": "DATASET",
            }
        }

        generator = SummaryGenerator(output_dir=temp_output_dir)

        # Test CSV encoding
        csv_path = generator.generate_csv(data)
        df = pd.read_csv(csv_path, encoding="utf-8")
        assert df.iloc[0]["dataset_name"] == "Datäset Ñame"
        assert df.iloc[0]["rule_name"] == "SPÉCIÅL_CHECK"

        # Test NL encoding
        nl_path = generator.generate_nl_sentences(data)
        with open(nl_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Datäset Ñame" in content
        assert "SPÉCIÅL_CHECK" in content


class TestStandaloneFunctions:
    """Test cases for standalone utility functions."""

    def test_generate_full_summary_csv_function(self):
        """Test standalone CSV generation function."""
        data = {
            ("SRC1", "tenant1", "uuid1", "Dataset A", 101): {
                "rule_name": "ROW_COUNT",
                "business_date_latest": date(2024, 1, 15),
                "fail_count_total": 5,
                "trend_flag": "up",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = generate_full_summary_csv(data, tmpdir)
            assert output_path.exists()
            assert output_path.name == "full_summary.csv"

    def test_generate_nl_sentences_function(self):
        """Test standalone NL generation function."""
        data = {
            ("SRC1", "tenant1", "uuid1", "Dataset A", 101): {
                "rule_name": "ROW_COUNT",
                "business_date_latest": date(2024, 1, 15),
                "fail_count_total": 5,
                "pass_count_total": 100,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = generate_nl_sentences(data, tmpdir)
            assert output_path.exists()
            assert output_path.name == "nl_all_rows.txt"


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.fixture
    def temp_output_dir(self):
        """Temporary directory for output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_empty_data_handling(self, temp_output_dir):
        """Test handling empty aggregated data."""
        generator = SummaryGenerator(output_dir=temp_output_dir)

        csv_path = generator.generate_csv({})
        df = pd.read_csv(csv_path)
        assert len(df) == 0
        assert len(df.columns) == 27  # Schema preserved

        nl_path = generator.generate_nl_sentences({})
        with open(nl_path, "r") as f:
            content = f.read().strip()
        assert content == ""  # Empty file for empty data

    def test_output_directory_permission_error(self):
        """Test handling of output directory permission errors."""
        # Test with an invalid path that would cause an error
        generator = SummaryGenerator(output_dir="/dev/null/invalid_subdir")

        # This should raise an appropriate exception when trying to create directory
        with pytest.raises((OSError, FileNotFoundError, NotADirectoryError)):
            generator._ensure_output_directory()

    def test_file_size_validation(self, temp_output_dir):
        """Test that output files meet size requirements (<2MB)."""
        # Generate larger dataset to test file size
        large_data = {}
        for i in range(1000):  # 1000 rows should still be well under 2MB
            key = (f"SRC{i}", f"tenant{i}", f"uuid{i}", f"Dataset {i}", 100 + i)
            large_data[key] = {
                "rule_name": f"RULE_{i}",
                "rule_type": "DATASET",
                "dimension": "Correctness",
                "rule_description": f"Test rule {i} description",
                "category": 1,
                "business_date_latest": date(2024, 1, 15),
                "dataset_record_count_latest": 50000,
                "filtered_record_count_latest": 48000,
                "pass_count_total": 100,
                "fail_count_total": 5,
                "pass_count_1m": 30,
                "fail_count_1m": 2,
                "pass_count_3m": 80,
                "fail_count_3m": 4,
                "pass_count_12m": 100,
                "fail_count_12m": 5,
                "fail_rate_total": 0.0476,
                "fail_rate_1m": 0.0625,
                "fail_rate_3m": 0.0476,
                "fail_rate_12m": 0.0476,
                "trend_flag": "up",
                "last_execution_level": "DATASET",
            }

        generator = SummaryGenerator(output_dir=temp_output_dir)
        csv_path = generator.generate_csv(large_data)
        nl_path = generator.generate_nl_sentences(large_data)

        # Check file sizes (2MB = 2,097,152 bytes)
        csv_size = csv_path.stat().st_size
        nl_size = nl_path.stat().st_size

        assert csv_size < 2_097_152, f"CSV file too large: {csv_size} bytes"
        assert nl_size < 2_097_152, f"NL file too large: {nl_size} bytes"
