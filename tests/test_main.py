"""
Test suite for CLI integration and end-to-end pipeline functionality.
Tests the complete workflow from CSV input to artifact generation.
"""

import pytest
import tempfile
import os
import sys
import json
import logging
import pandas as pd
from unittest.mock import patch
from io import StringIO
from data_quality_summarizer import __main__ as main_module

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def temp_files():
    """Create temporary test files for integration testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test CSV data
        csv_data = {
            "source": ["TestSystem"] * 3,
            "tenant_id": ["tenant1"] * 3,
            "dataset_uuid": ["uuid-123"] * 3,
            "dataset_name": ["TestDataset"] * 3,
            "business_date": ["2024-01-15", "2024-01-16", "2024-01-17"],
            "dataset_record_count": [1000, 1000, 1000],
            "rule_code": [101, 102, 101],
            "level_of_execution": ["DATASET", "ATTRIBUTE", "DATASET"],
            "attribute_name": ["", "test_column", ""],
            "results": [
                '{"result": "Pass"}',
                '{"result": "Fail"}',
                '{"result": "Pass"}',
            ],
            "context_id": ["ctx1", "ctx2", "ctx3"],
            "filtered_record_count": [950, 950, 950],
        }
        csv_file = os.path.join(temp_dir, "test_input.csv")
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)

        # Create test rule metadata
        rule_metadata = {
            101: {
                "rule_name": "ROW_COUNT",
                "rule_type": "DATASET",
                "dimension": "Completeness",
                "rule_description": "Check total row count",
                "category": "C1",
            },
            102: {
                "rule_name": "NULL_CHECK",
                "rule_type": "ATTRIBUTE",
                "dimension": "Completeness",
                "rule_description": "Check for null values",
                "category": "C2",
            },
        }
        rules_file = os.path.join(temp_dir, "test_rules.json")
        with open(rules_file, "w") as f:
            json.dump(rule_metadata, f)

        yield {
            "temp_dir": temp_dir,
            "csv_file": csv_file,
            "rules_file": rules_file,
            "output_dir": os.path.join(temp_dir, "artifacts"),
        }


class TestCLIArgumentParsing:
    """Test command-line argument parsing functionality."""

    def test_parse_args_with_required_arguments(self):
        """Test parsing with required CSV and rule metadata arguments."""
        with patch("sys.argv", ["__main__.py", "input.csv", "rules.json"]):
            args = main_module.parse_arguments()
            assert args.csv_file == "input.csv"
            assert args.rule_metadata_file == "rules.json"
            assert args.chunk_size == 20000  # default

    def test_parse_args_with_custom_chunk_size(self):
        """Test parsing with custom chunk size argument."""
        with patch(
            "sys.argv",
            ["__main__.py", "input.csv", "rules.json", "--chunk-size", "50000"],
        ):
            args = main_module.parse_arguments()
            assert args.chunk_size == 50000

    def test_parse_args_with_custom_output_dir(self):
        """Test parsing with custom output directory argument."""
        with patch(
            "sys.argv",
            ["__main__.py", "input.csv", "rules.json", "--output-dir", "/custom/path"],
        ):
            args = main_module.parse_arguments()
            assert args.output_dir == "/custom/path"

    def test_parse_args_missing_required_arguments(self):
        """Test that missing required arguments raises SystemExit."""
        with patch("sys.argv", ["__main__.py"]):
            with pytest.raises(SystemExit):
                main_module.parse_arguments()

    def test_parse_args_help_flag(self):
        """Test that help flag works correctly."""
        with patch("sys.argv", ["__main__.py", "--help"]):
            with pytest.raises(SystemExit):
                main_module.parse_arguments()


class TestPipelineOrchestration:
    """Test complete pipeline orchestration and integration."""

    def test_run_pipeline_success(self, temp_files):
        """Test successful end-to-end pipeline execution."""
        result = main_module.run_pipeline(
            csv_file=temp_files["csv_file"],
            rule_metadata_file=temp_files["rules_file"],
            chunk_size=1000,
            output_dir=temp_files["output_dir"],
        )

        assert result["success"] is True
        assert result["rows_processed"] == 3
        assert result["unique_keys"] == 2  # Two distinct (dataset, rule) combinations
        assert "processing_time" in result
        assert result["processing_time"] < 120  # Should be much faster than 2 minutes

        # Verify output files were created
        summary_file = os.path.join(temp_files["output_dir"], "full_summary.csv")
        nl_file = os.path.join(temp_files["output_dir"], "nl_all_rows.txt")

        assert os.path.exists(summary_file)
        assert os.path.exists(nl_file)

        # Verify output file sizes are reasonable
        assert os.path.getsize(summary_file) < 2 * 1024 * 1024  # <2MB
        assert os.path.getsize(nl_file) < 1024 * 1024  # <1MB

    def test_run_pipeline_file_not_found(self):
        """Test pipeline handles missing input files gracefully."""
        result = main_module.run_pipeline(
            csv_file="nonexistent.csv",
            rule_metadata_file="nonexistent_rules.json",
            chunk_size=1000,
            output_dir="/tmp/test_output",
        )

        assert result["success"] is False
        assert "error" in result
        # Should contain information about file not found
        assert "not found" in result["error"].lower()

    def test_run_pipeline_invalid_rules_file(self, temp_files):
        """Test pipeline handles invalid rule metadata file."""
        # Create invalid JSON file
        invalid_rules_file = os.path.join(temp_files["temp_dir"], "invalid_rules.json")
        with open(invalid_rules_file, "w") as f:
            f.write("invalid json content")

        result = main_module.run_pipeline(
            csv_file=temp_files["csv_file"],
            rule_metadata_file=invalid_rules_file,
            chunk_size=1000,
            output_dir=temp_files["output_dir"],
        )

        assert result["success"] is False
        assert "error" in result


class TestMainEntryPoint:
    """Test main entry point functionality."""

    @patch("sys.stderr", new_callable=StringIO)
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_with_valid_arguments(
        self, mock_stdout, mock_stderr, monkeypatch, temp_files
    ):
        """Test main function with valid arguments."""
        # Mock successful pipeline execution
        mock_result = {
            "success": True,
            "rows_processed": 1000,
            "unique_keys": 50,
            "processing_time": 45.5,
            "memory_peak_mb": 250,
            "output_files": {
                "summary_csv": "/tmp/test/full_summary.csv",
                "natural_language": "/tmp/test/nl_all_rows.txt",
            },
        }

        with patch.object(main_module, "run_pipeline", return_value=mock_result):
            with patch(
                "sys.argv",
                ["__main__.py", temp_files["csv_file"], temp_files["rules_file"]],
            ):
                exit_code = main_module.main()

        assert exit_code == 0
        output = mock_stdout.getvalue()
        assert "SUCCESS" in output
        assert "1,000" in output  # rows processed (with comma separator)
        assert "50" in output  # unique keys

    @patch("sys.stderr", new_callable=StringIO)
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_with_pipeline_failure(
        self, mock_stdout, mock_stderr, monkeypatch, temp_files
    ):
        """Test main function when pipeline fails."""
        mock_result = {"success": False, "error": "File not found: test.csv"}

        with patch.object(main_module, "run_pipeline", return_value=mock_result):
            with patch(
                "sys.argv",
                ["__main__.py", temp_files["csv_file"], temp_files["rules_file"]],
            ):
                exit_code = main_module.main()

        assert exit_code == 1
        output = mock_stderr.getvalue()
        assert "ERROR" in output
        assert "File not found" in output


class TestPerformanceRequirements:
    """Test performance and memory requirements."""

    def test_memory_usage_under_limit(self, temp_files):
        """Test that memory usage stays under 1GB during processing."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Run pipeline and monitor memory
        result = main_module.run_pipeline(
            csv_file=temp_files["csv_file"],
            rule_metadata_file=temp_files["rules_file"],
            chunk_size=1000,
            output_dir=temp_files["output_dir"],
        )

        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_used = peak_memory - initial_memory

        assert result["success"] is True
        assert memory_used < 1024  # <1GB memory usage requirement

    def test_processing_time_under_limit(self, temp_files):
        """Test that processing time is reasonable for small datasets."""
        import time

        start_time = time.time()
        result = main_module.run_pipeline(
            csv_file=temp_files["csv_file"],
            rule_metadata_file=temp_files["rules_file"],
            chunk_size=1000,
            output_dir=temp_files["output_dir"],
        )
        end_time = time.time()

        processing_time = end_time - start_time

        assert result["success"] is True
        assert processing_time < 10  # Should be much faster for small test data
        assert result["processing_time"] < 10


class TestLoggingAndProgress:
    """Test logging and progress reporting functionality."""

    @patch("logging.basicConfig")
    def test_pipeline_logging_setup(self, mock_basic_config):
        """Test that logging is properly configured."""
        main_module.setup_logging()

        mock_basic_config.assert_called_once()
        # Verify logging configuration parameters
        call_args = mock_basic_config.call_args
        assert call_args.kwargs["level"] == logging.INFO

    def test_progress_reporting(self, temp_files, caplog):
        """Test that progress is reported during pipeline execution."""
        with caplog.at_level("INFO"):
            result = main_module.run_pipeline(
                csv_file=temp_files["csv_file"],
                rule_metadata_file=temp_files["rules_file"],
                chunk_size=1000,
                output_dir=temp_files["output_dir"],
            )

        assert result["success"] is True

        # Check that progress messages are logged
        log_messages = [record.message for record in caplog.records]
        assert any("Starting pipeline" in msg for msg in log_messages)
        assert any("Pipeline completed" in msg for msg in log_messages)
