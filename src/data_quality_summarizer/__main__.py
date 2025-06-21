"""
CLI entry point for the Data Quality Summarizer.
Orchestrates the complete pipeline from CSV input to artifact generation.
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any

# Import pipeline components
from .ingestion import CSVIngester
from .rules import load_rule_metadata
from .aggregator import StreamingAggregator
from .summarizer import SummaryGenerator


def setup_logging() -> None:
    """Configure structured logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the CLI."""
    parser = argparse.ArgumentParser(
        description="Data Quality Summarizer - Generate summary artifacts from CSV"
        " data quality results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src input.csv rules.json
  python -m src input.csv rules.json --chunk-size 50000
  python -m src input.csv rules.json --output-dir /custom/path
        """,
    )

    parser.add_argument(
        "csv_file", help="Path to input CSV file containing data quality results"
    )

    parser.add_argument(
        "rule_metadata_file", help="Path to JSON file containing rule metadata mappings"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20000,
        help="Number of rows to process per chunk (default: 20000)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="resources/artifacts",
        help="Output directory for generated artifacts (default: resources/artifacts)",
    )

    return parser.parse_args()


def run_pipeline(
    csv_file: str,
    rule_metadata_file: str,
    chunk_size: int = 20000,
    output_dir: str = "resources/artifacts",
) -> Dict[str, Any]:
    """
    Execute the complete data quality summarization pipeline.

    Args:
        csv_file: Path to input CSV file
        rule_metadata_file: Path to rule metadata JSON file
        chunk_size: Number of rows per processing chunk
        output_dir: Directory for output artifacts

    Returns:
        Dictionary containing pipeline execution results
    """
    logger = logging.getLogger(__name__)

    try:
        start_time = time.time()

        logger.info(f"Starting pipeline - CSV: {csv_file}, Rules: {rule_metadata_file}")
        logger.info(f"Configuration - Chunk size: {chunk_size}, Output: {output_dir}")

        # Stage 1: Initialize components
        logger.info("Initializing pipeline components...")

        ingester = CSVIngester(chunk_size=chunk_size)
        aggregator = StreamingAggregator()
        summarizer = SummaryGenerator(output_dir=output_dir)

        # Stage 2: Load rule metadata
        logger.info("Loading rule metadata...")
        rule_metadata = load_rule_metadata(rule_metadata_file)
        logger.info(f"Loaded {len(rule_metadata)} rule definitions")

        # Stage 3: Process CSV in chunks
        logger.info("Starting chunked CSV processing...")
        rows_processed = 0
        row_failures = 0

        for chunk_num, chunk in enumerate(ingester.read_csv_chunks(csv_file)):
            logger.info(f"Processing chunk {chunk_num + 1} ({len(chunk)} rows)")

            # Process each row in the chunk
            for _, row in chunk.iterrows():
                try:
                    aggregator.process_row(row)
                except Exception as e:
                    row_failures += 1
                    logger.error(
                        f"Failed to process row {rows_processed + 1}: {str(e)} "
                        f"(Total failures: {row_failures})"
                    )
                    continue

            rows_processed += len(chunk)
            logger.debug(f"Processed {rows_processed} total rows")

        logger.info(
            f"Completed processing {rows_processed} rows "
            f"({row_failures} failures, {rows_processed - row_failures} successful)"
        )

        # Stage 4: Finalize aggregation and generate metrics
        logger.info("Finalizing aggregation and calculating metrics...")
        aggregator.finalize_aggregation()
        unique_keys = len(aggregator.accumulator)
        logger.info(
            f"Generated metrics for {unique_keys} unique dataset-rule combinations"
        )

        # Stage 5: Convert to summarizer format with rule metadata enrichment
        logger.info("Converting metrics to summary format...")
        summary_data = {}
        for key, metrics in aggregator.accumulator.items():
            source, tenant_id, dataset_uuid, dataset_name, rule_code = key

            # Get rule metadata
            rule_info = rule_metadata.get(rule_code)
            if rule_info is None:
                logger.warning(f"Rule metadata not found for rule_code: {rule_code}")
                continue

            # Convert AggregationMetrics to dictionary format
            summary_entry = {
                "rule_name": rule_info.rule_name,
                "rule_type": rule_info.rule_type,
                "dimension": rule_info.dimension,
                "rule_description": rule_info.rule_description,
                "category": rule_info.category,
                "business_date_latest": metrics.business_date_latest,
                "dataset_record_count_latest": metrics.dataset_record_count_latest,
                "filtered_record_count_latest": metrics.filtered_record_count_latest,
                "pass_count_total": metrics.pass_count_total,
                "fail_count_total": metrics.fail_count_total,
                "pass_count_1m": metrics.pass_count_1m,
                "fail_count_1m": metrics.fail_count_1m,
                "pass_count_3m": metrics.pass_count_3m,
                "fail_count_3m": metrics.fail_count_3m,
                "pass_count_12m": metrics.pass_count_12m,
                "fail_count_12m": metrics.fail_count_12m,
                "fail_rate_total": metrics.fail_rate_total,
                "fail_rate_1m": metrics.fail_rate_1m,
                "fail_rate_3m": metrics.fail_rate_3m,
                "fail_rate_12m": metrics.fail_rate_12m,
                "trend_flag": metrics.trend_flag,
                "last_execution_level": metrics.last_execution_level,
            }
            summary_data[key] = summary_entry

        logger.info(f"Converted {len(summary_data)} entries for export")

        # Stage 6: Export artifacts
        logger.info("Exporting summary artifacts...")
        summary_csv_path = summarizer.generate_csv(summary_data)
        nl_path = summarizer.generate_nl_sentences(summary_data)

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"Pipeline completed successfully in {processing_time:.2f} seconds")

        return {
            "success": True,
            "rows_processed": rows_processed,
            "row_failures": row_failures,
            "unique_keys": unique_keys,
            "processing_time": processing_time,
            "memory_peak_mb": _get_memory_usage(),
            "output_files": {
                "summary_csv": str(summary_csv_path),
                "natural_language": str(nl_path),
            },
        }

    except FileNotFoundError as e:
        error_msg = f"Input file not found: {e}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return {"success": False, "error": error_msg}


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0  # psutil not available


def main() -> int:
    """Main entry point for the CLI application."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        args = parse_arguments()

        # Validate input files exist
        if not Path(args.csv_file).exists():
            logger.error(f"CSV file not found: {args.csv_file}")
            return 1

        if not Path(args.rule_metadata_file).exists():
            logger.error(f"Rule metadata file not found: {args.rule_metadata_file}")
            return 1

        # Execute pipeline
        result = run_pipeline(
            csv_file=args.csv_file,
            rule_metadata_file=args.rule_metadata_file,
            chunk_size=args.chunk_size,
            output_dir=args.output_dir,
        )

        if result["success"]:
            print("\n🎉 SUCCESS: Data Quality Summarizer completed!")
            print(f"   📊 Processed: {result['rows_processed']:,} rows")
            print(f"   ❌ Failures: {result.get('row_failures', 0):,} rows")
            print(f"   🔑 Unique keys: {result['unique_keys']:,}")
            print(f"   ⏱️  Time: {result['processing_time']:.2f} seconds")
            print(f"   💾 Memory peak: {result.get('memory_peak_mb', 0):.1f} MB")
            print("   📁 Output files:")
            print(f"      • {result['output_files']['summary_csv']}")
            print(f"      • {result['output_files']['natural_language']}")
            return 0
        else:
            print(f"\n❌ ERROR: Pipeline failed - {result['error']}", file=sys.stderr)
            return 1

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\n⚠️  Pipeline interrupted by user", file=sys.stderr)
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n💥 UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
