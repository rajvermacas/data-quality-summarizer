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
    # Check if first argument is a known ML command
    import sys
    ml_commands = ['train-model', 'predict', 'batch-predict']
    
    if len(sys.argv) > 1 and sys.argv[1] in ml_commands:
        # Use subcommand parsing for ML commands
        return parse_ml_arguments()
    elif len(sys.argv) > 1 and sys.argv[1].startswith('-'):
        # Handle options like --help, use original parsing
        return parse_original_arguments()
    elif len(sys.argv) > 1 and not Path(sys.argv[1]).exists() and sys.argv[1] not in ['--help', '-h']:
        # Check if first argument looks like an invalid command
        potential_command = sys.argv[1]
        if not potential_command.endswith('.csv') and not potential_command.endswith('.json'):
            print(f"Error: Unknown command '{potential_command}'", file=sys.stderr)
            print(f"Available commands: {', '.join(ml_commands)}", file=sys.stderr)
            print("Or provide CSV and JSON files for data summarization", file=sys.stderr)
            sys.exit(2)
        # Fall through to original parsing for file arguments
        return parse_original_arguments()
    else:
        # Use original parsing for backward compatibility
        return parse_original_arguments()

def parse_original_arguments() -> argparse.Namespace:
    """Parse arguments for original summarizer functionality."""
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

    args = parser.parse_args()
    args.command = 'summarize'  # Set default command for backward compatibility
    return args

def parse_ml_arguments() -> argparse.Namespace:
    """Parse arguments for ML functionality."""
    parser = argparse.ArgumentParser(
        description="Data Quality Summarizer - ML Training and Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ML Commands:
  train-model input.csv rules.json --output-model model.pkl
  predict --model model.pkl --dataset-uuid uuid123 --rule-code R001 --date 2024-01-15
  batch-predict --model model.pkl --input predictions.csv --output results.csv
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train model command
    train_parser = subparsers.add_parser('train-model', help='Train ML model')
    train_parser.add_argument('csv_file', help='Input CSV file for training')
    train_parser.add_argument('rule_metadata_file', help='Rule metadata JSON file')
    train_parser.add_argument('--output-model', required=True, help='Output model file path')
    train_parser.add_argument('--chunk-size', type=int, default=20000, help='Chunk size for processing')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make single prediction')
    predict_parser.add_argument('--model', required=True, help='Trained model file')
    predict_parser.add_argument('--dataset-uuid', required=True, help='Dataset UUID')
    predict_parser.add_argument('--rule-code', required=True, help='Rule code')
    predict_parser.add_argument('--date', required=True, help='Business date (YYYY-MM-DD)')
    predict_parser.add_argument('--historical-data', help='Historical data CSV file')

    # Batch predict command
    batch_parser = subparsers.add_parser('batch-predict', help='Make batch predictions')
    batch_parser.add_argument('--model', required=True, help='Trained model file')
    batch_parser.add_argument('--input', required=True, help='Input predictions CSV file')
    batch_parser.add_argument('--output', required=True, help='Output results CSV file')
    batch_parser.add_argument('--historical-data', help='Historical data CSV file')

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


def handle_train_model_command(args: argparse.Namespace) -> int:
    """Handle train-model command."""
    from .ml.pipeline import MLPipeline
    from .rules import load_rule_metadata
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input files
        if not Path(args.csv_file).exists():
            logger.error(f"CSV file not found: {args.csv_file}")
            return 1
        
        if not Path(args.rule_metadata_file).exists():
            logger.error(f"Rule metadata file not found: {args.rule_metadata_file}")
            return 1
        
        # Load rule metadata
        rule_metadata = load_rule_metadata(args.rule_metadata_file)
        
        # Train model
        pipeline = MLPipeline()
        result = pipeline.train_model(
            csv_file=args.csv_file,
            rule_metadata=rule_metadata,
            output_model_path=args.output_model
        )
        
        if result['success']:
            print(f"\n🎉 SUCCESS: Model training completed!")
            print(f"   ⏱️  Training time: {result['training_time']:.2f} seconds")
            print(f"   📊 Samples trained: {result['samples_trained']:,}")
            print(f"   📊 Samples tested: {result['samples_tested']:,}")
            print(f"   💾 Memory peak: {result['memory_peak_mb']:.1f} MB")
            print(f"   📁 Model saved: {result['model_path']}")
            print(f"   📈 Evaluation metrics: {result['evaluation_metrics']}")
            return 0
        else:
            print(f"\n❌ ERROR: Model training failed - {result['error']}", file=sys.stderr)
            return 1
            
    except Exception as e:
        logger.error(f"Training command failed: {e}")
        print(f"\n💥 UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 1

def handle_predict_command(args: argparse.Namespace) -> int:
    """Handle predict command."""
    from .ml.predictor import Predictor
    import pandas as pd
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate model file
        if not Path(args.model).exists():
            logger.error(f"Model file not found: {args.model}")
            return 1
        
        # Load historical data if provided
        historical_data = pd.DataFrame()
        if args.historical_data and Path(args.historical_data).exists():
            historical_data = pd.read_csv(args.historical_data)
        
        # Make prediction
        predictor = Predictor(model_path=args.model, historical_data=historical_data)
        prediction = predictor.predict(
            dataset_uuid=getattr(args, 'dataset_uuid', ''),
            rule_code=getattr(args, 'rule_code', ''),  
            business_date=args.date
        )
        
        print(f"\n🎯 PREDICTION RESULT:")
        print(f"   Dataset UUID: {getattr(args, 'dataset_uuid', '')}")
        print(f"   Rule Code: {getattr(args, 'rule_code', '')}")
        print(f"   Business Date: {args.date}")
        print(f"   Predicted Pass Percentage: {prediction:.2f}%")
        return 0
        
    except Exception as e:
        logger.error(f"Prediction command failed: {e}")
        print(f"\n💥 PREDICTION ERROR: {e}", file=sys.stderr)
        return 1

def handle_batch_predict_command(args: argparse.Namespace) -> int:
    """Handle batch-predict command."""
    from .ml.batch_predictor import BatchPredictor
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input files
        if not Path(args.model).exists():
            logger.error(f"Model file not found: {args.model}")
            return 1
            
        if not Path(args.input).exists():
            logger.error(f"Input CSV file not found: {args.input}")
            return 1
        
        # Process batch predictions
        batch_predictor = BatchPredictor(model_path=args.model)
        result = batch_predictor.process_batch_csv(
            input_csv=args.input,
            output_csv=args.output,
            historical_data_csv=args.historical_data or ""
        )
        
        if result['success']:
            print(f"\n🎉 SUCCESS: Batch predictions completed!")
            print(f"   📊 Predictions processed: {result['predictions_processed']:,}")
            print(f"   ⏱️  Processing time: {result['processing_time']:.2f} seconds")
            print(f"   📁 Results saved: {result['output_file']}")
            return 0
        else:
            print(f"\n❌ ERROR: Batch prediction failed - {result['error']}", file=sys.stderr)
            return 1
            
    except Exception as e:
        logger.error(f"Batch prediction command failed: {e}")
        print(f"\n💥 UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 1

def main() -> int:
    """Main entry point for the CLI application."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        args = parse_arguments()
        
        # Route to appropriate command handler
        if args.command == 'train-model':
            return handle_train_model_command(args)
        elif args.command == 'predict':
            return handle_predict_command(args)
        elif args.command == 'batch-predict':
            return handle_batch_predict_command(args)
        elif args.command == 'summarize':
            # Original summarizer functionality
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
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        print("\n⚠️  Operation interrupted by user", file=sys.stderr)
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n💥 UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
