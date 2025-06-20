"""
Data ingestion module for chunked CSV reading.
Stage 1: Core Infrastructure & Data Ingestion

This module provides efficient chunked CSV reading capabilities
for large data quality result files while maintaining low memory usage.
"""

import pandas as pd
import structlog
from typing import Iterator, Dict, Any

# Configure structured logging
logger = structlog.get_logger(__name__)


class CSVIngester:
    """
    Chunked CSV reader for data quality results.

    Designed for memory-efficient processing of large CSV files
    by reading data in configurable chunks.
    """

    def __init__(self, chunk_size: int = 20000) -> None:
        """
        Initialize CSVIngester.

        Args:
            chunk_size: Number of rows to read per chunk (default: 20000)
        """
        self.chunk_size = chunk_size
        logger.info("CSVIngester initialized", chunk_size=chunk_size)

    def read_csv_chunks(self, file_path: str) -> Iterator[pd.DataFrame]:
        """
        Read CSV file in chunks for memory-efficient processing.

        Args:
            file_path: Path to the CSV file to read

        Yields:
            pd.DataFrame: Chunks of the CSV data

        Raises:
            FileNotFoundError: If the specified file doesn't exist
            pd.errors.EmptyDataError: If the CSV file is empty or malformed
            pd.errors.ParserError: If the CSV file has parsing errors
        """
        logger.info(
            "Starting CSV ingestion", file_path=file_path, chunk_size=self.chunk_size
        )

        # Define data types for memory efficiency
        dtype_mapping = self._get_dtype_mapping()

        try:
            # Use pandas read_csv with chunksize for memory-efficient reading
            chunk_reader = pd.read_csv(
                file_path,
                chunksize=self.chunk_size,
                dtype=dtype_mapping,  # type: ignore[arg-type]
                parse_dates=["business_date"],
            )

            chunk_count = 0
            total_rows = 0

            for chunk in chunk_reader:
                chunk_count += 1
                total_rows += len(chunk)

                logger.debug(
                    "Processing chunk",
                    chunk_number=chunk_count,
                    chunk_rows=len(chunk),
                    total_rows_processed=total_rows,
                )

                yield chunk

            logger.info(
                "CSV ingestion completed",
                total_chunks=chunk_count,
                total_rows=total_rows,
            )

        except FileNotFoundError as e:
            logger.error("CSV file not found", file_path=file_path, error=str(e))
            raise
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.error("CSV parsing error", file_path=file_path, error=str(e))
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during CSV ingestion",
                file_path=file_path,
                error=str(e),
            )
            raise

    def _get_dtype_mapping(self) -> Dict[str, Any]:
        """
        Get data type mapping for CSV columns to optimize memory usage.

        This mapping prevents pandas from performing expensive type inference
        and ensures consistent data types across all chunks.

        Returns:
            Dict mapping column names to pandas data types
        """
        return {
            # Identity fields - use string for memory efficiency
            "source": "string",
            "tenant_id": "string",
            "dataset_uuid": "string",
            "dataset_name": "string",
            # Numeric fields - use nullable Int64 to handle missing values
            "dataset_record_count": "Int64",
            "rule_code": "Int64",
            "filtered_record_count": "Int64",
            # Execution context fields
            "level_of_execution": "string",
            "attribute_name": "string",
            "context_id": "string",
            # JSON results field - keep as string for later parsing
            "results": "string",
        }
