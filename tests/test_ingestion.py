"""
Test cases for data ingestion module following TDD approach.
Stage 1: Core Infrastructure & Data Ingestion
"""

import pytest
import pandas as pd
import tempfile
import os

# Import will fail initially (RED phase)
from data_quality_summarizer.ingestion import CSVIngester


class TestCSVIngester:

    def test_csv_ingester_initialization(self):
        """Test CSVIngester can be instantiated with default chunk size."""
        ingester = CSVIngester()
        assert ingester.chunk_size == 20000

    def test_csv_ingester_custom_chunk_size(self):
        """Test CSVIngester accepts custom chunk size."""
        ingester = CSVIngester(chunk_size=10000)
        assert ingester.chunk_size == 10000

    def test_read_csv_chunks_returns_generator(self):
        """Test that read_csv_chunks returns a generator."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            header = (
                "source,tenant_id,dataset_uuid,dataset_name,business_date,"
                "dataset_record_count,rule_code,level_of_execution,"
                "attribute_name,results,context_id,filtered_record_count\n"
            )
            f.write(header)
            row = (
                "system1,tenant1,uuid1,dataset1,2023-01-01,1000,101,DATASET,,"
                '"{""result"":""Pass""}",ctx1,1000\n'
            )
            f.write(row)

        try:
            ingester = CSVIngester(chunk_size=1)
            chunks = ingester.read_csv_chunks(f.name)

            # Should be a generator
            assert hasattr(chunks, "__iter__")
            assert hasattr(chunks, "__next__")

            # Should yield DataFrame chunks
            first_chunk = next(chunks)
            assert isinstance(first_chunk, pd.DataFrame)
            assert len(first_chunk) == 1

        finally:
            os.unlink(f.name)

    def test_read_csv_chunks_with_proper_dtypes(self):
        """Test that CSV reading applies proper data types for memory efficiency."""
        # Create a temporary CSV file with multiple rows
        header = (
            "source,tenant_id,dataset_uuid,dataset_name,business_date,"
            "dataset_record_count,rule_code,level_of_execution,"
            "attribute_name,results,context_id,filtered_record_count"
        )
        row1 = (
            "system1,tenant1,uuid1,dataset1,2023-01-01,1000,101,DATASET,,"
            '"{""result"":""Pass""}",ctx1,1000'
        )
        row2 = (
            "system2,tenant2,uuid2,dataset2,2023-01-02,2000,102,ATTRIBUTE,attr1,"
            '"{""result"":""Fail""}",ctx2,1500'
        )
        csv_content = f"{header}\n{row1}\n{row2}"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write(csv_content)

        try:
            ingester = CSVIngester(chunk_size=5)
            chunks = list(ingester.read_csv_chunks(f.name))

            # Should have one chunk since chunk_size > number of rows
            assert len(chunks) == 1
            df = chunks[0]

            # Check that string columns are properly typed
            assert str(df["source"].dtype).startswith("string")
            assert str(df["tenant_id"].dtype).startswith("string")
            assert str(df["dataset_uuid"].dtype).startswith("string")
            assert str(df["dataset_name"].dtype).startswith("string")

            # Check that numeric columns are properly typed
            assert df["dataset_record_count"].dtype in ["int64", "Int64"]
            assert df["rule_code"].dtype in ["int64", "Int64"]
            assert df["filtered_record_count"].dtype in ["int64", "Int64"]

        finally:
            os.unlink(f.name)

    def test_memory_usage_stays_under_limit(self):
        """Test that memory usage during ingestion stays reasonable."""
        # This is a basic test - in real scenario we'd measure actual memory
        # For now, we'll test that chunk size is respected
        header = (
            "source,tenant_id,dataset_uuid,dataset_name,business_date,"
            "dataset_record_count,rule_code,level_of_execution,"
            "attribute_name,results,context_id,filtered_record_count\n"
        )
        csv_content = header

        # Create 25 rows of data
        for i in range(25):
            row = (
                f"system{i},tenant{i},uuid{i},dataset{i},2023-01-{i+1:02d},"
                f'1000,{100+i},DATASET,,"{{\\"result\\":\\"Pass\\"}}",ctx{i},1000\n'
            )
            csv_content += row

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write(csv_content)

        try:
            # Set chunk size to 10, so we should get 3 chunks (10, 10, 5)
            ingester = CSVIngester(chunk_size=10)
            chunks = list(ingester.read_csv_chunks(f.name))

            assert len(chunks) == 3
            assert len(chunks[0]) == 10
            assert len(chunks[1]) == 10
            assert len(chunks[2]) == 5

        finally:
            os.unlink(f.name)

    def test_handles_malformed_csv_gracefully(self):
        """Test that malformed CSV files are handled with appropriate logging."""
        # Create a malformed CSV (missing columns)
        malformed_csv = """source,tenant_id,dataset_uuid
system1,tenant1"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write(malformed_csv)

        try:
            ingester = CSVIngester()

            # Should raise an appropriate exception or handle gracefully
            with pytest.raises(
                (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError)
            ):
                list(ingester.read_csv_chunks(f.name))

        finally:
            os.unlink(f.name)

    def test_file_not_found_error(self):
        """Test that file not found errors are handled properly."""
        ingester = CSVIngester()

        with pytest.raises(FileNotFoundError):
            list(ingester.read_csv_chunks("nonexistent_file.csv"))
