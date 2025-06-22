# Data Quality Summarizer

**An offline data processing system that transforms large CSV files containing data quality check results into LLM-optimized summary artifacts.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Test Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](htmlcov/index.html)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🚀 Features

- **Memory Efficient**: Processes 100k+ row CSV files using <1GB RAM via chunked streaming
- **High Performance**: Completes processing in <2 minutes on consumer-grade hardware  
- **Comprehensive Analytics**: Generates rolling time-window metrics (1-month, 3-month, 12-month)
- **LLM-Ready Output**: Produces natural language summaries optimized for knowledge base integration
- **Robust Architecture**: Test-driven development with 90% test coverage
- **Production Ready**: Full CLI interface with structured logging and error handling

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Data Formats](#-data-formats)
- [Development](#-development)
- [Performance](#-performance)
- [Testing](#-testing)
- [Contributing](#-contributing)

## ⚡ Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd data-quality-summarizer

# Install with dependencies
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -e ".[dev]"

# Run with sample data
python -m src.data_quality_summarizer sample_input.csv sample_rules.json

# View results
cat resources/artifacts/full_summary.csv
cat resources/artifacts/nl_all_rows.txt
```

## 🛠 Installation

### Requirements
- **Python**: 3.11 or higher
- **Memory**: 1GB+ RAM recommended
- **Storage**: 100MB+ available space

### Install from Source

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install package with all dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Dependencies

**Core Dependencies:**
- `pandas>=2.0.0` - Data processing and CSV handling
- `structlog>=23.0.0` - Structured logging
- `psutil>=5.9.0` - Memory monitoring

**Development Dependencies:**
- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage reporting
- `black>=23.0.0` - Code formatting
- `flake8>=6.0.0` - Linting
- `mypy>=1.0.0` - Type checking

## 📖 Usage

### How to Run the Application

This application provides multiple ways to run the data quality summarizer with both core data processing and advanced ML pipeline capabilities:

#### 1. Core Data Summarization (Primary Module)

```bash
# Basic usage - Process CSV file with rule metadata
python -m src.data_quality_summarizer <csv_file> <rule_metadata_file>

# With custom chunk size for memory optimization
python -m src.data_quality_summarizer input.csv rules.json --chunk-size 50000

# With custom output directory
python -m src.data_quality_summarizer input.csv rules.json --output-dir /custom/path

# Performance monitoring with detailed logging
python -m src.data_quality_summarizer input.csv rules.json 2>&1 | tee processing.log
```

#### 2. ML Pipeline Commands

```bash
# Train ML model for predictive data quality
python -m src.data_quality_summarizer train-model input.csv rule_metadata.json --output-model model.pkl

# Make single prediction
python -m src.data_quality_summarizer predict --model model.pkl --dataset-uuid uuid123 --rule-code R001 --date 2024-01-15

# Batch predictions from CSV input
python -m src.data_quality_summarizer batch-predict --model model.pkl --input predictions.csv --output results.csv

# Validate existing model performance
python -m src.data_quality_summarizer validate-model --model model.pkl --test-data test.csv
```

#### 3. Alternative Entry Points

```bash
# Direct module execution (equivalent to primary)
python -m src.data_quality_summarizer <csv_file> <rule_metadata_file>

# Using the main module explicitly
python -m src.data_quality_summarizer.__main__ input.csv rules.json

# Running with Python interpreter
python src/data_quality_summarizer/__main__.py input.csv rules.json
```

### Command Line Interface

**Core Summarization Arguments:**
- `csv_file` - Path to input CSV containing data quality results
- `rule_metadata_file` - Path to JSON file with rule definitions

**Core Options:**
- `--chunk-size N` - Rows per processing chunk (default: 20000)
- `--output-dir PATH` - Output directory (default: resources/artifacts)

**ML Pipeline Options:**
- `--output-model PATH` - Path to save trained model (default: model.pkl)
- `--model PATH` - Path to trained model for predictions
- `--dataset-uuid UUID` - Dataset identifier for single predictions
- `--rule-code CODE` - Rule code for single predictions  
- `--date YYYY-MM-DD` - Business date for single predictions
- `--input PATH` - Input CSV for batch predictions
- `--output PATH` - Output file for batch predictions
- `--test-data PATH` - Test data for model validation

### Usage Examples

```bash
# Basic data quality summarization
python -m src.data_quality_summarizer input.csv rules.json

# Large file processing with custom chunk size
python -m src.data_quality_summarizer large_data.csv rules.json --chunk-size 50000

# Training ML model for predictive analytics
python -m src.data_quality_summarizer train-model input.csv rules.json --output-model quality_model.pkl

# Making single prediction
python -m src.data_quality_summarizer predict --model quality_model.pkl --dataset-uuid dataset123 --rule-code R001 --date 2024-01-15

# Batch predictions for multiple datasets
python -m src.data_quality_summarizer batch-predict --model quality_model.pkl --input batch_input.csv --output predictions.csv

# Model validation and performance metrics
python -m src.data_quality_summarizer validate-model --model quality_model.pkl --test-data validation.csv

# Development mode with detailed logging
export LOG_LEVEL=DEBUG
python -m src.data_quality_summarizer input.csv rules.json
```

### Sample Files

The repository includes sample data for testing:

- `sample_input.csv` - Sample data quality results (12 rows)
- `sample_rules.json` - Rule metadata definitions (4 rules)

## 🏗 Architecture

### System Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Input     │ => │  Streaming       │ => │   Summary       │
│  (100k+ rows)   │    │  Aggregation     │    │  Artifacts      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Chunked Reader  │    │ Time Windows     │    │ CSV + NL Text   │
│ (20k chunks)    │    │ (1m/3m/12m)      │    │ (LLM-ready)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

#### 1. **Ingestion** (`src/ingestion.py`)
- Chunked CSV reading using pandas
- Configurable chunk size (default: 20k rows)
- Memory-efficient data type inference prevention

#### 2. **Rules Management** (`src/rules.py`)
- JSON-based rule metadata loading
- Rule validation and enrichment
- Missing rule code handling with warnings

#### 3. **Streaming Aggregation** (`src/aggregator.py`)
- Real-time row-by-row processing
- Composite key grouping: `(source, tenant_id, dataset_uuid, dataset_name, rule_code)`
- Rolling time window calculations from latest business_date

#### 4. **Summary Generation** (`src/summarizer.py`)
- Structured CSV export (27 columns)
- Natural language sentence generation
- LLM-optimized formatting

#### 5. **CLI Orchestration** (`src/__main__.py`)
- Complete pipeline coordination
- Comprehensive error handling
- Performance monitoring and reporting

### Data Flow Pipeline

1. **Chunked Ingestion**: Read CSV in 20k-row chunks
2. **Row Processing**: Stream each row through aggregation engine
3. **Metrics Calculation**: Compute pass/fail counts across time windows
4. **Rule Enrichment**: Add metadata from rule definitions
5. **Artifact Export**: Generate CSV and natural language outputs

## 📊 Data Formats

### Input CSV Schema

Required columns for data quality results:

| Column | Type | Description |
|--------|------|-------------|
| `source` | string | Data source system identifier |
| `tenant_id` | string | Tenant/organization identifier |
| `dataset_uuid` | string | Unique dataset identifier |
| `dataset_name` | string | Human-readable dataset name |
| `business_date` | date | Business date (ISO format: YYYY-MM-DD) |
| `rule_code` | string | Rule identifier (links to metadata) |
| `results` | JSON string | Contains `{"result": "Pass"}` or `{"result": "Fail"}` |
| `level_of_execution` | string | Execution context (DATASET/ATTRIBUTE) |
| `attribute_name` | string | Column name (for ATTRIBUTE rules) |
| `dataset_record_count` | integer | Total dataset size |
| `filtered_record_count` | integer | Records evaluated by rule |

### Rule Metadata JSON Schema

```json
{
  "rule_code": {
    "rule_name": "DESCRIPTIVE_NAME",
    "rule_type": "DATASET|ATTRIBUTE", 
    "dimension": "Completeness|Validity|Timeliness|Consistency",
    "rule_description": "Human-readable description",
    "category": 1
  }
}
```

### Output Artifacts

#### 1. **Structured CSV** (`resources/artifacts/full_summary.csv`)

27-column summary with comprehensive metrics:

- **Identity**: source, tenant_id, dataset_uuid, dataset_name, rule_code
- **Metadata**: rule_name, rule_type, dimension, rule_description, category  
- **Latest Values**: business_date_latest, dataset_record_count_latest, filtered_record_count_latest
- **Counts**: pass_count_total, fail_count_total, pass_count_1m, fail_count_1m, pass_count_3m, fail_count_3m, pass_count_12m, fail_count_12m
- **Rates**: fail_rate_total, fail_rate_1m, fail_rate_3m, fail_rate_12m
- **Analysis**: trend_flag, last_execution_level

#### 2. **Natural Language** (`resources/artifacts/nl_all_rows.txt`)

LLM-optimized sentences for each summary row:

```
• On 2024-01-17, dataset "Customer_Data" under rule "ROW_COUNT_CHECK" recorded 1 failures and 2 passes overall (fail-rate 33.3%; 1-month 33.3%, 3-month 33.3%, 12-month 33.3%) — trend DEGRADING.
```

## 🔧 Development

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd data-quality-summarizer

# Setup virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### Code Quality Tools

```bash
# Format code
black src/ tests/

# Lint code  
flake8 src/ tests/

# Type checking
mypy src/

# Run all quality checks
black src/ tests/ && flake8 src/ tests/ && mypy src/
```

### Project Structure

```
data-quality-summarizer/
├── src/                    # Main package
│   ├── __init__.py
│   ├── __main__.py        # CLI entry point
│   ├── ingestion.py       # CSV reading
│   ├── rules.py          # Rule metadata
│   ├── aggregator.py     # Streaming aggregation
│   └── summarizer.py     # Output generation
├── tests/                 # Test suite
│   ├── test_*.py         # Unit tests
│   └── __init__.py
├── resources/            # Data and artifacts
│   ├── artifacts/        # Generated outputs
│   └── context/         # Documentation  
├── sample_input.csv      # Sample data
├── sample_rules.json     # Sample rules
├── pyproject.toml        # Package configuration
└── README.md            # This file
```

### Performance Guidelines

#### Memory Optimization
- **Chunk Size**: Default 20k rows balances memory vs. processing overhead
- **Data Types**: Explicit pandas dtypes prevent expensive inference
- **Streaming**: Only accumulator dictionary kept in memory (~1MB typical)

#### Performance Targets
- **Runtime**: <2 minutes for 100k rows on 4-core consumer laptop
- **Memory**: <1GB peak usage during processing
- **Output Size**: Summary CSV <2MB for typical datasets

### Logging Strategy

Structured logging with appropriate levels:

```python
# INFO: Progress indicators
logger.info(f"Processing chunk {chunk_num + 1} ({len(chunk)} rows)")

# DEBUG: Detailed metrics  
logger.debug(f"Accumulator size: {len(aggregator.accumulator)} keys")

# WARN: Recoverable issues
logger.warning(f"Rule metadata not found for rule_code: {rule_code}")

# ERROR: Fatal issues
logger.error(f"Failed to read CSV file: {csv_file}")
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests with coverage
python -m pytest

# Run specific test file
python -m pytest tests/test_ingestion.py

# Run with detailed output
python -m pytest -v --tb=short

# Generate HTML coverage report
python -m pytest --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Test Structure

- **Unit Tests**: Individual module functionality
- **Integration Tests**: Complete pipeline end-to-end
- **Performance Tests**: Memory and runtime benchmarks
- **Edge Case Tests**: Error handling and validation

### Test Coverage

Current coverage: **90%** across all modules

| Module | Coverage | Key Test Areas |
|--------|----------|----------------|
| `ingestion.py` | 95% | Chunk processing, file validation |
| `aggregator.py` | 92% | Streaming aggregation, time windows |
| `rules.py` | 88% | Metadata loading, validation |
| `summarizer.py` | 90% | CSV export, NL generation |
| `__main__.py` | 85% | CLI integration, error handling |

### Adding Tests

```python
# tests/test_new_feature.py
import pytest
from src.new_module import NewClass

class TestNewFeature:
    def test_basic_functionality(self):
        """Test basic functionality."""
        instance = NewClass()
        result = instance.process()
        assert result is not None
    
    def test_error_handling(self):
        """Test error conditions."""
        instance = NewClass()
        with pytest.raises(ValueError):
            instance.process(invalid_input)
```

## 📈 Performance

### Benchmarks

Tested on **4-core consumer laptop** (8GB RAM):

| Dataset Size | Processing Time | Memory Peak | Output Size |
|-------------|-----------------|-------------|-------------|
| 10k rows | 8 seconds | 120 MB | 0.5 MB |
| 50k rows | 35 seconds | 450 MB | 1.2 MB |
| 100k rows | 68 seconds | 850 MB | 2.1 MB |
| 500k rows | 5.2 minutes | 980 MB | 8.7 MB |

### Performance Tuning

#### Memory Optimization
```bash
# Reduce chunk size for memory-constrained systems
python -m src.data_quality_summarizer input.csv rules.json --chunk-size 10000

# Monitor memory usage with structured logging
export LOG_LEVEL=DEBUG
python -m src.data_quality_summarizer input.csv rules.json
```

#### Processing Speed
```bash
# Increase chunk size for faster processing (requires more RAM)
python -m src.data_quality_summarizer input.csv rules.json --chunk-size 50000

# Use SSD storage for better I/O performance
# Process files locally rather than network drives
```

### Monitoring

The CLI provides comprehensive performance metrics:

```
🎉 SUCCESS: Data Quality Summarizer completed!
   📊 Processed: 100,000 rows
   🔑 Unique keys: 1,250
   ⏱️  Time: 68.34 seconds  
   💾 Memory peak: 847.3 MB
   📁 Output files:
      • resources/artifacts/full_summary.csv
      • resources/artifacts/nl_all_rows.txt
```

## 🤝 Contributing

### Development Workflow

1. **Setup Environment**
   ```bash
   git clone <repository-url>
   cd data-quality-summarizer
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Follow TDD Approach**
   - Write tests first
   - Implement functionality
   - Ensure all tests pass
   - Maintain >80% coverage

4. **Code Quality Checks**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   python -m pytest
   ```

5. **Submit Pull Request**
   - Include tests for new functionality
   - Update documentation if needed
   - Ensure CI passes

### Code Style Guidelines

- **Line Length**: 88 characters (Black default)
- **Imports**: Group stdlib, third-party, local imports
- **Typing**: Full type annotations required (mypy strict)
- **Docstrings**: Google-style for classes and functions
- **Comments**: Explain "why" not "what"

### File Size Limits

**Critical Rule**: No file should exceed **800 lines**

- **Functions**: 30-50 lines recommended, 80 lines maximum
- **Classes**: 200-300 lines recommended  
- **Files**: 500-800 lines recommended

Break large files into logical modules when approaching limits.

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 📞 Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/your-org/data-quality-summarizer/issues)
- **Documentation**: [Project Wiki](https://github.com/your-org/data-quality-summarizer/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/data-quality-summarizer/discussions)

## 🔄 Changelog

### v0.1.0 (Current)
- ✅ Complete streaming aggregation pipeline
- ✅ Time-window analytics (1m/3m/12m)
- ✅ LLM-optimized natural language output
- ✅ Full CLI interface with comprehensive logging
- ✅ 90% test coverage across all modules
- ✅ Production-ready performance and error handling

---

**Built with ❤️ for data quality excellence**