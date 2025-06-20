# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Data Quality Summarizer** - an offline data processing system that ingests large CSV files (~100k rows) containing data quality check results and produces LLM-optimized summary artifacts. The system is designed for consumer-grade machines (<8GB RAM) and follows a strict test-driven development approach.

## Architecture

The system implements a **streaming aggregation pipeline** with the following key components:

### Core Modules (src/)
- `ingestion.py` - Chunked CSV reader using pandas with configurable chunk size (20k default)
- `aggregator.py` - Streaming aggregation engine with rolling time windows
- `rules.py` - Rule metadata loader and validation utilities  
- `summarizer.py` - CSV and natural language artifact generation
- `__main__.py` - CLI entry point and pipeline orchestration

### Data Flow
1. **Chunked Ingestion**: Reads CSV in 20k-row chunks to maintain <1GB memory usage
2. **Streaming Aggregation**: Groups by `(source, tenant_id, dataset_uuid, dataset_name, rule_code)` 
3. **Time Window Analysis**: Calculates rolling metrics for 1-month, 3-month, 12-month periods
4. **Artifact Generation**: Produces structured CSV and natural language summaries

### Key Data Schemas

**Input CSV Columns**:
- `source`, `tenant_id`, `dataset_uuid`, `dataset_name` - Identity fields
- `business_date` - ISO date for time-based aggregations
- `rule_code` - Links to rule metadata mapping
- `results` - JSON string containing pass/fail status
- `level_of_execution`, `attribute_name` - Execution context

**Output Artifacts**:
- `resource/artifacts/full_summary.csv` - 27-column structured summary
- `resource/artifacts/nl_all_rows.txt` - Natural language sentences for LLM consumption

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies (once pyproject.toml is created)
pip install -e .
```

### Testing
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_ingestion.py
```

### Code Quality
```bash
# Format code (if black is configured)
black src/ tests/

# Type checking (if mypy is configured) 
mypy src/

# Linting (if flake8/ruff is configured)
flake8 src/
# or
ruff check src/
```

### Running the Application
```bash
# Run the summarizer (once implemented)
python -m src input.csv rule_metadata.json

# With custom chunk size
python -m src input.csv rule_metadata.json --chunk-size 50000
```

## Development Guidelines

### Performance Requirements
- **Runtime**: <2 minutes for 100k rows on 4-core laptop
- **Memory**: <1GB peak usage
- **Output Size**: Summary CSV <2MB typical

### Test-Driven Development Stages
The project follows a 5-stage TDD approach:
1. **Stage 1**: Core infrastructure & data ingestion
2. **Stage 2**: Rule metadata management
3. **Stage 3**: Streaming aggregation engine  
4. **Stage 4**: Summary generation & export
5. **Stage 5**: CLI integration & end-to-end testing

### File Size Limits
- **Critical**: No file should exceed 800 lines - break into multiple files if needed
- Target function size: 30-50 lines (max 80)
- Target class size: 200-300 lines

### Logging Strategy
Use structured logging with these levels:
- **INFO**: Chunk processing progress, accumulator size
- **DEBUG**: Sample accumulator entries, detailed metrics
- **WARN**: Unrecognized rule codes, malformed JSON results
- **ERROR**: File I/O failures, validation errors

### Memory Optimization
- Use pandas `dtype` mapping to prevent expensive type inference
- Keep only accumulator dictionary in memory (~20 datasets × rules ≪ 1MB)
- Process data in streaming fashion, not loading entire dataset

## Key Design Patterns

### Streaming Aggregation
The core pattern uses an accumulator with composite keys:
```python
# Key: (source, tenant_id, dataset_uuid, dataset_name, rule_code)
# Value: Metrics object with pass/fail counts across time windows
```

### Time Window Calculations
All time windows are calculated from the **latest business_date** in the file:
- 1-month: 30 days back from latest date
- 3-month: 90 days back from latest date  
- 12-month: 365 days back from latest date

### Natural Language Generation
Each summary row generates an LLM-optimized sentence following this template:
```
• On {date}, dataset "{name}" under rule "{rule}" recorded {failures} failures and {passes} passes overall (fail-rate {rate}; 1-month {rate_1m}, 3-month {rate_3m}, 12-month {rate_12m}) — trend {trend}.
```

## Testing Strategy

### Coverage Requirements
- Minimum 80% test coverage across all modules
- Unit tests for each module's core functionality
- Integration tests for complete pipeline
- Performance benchmarks for memory and runtime requirements

### Test Data Patterns
- Use small synthetic datasets for unit tests
- Create fixtures for various edge cases (missing rule codes, malformed JSON)
- Mock large datasets for performance testing
- Validate output format exactly matches schema requirements

## Common Pitfalls to Avoid

- **Memory Issues**: Never load entire CSV into memory - always use chunked processing
- **Type Inference**: Always specify pandas dtypes to prevent memory bloat
- **Time Calculations**: Always calculate windows from latest business_date, not current date
- **Rule Validation**: Handle missing rule codes gracefully with warnings, don't fail hard
- **Output Format**: Natural language template must match exactly for LLM consumption