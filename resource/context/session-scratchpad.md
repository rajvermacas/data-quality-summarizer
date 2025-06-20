# Session Context: Data Quality Summarizer

## Project Overview
**Goal**: Build an offline data quality summarizer that processes 100k-row CSV files and generates knowledge base artifacts for LLM chatbot integration.

## Key Requirements
- **Input**: CSV with 12 columns (source, tenant_id, dataset_uuid, etc.) + rule metadata JSON
- **Output**: 27-column summary CSV + natural language text file
- **Performance**: <2min runtime, <1GB memory, <2MB output
- **Location**: Artifacts saved to `resource/artifacts/`

## Development Approach
- **Method**: 5-stage Test-Driven Development (TDD)
- **Coverage**: >80% test coverage required
- **Architecture**: Modular design with src/ folder structure

## Current Status
✅ **COMPLETED**:
- Read and analyzed PRD requirements
- Created comprehensive 5-stage TDD plan
- Saved development plan to `resource/development_plan/5-stage-tdd-plan.md`
- Plan approved by user

🔄 **NEXT STEPS**:
- Begin Stage 1: Core Infrastructure & Data Ingestion
- Set up Python venv and project structure
- Implement chunked CSV reader with tests

## Key Technical Details

### Input Schema (12 columns):
- source, tenant_id, dataset_uuid, dataset_name, business_date
- dataset_record_count, rule_code, level_of_execution, attribute_name
- results (JSON), context_id, filtered_record_count

### Output Schema (27 columns):
- Core: source, tenant_id, dataset_uuid, dataset_name, rule_code
- Enriched: rule_name, rule_type, dimension, rule_description, category
- Metrics: pass/fail counts (total, 1m, 3m, 12m), fail rates, trend_flag
- Meta: business_date_latest, dataset_record_count_latest, last_execution_level

### Architecture:
```
src/
  ingestion.py     # Chunked CSV reader
  aggregator.py    # Streaming aggregation
  rules.py         # Rule metadata management
  summarizer.py    # CSV + NL export
  __main__.py      # CLI entry point
```

### Processing Pipeline:
1. Chunked ingestion (20k rows/chunk)
2. Streaming aggregation by key: (source, tenant_id, dataset_uuid, dataset_name, rule_code)
3. Rolling window calculations (30, 90, 365 days from latest business_date)
4. Rule metadata enrichment
5. Export to CSV + natural language artifacts

## Development Plan Stages:
1. **Stage 1**: Core Infrastructure & Data Ingestion
2. **Stage 2**: Rule Metadata Management  
3. **Stage 3**: Streaming Aggregation Engine
4. **Stage 4**: Summary Generation & Export
5. **Stage 5**: CLI Integration & End-to-End Testing

## Critical Constraints:
- File size limit: 800 lines per file (break into multiple if exceeded)
- Memory efficiency: Use pandas dtype mapping, chunked processing
- Error handling: Robust logging with structlog
- TDD mandatory: RED → GREEN → REFACTOR cycle for all features