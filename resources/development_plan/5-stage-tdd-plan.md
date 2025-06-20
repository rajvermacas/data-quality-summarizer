# 5-Stage Test-Driven Development Plan: Data Quality Summarizer

## Stage 1: Core Infrastructure & Data Ingestion
**Focus**: Establish chunked CSV reading and basic data structures

### Input Schema Reference:
| Column | Type | Notes |
|--------|------|-------|
| `source` | string | Data source system |
| `tenant_id` | string | Tenant / client identifier |
| `dataset_uuid` | string | Stable dataset key |
| `dataset_name` | string | Human-readable name |
| `business_date` | string (ISO) | Row-level execution date |
| `dataset_record_count` | number | Total rows in dataset (raw) |
| `rule_code` | number | Foreign key to rule-metadata mapping |
| `level_of_execution` | string | ATTRIBUTE / DATASET etc. |
| `attribute_name` | string | Populated when `level_of_execution = ATTRIBUTE` |
| `results` | JSON (string) | e.g. `{ "result":"Pass" , ... }` |
| `context_id` | string | Upstream batch context |
| `filtered_record_count` | number | Rows considered after filters |

### TDD Approach:
- **RED**: Write tests for chunked CSV reader that fails when module doesn't exist
- **GREEN**: Implement minimal ingestion.py with pandas chunk reader
- **REFACTOR**: Optimize memory usage and error handling

### Key Components:
- `src/ingestion.py` - Chunked CSV reader with configurable chunk size (20k default)
- Data type mapping for efficient memory usage
- Basic error handling for malformed CSV files
- Logging setup with structured logging

### Acceptance Criteria:
- [x] Can read CSV files in chunks of 20,000 rows
- [x] Memory usage stays under 200MB during ingestion
- [x] Proper dtype mapping prevents expensive type inference
- [x] Handles malformed CSV gracefully with appropriate logging
- [x] Unit tests achieve >80% coverage (achieved 91%)
- [x] All tests pass (RED → GREEN → REFACTOR cycle complete)

**✅ STAGE 1 COMPLETED** - Code review passed with excellent quality standards

---

## Stage 2: Rule Metadata Management
**Focus**: Rule-metadata loading and validation system

### Rule Metadata Schema Reference:
Keyed by `rule_code`, provides:
- `rule_name` - e.g. "ROW_COUNT"
- `rule_type` - DATASET / ATTRIBUTE
- `dimension` - Correctness, etc.
- `rule_description` - Verbose description
- `category` - Category 1–4

### TDD Approach:
- **RED**: Write tests for rule metadata loader and validation
- **GREEN**: Implement rules.py with JSON loading and lookup functions
- **REFACTOR**: Add caching and performance optimizations

### Key Components:
- `src/rules.py` - Rule metadata loader and utilities
- JSON rule metadata validation
- Rule code lookup and enrichment functions
- Error handling for missing/invalid rule codes

### Acceptance Criteria:
- [x] Successfully loads and validates rule metadata JSON
- [x] Provides fast lookup by rule_code
- [x] Handles missing rule codes with appropriate warnings
- [x] Validates rule metadata structure (rule_name, rule_type, dimension, etc.)
- [x] Unit tests cover all rule metadata scenarios
- [x] All tests pass with >80% coverage (achieved 96%)

**✅ STAGE 2 COMPLETED** - Code review passed with excellent quality standards

---

## Stage 3: Streaming Aggregation Engine
**Focus**: Core aggregation logic with rolling time windows

### Aggregation Key Schema:
- Accumulator key: `(source, tenant_id, dataset_uuid, dataset_name, rule_code)`
- Rolling windows from **latest business_date**: 1-month (30 days), 3-month (90 days), 12-month (365 days)
- Trend calculation: `fail_rate_1m` vs `fail_rate_3m` with epsilon threshold

### TDD Approach:
- **RED**: Write tests for aggregation accumulator and time window calculations
- **GREEN**: Implement aggregator.py with streaming aggregation
- **REFACTOR**: Optimize performance and add trend calculations

### Key Components:
- `src/aggregator.py` - Streaming aggregation and metrics calculation
- Accumulator with composite key
- Rolling window calculations (1m, 3m, 12m from latest business_date)
- Trend flag computation (↑/↓/= based on fail rates)
- Pass/fail count tracking from JSON results

### Acceptance Criteria:
- [x] Correctly aggregates data by the composite key
- [x] Accurately calculates rolling window metrics (30, 90, 365 days)
- [x] Computes fail rates for all time periods
- [x] Determines trend flags based on 1m vs 3m comparison
- [x] Handles edge cases (no data in windows, malformed JSON)
- [x] Memory usage remains under 50MB for accumulator
- [x] Unit tests validate all aggregation scenarios
- [x] All tests pass with >80% coverage (achieved 93%)

**✅ STAGE 3 COMPLETED** - Code review passed with excellent quality standards

---

## Stage 4: Summary Generation & Export
**Focus**: CSV and natural language artifact generation

### Output Schema Reference - `full_summary.csv`:
| # | Column | Description |
|---|---------|-------------|
| 1 | `source` | Data source system |
| 2 | `tenant_id` | Tenant identifier |
| 3 | `dataset_uuid` | Dataset UUID |
| 4 | `dataset_name` | Dataset name |
| 5 | `rule_code` | Rule identifier |
| 6 | `rule_name` | e.g. *ROW_COUNT* |
| 7 | `rule_type` | DATASET / ATTRIBUTE |
| 8 | `dimension` | Correctness, etc. |
| 9 | `rule_description` | Verbose description |
| 10 | `category` | Category 1–4 |
| 11 | `business_date_latest` | Max `business_date` seen for this key |
| 12 | `dataset_record_count_latest` | From latest row |
| 13 | `filtered_record_count_latest` | From latest row |
| 14 | `pass_count_total` | Cumulative passes |
| 15 | `fail_count_total` | Cumulative fails |
| 16 | `pass_count_1m` | Passes in last 30 days |
| 17 | `fail_count_1m` | Fails in last 30 days |
| 18 | `pass_count_3m` | Passes in last 90 days |
| 19 | `fail_count_3m` | Fails in last 90 days |
| 20 | `pass_count_12m` | Passes in last 365 days |
| 21 | `fail_count_12m` | Fails in last 365 days |
| 22 | `fail_rate_total` | `fail_count_total / (pass+fail)_total` |
| 23 | `fail_rate_1m` | Analogous |
| 24 | `fail_rate_3m` | Analogous |
| 25 | `fail_rate_12m` | Analogous |
| 26 | `trend_flag` | ↑ if `fail_rate_1m` > `fail_rate_3m` + ε; ↓ if < −ε; = otherwise |
| 27 | `last_execution_level` | Most common `level_of_execution` for key |

### NL Sentence Template:
```
• On {business_date_latest}, dataset "{dataset_name}" (source: {source}, tenant: {tenant_id}, UUID: {dataset_uuid}) under rule "{rule_name}" [{rule_code}] recorded {fail_count_total} failures and {pass_count_total} passes overall (fail-rate {fail_rate_total:.2%}; 1-month {fail_rate_1m:.2%}, 3-month {fail_rate_3m:.2%}, 12-month {fail_rate_12m:.2%}) — trend {trend_flag}.
```

### TDD Approach:
- **RED**: Write tests for CSV export and NL sentence generation
- **GREEN**: Implement summarizer.py with export functions
- **REFACTOR**: Optimize file I/O and formatting

### Key Components:
- `src/summarizer.py` - CSV and NL generation
- Full summary CSV with all 27 columns as per schema
- Natural language sentence generation per template
- File output to `resource/artifacts/` directory
- Proper formatting and encoding (UTF-8)

### Acceptance Criteria:
- [x] Generates `full_summary.csv` with exact schema (27 columns)
- [x] Creates `nl_all_rows.txt` with proper sentence formatting
- [x] Follows exact NL template with all placeholders filled
- [x] Creates output directory if it doesn't exist
- [x] Handles Unicode characters properly
- [x] Output files are under 2MB for typical datasets
- [x] Unit tests verify file content and formatting
- [x] All tests pass with >80% coverage (achieved 91%)

**✅ STAGE 4 COMPLETED** - Code review passed with excellent quality standards

---

## Stage 5: CLI Integration & End-to-End Testing
**Focus**: Command-line interface and complete pipeline integration

### CLI Interface Schema:
- Input: CSV file path (raw data quality results)
- Input: Rule metadata JSON file path
- Output: `resource/artifacts/full_summary.csv`
- Output: `resource/artifacts/nl_all_rows.txt`

### Performance Requirements:
- Runtime: <2 minutes for 100k rows on 4-core laptop
- Memory: <1GB peak usage
- Output CSV: <2MB typical size

### TDD Approach:
- **RED**: Write integration tests for complete pipeline
- **GREEN**: Implement __main__.py with CLI argument handling
- **REFACTOR**: Add performance monitoring and final optimizations

### Key Components:
- `src/__main__.py` - CLI entry point
- Argument parsing for input CSV and rule metadata files
- Complete pipeline orchestration
- Performance monitoring and reporting
- Final error handling and logging

### Acceptance Criteria:
- [x] CLI accepts CSV file and rule metadata JSON as arguments
- [x] Complete pipeline runs end-to-end under 2 minutes for 100k rows
- [x] Memory usage stays under 1GB throughout execution
- [x] Proper progress reporting and logging at INFO level
- [x] Graceful error handling with meaningful messages
- [x] Integration tests validate complete workflow
- [x] Performance benchmarks meet requirements
- [x] All unit and integration tests pass
- [x] Code coverage exceeds 80% across all modules (achieved 90%)
- [x] Documentation updated in README.md

**✅ STAGE 5 COMPLETED** - Code review passed with exceptional quality standards

---

## Additional Requirements Met:
- Python virtual environment setup
- Comprehensive logging with structlog
- Exception handling throughout
- .gitignore and pyproject.toml creation
- File size limits enforced (no file >800 lines)
- Modular design following single responsibility principle

## Success Metrics:
- Runtime: <2 minutes for 100k rows
- Memory: <1GB peak usage
- Output: Summary CSV <2MB
- Test Coverage: >80%
- Code Quality: All linting and type checking passes