# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Data Quality Summarizer with ML Pipeline and React UI** - an offline data processing system that ingests large CSV files (~100k rows) containing data quality check results and produces both LLM-optimized summary artifacts and predictive ML models. The system combines traditional data summarization with machine learning capabilities and a modern React web interface for user interaction. Designed for consumer-grade machines (<8GB RAM) and follows a strict test-driven development approach.

## Architecture

The system implements a **streaming aggregation pipeline** with the following key components:

### Core Modules (src/data_quality_summarizer/)
- `ingestion.py` - Chunked CSV reader using pandas with configurable chunk size (20k default)
- `aggregator.py` - Streaming aggregation engine with rolling time windows
- `rules.py` - Rule metadata loader and validation utilities  
- `summarizer.py` - CSV and natural language artifact generation
- `__main__.py` - CLI entry point and pipeline orchestration
- `constants.py` - System-wide constants and rule categories

### ML Pipeline Modules (src/data_quality_summarizer/ml/)
- `feature_engineering.py` - Feature extraction from aggregated data
- `model_training.py` - LightGBM model training with cross-validation
- `model_validation.py` - Model evaluation and performance metrics
- `prediction_service.py` - Real-time prediction API and batch processing
- `data_loader.py` - ML-specific data loading and preprocessing
- `model_registry.py` - Model versioning and persistence management
- `config.py` - ML pipeline configuration and hyperparameters
- `preprocessing.py` - Data transformation and feature scaling
- `evaluation.py` - Model performance assessment utilities
- `cli.py` - ML command-line interface
- `batch_prediction.py` - Batch prediction processing
- `exceptions.py` - Custom ML pipeline exceptions
- `utils.py` - ML utility functions and helpers

### React UI Modules (src/data_quality_summarizer/ui/)
- `App.tsx` - Main React application with routing and state management
- `main.tsx` - React application entry point
- `backend_integration.py` - FastAPI server providing REST API for UI
- `pages/` - React page components (FileUpload, Processing, Results, MLPipeline)
- `components/` - Reusable React components (DataTable, FileUpload, ProgressBar)
- `visualizations/` - Chart components using Recharts library
- `types/common.ts` - TypeScript type definitions
- `index.css` - Global styles and CSS

### Data Flow
1. **Chunked Ingestion**: Reads CSV in 20k-row chunks to maintain <1GB memory usage
2. **Streaming Aggregation**: Groups by `(source, tenant_id, dataset_uuid, dataset_name, rule_code)` 
3. **Time Window Analysis**: Calculates rolling metrics for 1-month, 3-month, 12-month periods
4. **Artifact Generation**: Produces structured CSV and natural language summaries
5. **ML Pipeline**: Optionally trains LightGBM models for predictive data quality monitoring
6. **Web UI Integration**: FastAPI backend serves React frontend with REST API endpoints

### Key Data Schemas

**Input CSV Columns**:
- `source`, `tenant_id`, `dataset_uuid`, `dataset_name` - Identity fields
- `business_date` - ISO date for time-based aggregations
- `rule_code` - Links to rule metadata mapping
- `results` - JSON string containing pass/fail status
- `level_of_execution`, `attribute_name` - Execution context

**Output Artifacts**:
- `resources/artifacts/full_summary.csv` - 27-column structured summary
- `resources/artifacts/nl_all_rows.txt` - Natural language sentences for LLM consumption

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install project in development mode with all dependencies
pip install -e .

# Install development dependencies for testing/linting
pip install -e ".[dev]"

# Install Node.js dependencies for React UI
npm install
```

### Web UI Commands
```bash
# Start production server (recommended for most users)
python scripts/start_ui_server.py

# Start development servers (React dev server + FastAPI backend)
python scripts/start_ui_server.py dev

# Build React app for production
npm run build

# Start only React development server (port 3000)
npm run dev

# Preview production build
npm run preview
```

### Testing
```bash
# Run all Python tests with coverage (configured in pyproject.toml)
python -m pytest

# Run with detailed coverage report
python -m pytest --cov=src --cov-report=term-missing --cov-report=html

# Run specific test file
python -m pytest tests/test_ingestion.py

# Run ML pipeline tests specifically
python -m pytest tests/test_ml/

# Run UI backend tests
python -m pytest tests/test_ui/

# View HTML coverage report (opens in browser)
open htmlcov/index.html

# Note: No frontend JS/TS tests currently configured
```

### Code Quality
```bash
# Python formatting
black src/ tests/

# Python linting  
flake8 src/ tests/

# Python type checking
mypy src/

# Run all Python quality checks
black src/ tests/ && flake8 src/ tests/ && mypy src/

# TypeScript type checking
npx tsc --noEmit
```

### Running the Application

#### Web UI (Recommended for Most Users)
```bash
# Complete end-to-end web interface
python scripts/start_ui_server.py

# Access at: http://localhost:8000
# Features: File upload, real-time processing, data visualization, ML pipeline
```

#### Core Data Summarization (Primary Module)
```bash
# Basic usage - Process CSV file with rule metadata
python -m src.data_quality_summarizer <csv_file> <rule_metadata_file>

# With custom chunk size for memory optimization
python -m src.data_quality_summarizer input.csv rules.json --chunk-size 50000

# With custom output directory
python -m src.data_quality_summarizer input.csv rules.json --output-dir /custom/path

# Performance monitoring with detailed logging
python -m src.data_quality_summarizer input.csv rules.json 2>&1 | tee processing.log

# Quick start with sample data
python -m src sample_input.csv sample_rules.json
```

#### ML Pipeline Commands
```bash
# Train ML model for predictive data quality
python -m src train-model input.csv rule_metadata.json --output-model model.pkl

# Make single prediction
python -m src predict --model model.pkl --dataset-uuid uuid123 --rule-code R001 --date 2024-01-15

# Batch predictions from CSV input
python -m src batch-predict --model model.pkl --input predictions.csv --output results.csv

# Validate existing model performance
python -m src validate-model --model model.pkl --test-data test.csv
```

#### Alternative Entry Points
```bash
# Direct module execution (equivalent to primary)
python -m src <csv_file> <rule_metadata_file>

# Using the main module explicitly
python -m src.data_quality_summarizer.__main__ input.csv rules.json

# Running with Python interpreter
python src/data_quality_summarizer/__main__.py input.csv rules.json
```

#### Development Mode and Logging
```bash
# Development mode with detailed logging
export LOG_LEVEL=DEBUG
python -m src.data_quality_summarizer input.csv rules.json

# Monitor memory usage with structured logging at DEBUG level
LOG_LEVEL=DEBUG python -m src.data_quality_summarizer input.csv rules.json
```

## Project Status

This project is **production-ready** with all planned features implemented. Key metrics:
- **Test Coverage**: 86% across all modules (302 test cases, 14 test files)
- **Code Quality**: All files under 800-line limit, strict typing with mypy
- **Performance**: Meets all benchmarks (<2min runtime, <1GB memory for 100k rows)
- **Architecture**: Clean modular design with streaming aggregation + ML pipeline
- **ML Capabilities**: LightGBM-based predictive modeling with CLI and batch processing

## Development Guidelines

### Dual Architecture (Python + React)
The project maintains a clean separation between:
- **Backend**: Python-based data processing, ML pipeline, and FastAPI server
- **Frontend**: React/TypeScript UI with Vite build system
- **Integration**: REST API endpoints connecting UI to backend services
- **Development**: Both can be developed independently or together

### Performance Requirements
- **Runtime**: <2 minutes for 100k rows on 4-core laptop
- **Memory**: <1GB peak usage
- **Output Size**: Summary CSV <2MB typical
- **UI Response**: <2 seconds for typical file uploads
- **API Endpoints**: <5 seconds for processing requests

### Test-Driven Development Stages
The project follows a 5-stage TDD approach (all stages completed):
1. **Stage 1**: Core infrastructure & data ingestion ✅
2. **Stage 2**: Rule metadata management ✅  
3. **Stage 3**: Streaming aggregation engine ✅
4. **Stage 4**: ML pipeline with LightGBM training and prediction ✅
5. **Stage 5**: CLI integration and batch processing ✅

Current test coverage: 86% across all modules with comprehensive ML pipeline testing.

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

## Troubleshooting

### Common Issues
- **ModuleNotFoundError**: Ensure virtual environment is activated and dependencies installed
- **Memory Errors**: Reduce chunk size with `--chunk-size` parameter (default: 20000)
- **Test Failures**: Run `python -m pytest -v` for detailed test output
- **Type Errors**: Run `mypy src/` to check type annotations
- **Performance Issues**: Monitor memory usage with structured logging at DEBUG level

### Development Dependencies
All development tools are configured in `pyproject.toml` and `package.json`:

**Python Dependencies:**
- **Testing**: pytest with coverage reporting
- **Formatting**: black with 88-character line length  
- **Linting**: flake8 with E203/W503 exceptions
- **Type Checking**: mypy with strict configuration
- **ML Libraries**: LightGBM, scikit-learn for machine learning pipeline
- **Data Processing**: pandas, numpy for data manipulation
- **Web Framework**: FastAPI, uvicorn for REST API server

**Node.js Dependencies:**
- **Build System**: Vite with React plugin
- **Framework**: React 19.1.0 with TypeScript
- **UI Components**: Recharts for data visualization
- **Icons**: Lucide React for modern icons
- **Type Checking**: TypeScript with strict configuration

## Key Architectural Patterns

### FastAPI Backend Integration
The React UI connects to the Python backend through REST API endpoints:
- `/api/process` - Upload and process CSV/rules files
- `/api/ml/train` - Train ML models from processed data
- `/api/ml/predict` - Single predictions
- `/api/ml/batch-predict` - Batch predictions
- `/api/health` - Health check endpoint

### React State Management
The UI uses simple React state with TypeScript:
- `App.tsx` manages global state and routing
- Page components handle local state
- Form data and files processed via FormData API
- Real-time progress tracking during processing

### Build and Deployment Architecture
- **Development**: Vite dev server (port 3000) + FastAPI (port 8000)
- **Production**: FastAPI serves built React static files
- **Proxy**: React dev server proxies `/api` requests to FastAPI
- **Static Files**: Built React app served from `dist/ui/`

## Memories

- Remember these commands to run the application:
  - **Web UI**: `python scripts/start_ui_server.py` (recommended)
  - **CLI Demo**: `source venv/bin/activate && python scripts/demo_predictions.py`
  - **Development**: `python scripts/start_ui_server.py dev` (for React/FastAPI dev)
- The UI command is: `npm run dev` (React only) or use the Python script for full stack