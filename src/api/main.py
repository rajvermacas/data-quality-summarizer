"""FastAPI application for Data Quality Summarizer UI."""
import os
import uuid
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from src.data_quality_summarizer.ingestion import CSVIngester
from src.data_quality_summarizer.rules import RuleMetadata
from src.data_quality_summarizer.aggregator import DataAggregator
from src.data_quality_summarizer.summarizer import SummaryGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Data Quality Summarizer API",
    description="API for processing and analyzing data quality check results",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for processing status (in production, use Redis or database)
processing_store: Dict[str, Dict[str, Any]] = {}

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=4)


class ProcessingResponse(BaseModel):
    """Response model for file upload."""
    processing_id: str
    status: str
    message: str


class ProcessingStatus(BaseModel):
    """Model for processing status."""
    processing_id: str
    status: str
    progress: int
    message: str
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ResultsSummary(BaseModel):
    """Model for results summary."""
    total_checks: int
    passed: int
    failed: int
    pass_rate: float


def process_data_quality(processing_id: str, csv_path: Path, rules_path: Path):
    """Process data quality CSV file."""
    try:
        # Update status
        processing_store[processing_id]['status'] = 'processing'
        processing_store[processing_id]['progress'] = 10
        
        # Load rule metadata
        rule_metadata = RuleMetadata(str(rules_path))
        processing_store[processing_id]['progress'] = 20
        
        # Initialize components
        ingester = CSVIngester(chunk_size=20000)
        aggregator = DataAggregator()
        
        # Process chunks
        total_rows = 0
        for i, chunk in enumerate(ingester.read_csv_chunks(str(csv_path))):
            aggregator.aggregate_chunk(chunk)
            total_rows += len(chunk)
            
            # Update progress
            progress = 20 + (60 * (i + 1) / 10)  # Assume ~10 chunks
            processing_store[processing_id]['progress'] = min(int(progress), 80)
            processing_store[processing_id]['message'] = f"Processed {total_rows:,} rows"
        
        # Get final metrics
        processing_store[processing_id]['progress'] = 85
        metrics = aggregator.get_final_metrics()
        
        # Create summarizer and generate outputs
        processing_store[processing_id]['progress'] = 90
        summarizer = SummaryGenerator(metrics, rule_metadata)
        
        # Generate outputs
        output_dir = Path(f"resources/processing/{processing_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / "summary.csv"
        nl_path = output_dir / "summary.txt"
        
        summarizer.save_csv_summary(str(csv_path))
        summarizer.save_natural_language_summary(str(nl_path))
        
        # Calculate summary statistics
        summary_stats = calculate_summary_stats(metrics)
        
        # Update completion status
        processing_store[processing_id].update({
            'status': 'completed',
            'progress': 100,
            'message': 'Processing completed successfully',
            'completed_at': datetime.now().isoformat(),
            'results': {
                'summary': summary_stats,
                'output_files': {
                    'csv': str(csv_path),
                    'text': str(nl_path)
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing {processing_id}: {str(e)}")
        processing_store[processing_id].update({
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        })


def calculate_summary_stats(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary statistics from metrics."""
    total_checks = 0
    total_passed = 0
    total_failed = 0
    
    for metric in metrics.values():
        total_checks += metric.total_count
        total_passed += metric.pass_count
        total_failed += metric.fail_count
    
    pass_rate = total_passed / total_checks if total_checks > 0 else 0
    
    return {
        'total_checks': total_checks,
        'passed': total_passed,
        'failed': total_failed,
        'pass_rate': round(pass_rate, 4)
    }


def get_processing_status(processing_id: str) -> Optional[Dict[str, Any]]:
    """Get processing status by ID."""
    return processing_store.get(processing_id)


def get_processing_results(processing_id: str) -> Optional[Dict[str, Any]]:
    """Get processing results."""
    status = processing_store.get(processing_id)
    if not status or status['status'] != 'completed':
        return None
    
    # Load and parse results
    results = status.get('results', {})
    metrics = results.get('summary', {})
    
    # Calculate trends (mock data for now)
    return {
        'summary': metrics,
        'by_rule': [
            {'rule_code': 'R001', 'pass_rate': 0.98},
            {'rule_code': 'R002', 'pass_rate': 0.92}
        ],
        'trends': {
            '1_month': {'pass_rate': 0.94},
            '3_month': {'pass_rate': 0.93},
            '12_month': {'pass_rate': 0.95}
        }
    }


def export_results_csv(processing_id: str) -> Optional[str]:
    """Export results as CSV."""
    status = processing_store.get(processing_id)
    if not status or status['status'] != 'completed':
        return None
    
    csv_path = status['results']['output_files']['csv']
    if Path(csv_path).exists():
        return Path(csv_path).read_text()
    
    return None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Data Quality Summarizer API",
        "version": "0.1.0",
        "endpoints": [
            "/api/upload",
            "/api/processing/{processing_id}",
            "/api/results/{processing_id}",
            "/api/export/{processing_id}"
        ]
    }


@app.post("/api/upload", response_model=ProcessingResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    rules: UploadFile = File(...)
):
    """Upload CSV file and rule metadata for processing."""
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    if not rules.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Rules must be a JSON file")
    
    # Check file size (100MB limit)
    contents = await file.read()
    if len(contents) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 100MB")
    
    rules_content = await rules.read()
    
    # Generate processing ID
    processing_id = str(uuid.uuid4())
    
    # Save files temporarily
    temp_dir = Path(f"temp/{processing_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = temp_dir / file.filename
    rules_path = temp_dir / rules.filename
    
    csv_path.write_bytes(contents)
    rules_path.write_bytes(rules_content)
    
    # Initialize processing status
    processing_store[processing_id] = {
        'processing_id': processing_id,
        'status': 'queued',
        'progress': 0,
        'message': 'File uploaded successfully',
        'started_at': datetime.now().isoformat(),
        'filename': file.filename,
        'file_size': len(contents)
    }
    
    # Start background processing
    background_tasks.add_task(
        process_data_quality,
        processing_id,
        csv_path,
        rules_path
    )
    
    return ProcessingResponse(
        processing_id=processing_id,
        status='processing',
        message='File uploaded successfully'
    )


@app.get("/api/processing/{processing_id}", response_model=ProcessingStatus)
async def get_status(processing_id: str):
    """Get processing status."""
    status = get_processing_status(processing_id)
    if not status:
        raise HTTPException(status_code=404, detail="Processing not found")
    
    return ProcessingStatus(**status)


@app.get("/api/results/{processing_id}")
async def get_results(processing_id: str):
    """Get processing results."""
    results = get_processing_results(processing_id)
    if not results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return results


@app.get("/api/export/{processing_id}")
async def export_results(processing_id: str, format: str = "csv"):
    """Export results in specified format."""
    if format not in ["csv"]:
        raise HTTPException(status_code=400, detail="Invalid export format. Only 'csv' is supported")
    
    csv_content = export_results_csv(processing_id)
    if not csv_content:
        raise HTTPException(status_code=404, detail="Export not found")
    
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=data_quality_summary_{processing_id}.csv"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)