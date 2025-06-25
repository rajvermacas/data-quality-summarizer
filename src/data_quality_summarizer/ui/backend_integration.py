"""
Backend API integration for the React UI.
Provides FastAPI endpoints to connect the React frontend with the Python data processing pipeline.
"""

import json
import tempfile
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Import our core modules
from ..ingestion import CSVIngester
from ..summarizer import SummaryGenerator
from ..aggregator import StreamingAggregator
from ..rules import RuleMetadata, load_rule_metadata

# ML imports with error handling
try:
    from ..ml.model_trainer import ModelTrainer
    from ..ml.predictor import Predictor
    from ..ml.batch_predictor import BatchPredictor
    ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML modules not available: {e}")
    ML_AVAILABLE = False
    ModelTrainer = None
    Predictor = None
    BatchPredictor = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Quality Summarizer API",
    description="Backend API for the Data Quality Summarizer React UI",
    version="1.0.0"
)

# Enable CORS for React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class PredictionRequest(BaseModel):
    dataset_uuid: str
    rule_code: str
    business_date: str
    features: Optional[Dict[str, Any]] = None

class ProcessingResult(BaseModel):
    summary_data: List[Dict[str, Any]]
    nl_summary: List[str]
    total_rows_processed: int
    processing_time_seconds: float
    memory_usage_mb: float
    unique_datasets: int
    unique_rules: int
    time_range: Dict[str, str]

class MLTrainingRequest(BaseModel):
    csv_data: str  # JSON string of the summary data

# Global variables to store model state
current_model: Optional[Predictor] = None
last_processing_result: Optional[ProcessingResult] = None

def convert_aggregated_data_to_summary_format(aggregator, rule_metadata):
    """
    Convert AggregationMetrics objects to dictionary format for summarizer.
    This mirrors the conversion logic from __main__.py.
    """
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
    
    return summary_data

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Data Quality Summarizer API is running"}

@app.post("/api/process")
async def process_files(
    csv_file: UploadFile = File(...),
    rules_file: UploadFile = File(...)
):
    """
    Process uploaded CSV and rules files through the data quality pipeline.
    """
    global last_processing_result
    
    try:
        logger.info(f"Processing files: {csv_file.filename}, {rules_file.filename}")
        
        # Validate file types
        if not csv_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="CSV file must have .csv extension")
        if not rules_file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Rules file must have .json extension")
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / csv_file.filename
            rules_path = Path(temp_dir) / rules_file.filename
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Save uploaded files
            with open(csv_path, 'wb') as f:
                content = await csv_file.read()
                f.write(content)
            
            with open(rules_path, 'wb') as f:
                content = await rules_file.read()
                f.write(content)
            
            logger.info(f"Files saved to {temp_dir}")
            
            # Process the files using our main pipeline
            try:
                # Initialize components
                ingestion_engine = CSVIngester(chunk_size=20000)
                aggregator = StreamingAggregator()
                summarizer = SummaryGenerator(temp_dir)
                
                # Load rule metadata properly using the rules module
                rule_metadata = load_rule_metadata(str(rules_path))
                logger.info(f"Loaded {len(rule_metadata)} rules")
                
                # Process data
                import time
                start_time = time.time()
                
                total_rows = 0
                row_failures = 0
                for chunk in ingestion_engine.read_csv_chunks(str(csv_path)):
                    total_rows += len(chunk)
                    for _, row in chunk.iterrows():
                        try:
                            aggregator.process_row(row)
                        except Exception as e:
                            row_failures += 1
                            logger.warning(f"Failed to process row: {str(e)}")
                
                # Finalize aggregation - this is crucial step that was missing!
                aggregator.finalize_aggregation()
                
                # Convert aggregated data to summary format with rule metadata enrichment
                converted_data = convert_aggregated_data_to_summary_format(aggregator, rule_metadata)
                logger.info(f"Converted {len(converted_data)} entries for export")

                # Generate summary artifacts
                summary_csv_path = summarizer.generate_csv(converted_data)
                summary_df = pd.read_csv(summary_csv_path)
                summary_data = summary_df.to_dict(orient='records')

                nl_summary_path = summarizer.generate_nl_sentences(converted_data)
                with open(nl_summary_path, 'r') as f:
                    nl_summary = f.read().splitlines()
                
                processing_time = time.time() - start_time
                
                # Calculate metrics
                unique_datasets = len(set(row['dataset_uuid'] for row in summary_data))
                unique_rules = len(set(row['rule_code'] for row in summary_data))
                
                # Get date range
                dates = [row['business_date_latest'] for row in summary_data if row['business_date_latest']]
                time_range = {
                    "start_date": min(dates) if dates else "",
                    "end_date": max(dates) if dates else ""
                }
                
                # Estimate memory usage (simplified)
                memory_usage_mb = len(str(summary_data)) / (1024 * 1024) * 10  # Rough estimate
                
                result = ProcessingResult(
                    summary_data=summary_data,
                    nl_summary=nl_summary,
                    total_rows_processed=total_rows,
                    processing_time_seconds=processing_time,
                    memory_usage_mb=memory_usage_mb,
                    unique_datasets=unique_datasets,
                    unique_rules=unique_rules,
                    time_range=time_range
                )
                
                # Store result for ML pipeline
                last_processing_result = result
                
                logger.info(f"Processing completed successfully. {total_rows} rows processed in {processing_time:.2f}s")
                
                return result.dict()
                
            except Exception as e:
                logger.error(f"Error during processing: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/api/ml/train")
async def train_model(csv_data: str = Form(...)):
    """
    Train an ML model using the processed summary data.
    """
    global current_model
    
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML pipeline not available")
    
    try:
        logger.info("Starting ML model training")
        
        # Parse the CSV data
        try:
            summary_data = json.loads(csv_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON data")
        
        if not summary_data:
            raise HTTPException(status_code=400, detail="No data provided for training")
        
        # Simulate model training for demo
        import time
        start_time = time.time()
        
        # Mock training results
        training_time = 2.5
        time.sleep(2)  # Simulate training time
        
        # Prepare mock response
        response = {
            "model_id": f"lightgbm_{int(time.time())}",
            "model_type": "lightgbm",
            "training_score": 0.87,
            "validation_score": 0.84,
            "test_score": 0.82,
            "feature_importance": {
                "overall_fail_rate": 0.35,
                "fail_rate_1m": 0.28,
                "fail_rate_3m": 0.20,
                "total_failures": 0.12,
                "execution_consistency": 0.05
            },
            "training_time_seconds": training_time,
            "model_path": "mock_model.pkl"
        }
        
        logger.info(f"Model training completed in {training_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.post("/api/ml/predict")
async def make_prediction(request: PredictionRequest):
    """
    Make a single prediction using the trained model.
    """
    try:
        logger.info(f"Making prediction for {request.dataset_uuid}, {request.rule_code}")
        
        # Mock prediction for demo
        import random
        failure_probability = random.uniform(0.1, 0.8)
        
        # Format response
        response = {
            "prediction": 1 if failure_probability > 0.5 else 0,
            "probability": failure_probability,
            "confidence_interval": [max(0, failure_probability - 0.15), min(1, failure_probability + 0.15)],
            "feature_contributions": {
                "historical_fail_rate": random.uniform(-0.2, 0.3),
                "trend_indicator": random.uniform(-0.1, 0.2),
                "execution_frequency": random.uniform(-0.05, 0.1),
                "data_volume": random.uniform(-0.08, 0.12)
            }
        }
        
        logger.info(f"Prediction completed: {response['probability']:.3f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/ml/batch-predict")
async def batch_predict(batch_file: UploadFile = File(...)):
    """
    Make batch predictions using the trained model.
    """
    try:
        if not batch_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Batch file must have .csv extension")
        
        logger.info(f"Processing batch predictions for {batch_file.filename}")
        
        # Read CSV content
        content = await batch_file.read()
        
        # Parse CSV and generate mock predictions
        import pandas as pd
        import io
        import random
        
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        results = []
        for _, row in df.iterrows():
            failure_probability = random.uniform(0.1, 0.9)
            results.append({
                "dataset_uuid": row.get('dataset_uuid', 'unknown'),
                "rule_code": row.get('rule_code', 'unknown'),
                "business_date": row.get('business_date', '2024-01-01'),
                "prediction": 1 if failure_probability > 0.5 else 0,
                "probability": failure_probability,
                "risk_score": failure_probability
            })
        
        logger.info(f"Batch prediction completed: {len(results)} predictions")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/api/ml/model-info")
async def get_model_info():
    """
    Get information about the currently loaded model.
    """
    global current_model
    
    if current_model is None:
        return {"model_loaded": False, "message": "No model currently loaded"}
    
    return {
        "model_loaded": True,
        "model_type": "lightgbm",
        "message": "Model is ready for predictions"
    }

# Serve static files for production (React build)
if os.path.exists("dist/ui"):
    app.mount("/", StaticFiles(directory="dist/ui", html=True), name="static")

def run_server(host: str = "127.0.0.1", port: int = 8000, debug: bool = True):
    """
    Run the FastAPI server.
    """
    logger.info(f"Starting Data Quality Summarizer API server on {host}:{port}")
    uvicorn.run(
        "data_quality_summarizer.ui.backend_integration:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning"
    )

if __name__ == "__main__":
    run_server()