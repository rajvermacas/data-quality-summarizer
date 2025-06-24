"""
Simple FastAPI backend for demo purposes.
"""

import json
import tempfile
import os
import logging
import time
import random
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import io

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PredictionRequest(BaseModel):
    dataset_uuid: str
    rule_code: str
    business_date: str

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Data Quality Summarizer API is running"}

@app.post("/api/process")
async def process_files(
    csv_file: UploadFile = File(...),
    rules_file: UploadFile = File(...)
):
    """Process uploaded CSV and rules files."""
    try:
        logger.info(f"Processing files: {csv_file.filename}, {rules_file.filename}")
        
        # Validate file types
        if not csv_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="CSV file must have .csv extension")
        if not rules_file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Rules file must have .json extension")
        
        # Read files
        csv_content = await csv_file.read()
        rules_content = await rules_file.read()
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
        rules = json.loads(rules_content.decode('utf-8'))
        
        # Simulate processing
        time.sleep(1)  # Simulate processing time
        
        # Generate mock summary data
        summary_data = []
        unique_datasets = df['dataset_uuid'].nunique() if 'dataset_uuid' in df.columns else 5
        unique_rules = len(rules)
        
        # Create sample summary data
        for i, (rule_code, rule_info) in enumerate(rules.items()):
            for j in range(min(3, unique_datasets)):  # Limit to 3 datasets per rule for demo
                dataset_uuid = f"dataset_{j+1}"
                dataset_name = f"Sample Dataset {j+1}"
                
                summary_data.append({
                    "source": "demo_source",
                    "tenant_id": "tenant_001",
                    "dataset_uuid": dataset_uuid,
                    "dataset_name": dataset_name,
                    "rule_code": rule_code,
                    "rule_category": rule_info.get("category", "Unknown"),
                    "rule_description": rule_info.get("description", "Sample rule description"),
                    "total_passes": random.randint(80, 200),
                    "total_failures": random.randint(5, 30),
                    "overall_fail_rate": random.uniform(0.02, 0.25),
                    "pass_rate_1m": random.uniform(0.75, 0.98),
                    "fail_rate_1m": random.uniform(0.02, 0.25),
                    "pass_rate_3m": random.uniform(0.75, 0.98),
                    "fail_rate_3m": random.uniform(0.02, 0.25),
                    "pass_rate_12m": random.uniform(0.75, 0.98),
                    "fail_rate_12m": random.uniform(0.02, 0.25),
                    "trend_1m_vs_3m": random.choice(["IMPROVING", "STABLE", "DEGRADING"]),
                    "trend_3m_vs_12m": random.choice(["IMPROVING", "STABLE", "DEGRADING"]),
                    "latest_business_date": "2024-01-31",
                    "earliest_business_date": "2023-02-01",
                    "total_execution_days": random.randint(300, 365),
                    "avg_daily_executions": random.uniform(0.8, 1.2),
                    "execution_consistency": random.uniform(0.85, 1.0),
                    "data_volume_trend": random.choice(["INCREASING", "STABLE", "DECREASING"]),
                    "risk_level": random.choice(["LOW", "MEDIUM", "HIGH"]),
                    "improvement_needed": random.choice([True, False]),
                    "last_failure_date": "2024-01-25" if random.choice([True, False]) else None
                })
        
        # Generate natural language summary
        nl_summary = []
        for row in summary_data[:10]:  # Limit to first 10 for demo
            sentence = f"• On {row['latest_business_date']}, dataset \"{row['dataset_name']}\" under rule \"{row['rule_code']}\" recorded {row['total_failures']} failures and {row['total_passes']} passes overall (fail-rate {row['overall_fail_rate']:.1%}; 1-month {row['fail_rate_1m']:.1%}, 3-month {row['fail_rate_3m']:.1%}, 12-month {row['fail_rate_12m']:.1%}) — trend {row['trend_1m_vs_3m']}."
            nl_summary.append(sentence)
        
        # Calculate metrics
        dates = [row['latest_business_date'] for row in summary_data]
        time_range = {
            "start_date": min(dates) if dates else "",
            "end_date": max(dates) if dates else ""
        }
        
        result = {
            "summary_data": summary_data,
            "nl_summary": nl_summary,
            "total_rows_processed": len(df),
            "processing_time_seconds": 1.5,
            "memory_usage_mb": 45.2,
            "unique_datasets": unique_datasets,
            "unique_rules": unique_rules,
            "time_range": time_range
        }
        
        logger.info(f"Processing completed: {len(summary_data)} summary rows generated")
        return result
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/ml/train")
async def train_model(csv_data: str = Form(...)):
    """Train an ML model."""
    try:
        logger.info("Starting ML model training")
        
        # Parse the CSV data
        summary_data = json.loads(csv_data)
        
        if not summary_data:
            raise HTTPException(status_code=400, detail="No data provided for training")
        
        # Simulate training
        time.sleep(2)
        
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
            "training_time_seconds": 2.0,
            "model_path": "mock_model.pkl"
        }
        
        logger.info("Model training completed")
        return response
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.post("/api/ml/predict")
async def make_prediction(request: PredictionRequest):
    """Make a single prediction."""
    try:
        logger.info(f"Making prediction for {request.dataset_uuid}, {request.rule_code}")
        
        # Mock prediction
        failure_probability = random.uniform(0.1, 0.8)
        
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
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/ml/batch-predict")
async def batch_predict(batch_file: UploadFile = File(...)):
    """Make batch predictions."""
    try:
        if not batch_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Batch file must have .csv extension")
        
        logger.info(f"Processing batch predictions for {batch_file.filename}")
        
        # Read CSV content
        content = await batch_file.read()
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
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/api/ml/model-info")
async def get_model_info():
    """Get model information."""
    return {
        "model_loaded": True,
        "model_type": "lightgbm",
        "message": "Model is ready for predictions"
    }

# Serve static files for production
if os.path.exists("dist/ui"):
    app.mount("/", StaticFiles(directory="dist/ui", html=True), name="static")

def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI server."""
    logger.info(f"Starting Data Quality Summarizer API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server(host="0.0.0.0", port=8000)