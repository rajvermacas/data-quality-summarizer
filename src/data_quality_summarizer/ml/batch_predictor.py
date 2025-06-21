"""
Batch Predictor for ML Pipeline.
Handles multiple prediction requests efficiently with progress tracking.
"""

import logging
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import psutil
import os

from .predictor import Predictor


class BatchPredictor:
    """
    Handles batch prediction requests for multiple dataset/rule/date combinations.
    
    Provides efficient processing of multiple predictions with progress tracking
    and comprehensive error handling.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the batch predictor.
        
        Args:
            model_path: Optional path to trained model file
        """
        self.model_path = model_path
        # Predictor will be initialized when we have historical data
        self.predictor = None
        self.logger = logging.getLogger(__name__)
        self.progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int], None]) -> None:
        """
        Set callback function for progress tracking.
        
        Args:
            callback: Function that receives (stage, current, total) progress updates
        """
        self.progress_callback = callback
    
    def _report_progress(self, stage: str, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(stage, current, total)
    
    def process_batch_csv(
        self,
        input_csv: str,
        output_csv: str,
        historical_data_csv: str
    ) -> Dict[str, Any]:
        """
        Process batch predictions from CSV file.
        
        Args:
            input_csv: Path to CSV file with prediction requests
            output_csv: Path where to save prediction results
            historical_data_csv: Path to historical data for feature engineering
            
        Returns:
            Dictionary containing processing results
        """
        try:
            if not Path(input_csv).exists():
                return {'success': False, 'error': f'Input CSV not found: {input_csv}'}
            
            start_time = time.time()
            self.logger.info(f"Starting batch prediction processing: {input_csv}")
            
            # Load prediction requests
            try:
                requests_df = pd.read_csv(input_csv)
            except Exception as e:
                return {'success': False, 'error': f'Failed to read input CSV: {str(e)}'}
            
            # Validate required columns
            required_columns = ['dataset_uuid', 'rule_code', 'business_date']
            missing_columns = [col for col in required_columns if col not in requests_df.columns]
            if missing_columns:
                return {
                    'success': False, 
                    'error': f'Missing required columns: {missing_columns}'
                }
            
            # Load historical data
            historical_data = self._load_historical_data(historical_data_csv)
            
            # Initialize predictor with historical data
            if self.predictor is None:
                self.predictor = Predictor(model_path=self.model_path, historical_data=historical_data)
            
            # Convert to list of requests
            prediction_requests = requests_df.to_dict('records')
            
            # Process batch predictions
            results = self.process_batch_list(prediction_requests, historical_data)
            
            # Convert results to DataFrame and save
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_csv, index=False)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
            
            return {
                'success': True,
                'predictions_processed': len(results),
                'processing_time': processing_time,
                'output_file': output_csv
            }
            
        except Exception as e:
            error_msg = f"Batch CSV processing failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return {'success': False, 'error': error_msg}
    
    def process_batch_list(
        self,
        prediction_requests: List[Dict[str, Any]],
        historical_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Process a list of prediction requests.
        
        Args:
            prediction_requests: List of prediction request dictionaries
            historical_data: Historical data for feature engineering
            
        Returns:
            List of prediction results
        """
        if not prediction_requests:
            return []
        
        # Initialize predictor if not already done
        if self.predictor is None:
            self.predictor = Predictor(model_path=self.model_path, historical_data=historical_data)
        
        results = []
        total_requests = len(prediction_requests)
        
        self.logger.info(f"Processing {total_requests} prediction requests")
        
        for i, request in enumerate(prediction_requests):
            try:
                self._report_progress("Processing predictions", i + 1, total_requests)
                
                # Extract request parameters
                dataset_uuid = request['dataset_uuid']
                rule_code = request['rule_code']
                business_date = request['business_date']
                
                # Make prediction
                prediction = self.predictor.predict(
                    dataset_uuid=dataset_uuid,
                    rule_code=rule_code,
                    business_date=business_date
                )
                
                # Create result entry
                result = {
                    'dataset_uuid': dataset_uuid,
                    'rule_code': rule_code,
                    'business_date': business_date,
                    'predicted_pass_percentage': prediction
                }
                
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    self.logger.debug(f"Processed {i + 1}/{total_requests} predictions")
                
            except Exception as e:
                # Handle individual prediction errors
                error_result = {
                    'dataset_uuid': request.get('dataset_uuid', 'unknown'),
                    'rule_code': request.get('rule_code', 'unknown'),
                    'business_date': request.get('business_date', 'unknown'),
                    'error': str(e)
                }
                results.append(error_result)
                self.logger.warning(f"Prediction failed for request {i + 1}: {str(e)}")
        
        self.logger.info(f"Completed processing {len(results)} predictions")
        return results
    
    def _load_historical_data(self, historical_data_csv: str) -> pd.DataFrame:
        """
        Load historical data for feature engineering.
        
        Args:
            historical_data_csv: Path to historical data CSV
            
        Returns:
            Historical data DataFrame
        """
        try:
            if not Path(historical_data_csv).exists():
                self.logger.warning(f"Historical data file not found: {historical_data_csv}")
                return pd.DataFrame()
            
            historical_data = pd.read_csv(historical_data_csv)
            self.logger.info(f"Loaded {len(historical_data)} historical records")
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {str(e)}")
            return pd.DataFrame()
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0