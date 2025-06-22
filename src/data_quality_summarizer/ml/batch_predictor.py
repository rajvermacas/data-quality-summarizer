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
    
    def process_batch_with_recovery(
        self,
        input_file: str,
        output_file: str,
        error_recovery_strategy: str = 'continue_on_error'
    ) -> Dict[str, Any]:
        """
        Process batch with enhanced error recovery and progress tracking.
        
        Stage 2 enhancement: Provides error-resilient batch predictions that can
        continue processing despite individual record failures.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            error_recovery_strategy: Strategy for handling errors ('continue_on_error', 'stop_on_error')
            
        Returns:
            Dictionary containing processing results with error details
        """
        try:
            if not Path(input_file).exists():
                return {'success': False, 'error': f'Input file not found: {input_file}'}
            
            # Load input data
            input_data = pd.read_csv(input_file)
            
            # Validate input
            validation_result = self.validate_batch_input(input_data)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': 'Input validation failed',
                    'validation_details': validation_result
                }
            
            results = []
            errors = []
            processed = 0
            
            self.logger.info(f"Processing {len(input_data)} records with error recovery")
            
            for i, row in input_data.iterrows():
                try:
                    # Make prediction
                    prediction = self.predictor.predict(
                        dataset_uuid=row['dataset_uuid'],
                        rule_code=row['rule_code'],
                        business_date=row['business_date']
                    )
                    
                    results.append({
                        'dataset_uuid': row['dataset_uuid'],
                        'rule_code': row['rule_code'],
                        'business_date': row['business_date'],
                        'predicted_pass_percentage': prediction,
                        'status': 'success'
                    })
                    processed += 1
                    
                except Exception as e:
                    error_record = {
                        'dataset_uuid': row.get('dataset_uuid', 'unknown'),
                        'rule_code': row.get('rule_code', 'unknown'),
                        'business_date': row.get('business_date', 'unknown'),
                        'error': str(e),
                        'status': 'error'
                    }
                    
                    if error_recovery_strategy == 'continue_on_error':
                        results.append(error_record)
                        errors.append(error_record)
                    else:  # stop_on_error
                        return {
                            'success': False,
                            'error': f'Processing stopped at record {i}: {str(e)}',
                            'processed_count': processed,
                            'failed_record': error_record
                        }
            
            # Save results
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_file, index=False)
            
            return {
                'success': True,
                'total_records': len(input_data),
                'processed_successfully': processed,
                'error_count': len(errors),
                'error_details': errors,
                'output_file': output_file
            }
            
        except Exception as e:
            self.logger.error(f"Batch processing with recovery failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def validate_batch_input(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate batch input data comprehensively.
        
        Stage 2 enhancement: Provides comprehensive validation of batch input
        to ensure data quality before processing.
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            'valid': True,
            'issues': [],
            'warning_count': 0,
            'error_count': 0
        }
        
        # Check required columns
        required_columns = ['dataset_uuid', 'rule_code', 'business_date']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            validation_result['valid'] = False
            validation_result['error_count'] += 1
            validation_result['issues'].append({
                'type': 'error',
                'message': f'Missing required columns: {missing_columns}'
            })
        
        # Check for empty data
        if data.empty:
            validation_result['valid'] = False
            validation_result['error_count'] += 1
            validation_result['issues'].append({
                'type': 'error',
                'message': 'Input data is empty'
            })
            return validation_result
        
        # Check for null values in required columns
        for col in required_columns:
            if col in data.columns:
                null_count = data[col].isnull().sum()
                if null_count > 0:
                    validation_result['warning_count'] += 1
                    validation_result['issues'].append({
                        'type': 'warning',
                        'message': f'Column {col} has {null_count} null values'
                    })
        
        # Validate date format
        if 'business_date' in data.columns:
            try:
                pd.to_datetime(data['business_date'])
            except Exception as e:
                validation_result['valid'] = False
                validation_result['error_count'] += 1
                validation_result['issues'].append({
                    'type': 'error',
                    'message': f'Invalid date format in business_date: {str(e)}'
                })
        
        self.logger.info(f"Input validation: {'PASSED' if validation_result['valid'] else 'FAILED'} "
                        f"(errors: {validation_result['error_count']}, warnings: {validation_result['warning_count']})")
        
        return validation_result
    
    def resume_batch_processing(
        self,
        checkpoint_file: str,
        input_file: str,
        output_file: str
    ) -> Dict[str, Any]:
        """
        Resume batch processing from a checkpoint.
        
        Stage 2 enhancement: Provides resumable batch operations for large datasets
        that may need to be processed in multiple sessions.
        
        Args:
            checkpoint_file: Path to checkpoint file with progress information
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            
        Returns:
            Dictionary containing resumption results
        """
        try:
            if not Path(checkpoint_file).exists():
                return {
                    'success': False,
                    'error': f'Checkpoint file not found: {checkpoint_file}'
                }
            
            # Load checkpoint data
            import json
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            last_processed_index = checkpoint_data.get('last_processed_index', -1)
            processed_count = checkpoint_data.get('processed_count', 0)
            
            # Load input data
            input_data = pd.read_csv(input_file)
            
            # Skip already processed records
            remaining_data = input_data.iloc[last_processed_index + 1:]
            
            if remaining_data.empty:
                return {
                    'success': True,
                    'message': 'All records already processed',
                    'total_processed': processed_count
                }
            
            self.logger.info(f"Resuming processing from index {last_processed_index + 1}, "
                           f"{len(remaining_data)} records remaining")
            
            # Process remaining data
            return self.process_batch_with_recovery(
                input_file=input_file,
                output_file=output_file,
                error_recovery_strategy='continue_on_error'
            )
            
        except Exception as e:
            self.logger.error(f"Resume batch processing failed: {str(e)}")
            return {'success': False, 'error': str(e)}