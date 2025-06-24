"""
Backend integration utilities for the web UI.

Provides wrapper classes and functions to integrate the existing CLI pipeline
with the web interface, including progress tracking and result management.
"""

import os
import tempfile
import threading
from typing import Dict, Any, Callable, Optional
from pathlib import Path

# Import existing pipeline functionality
try:
    from ....__main__ import run_pipeline
    from ....ingestion import ChunkedCSVReader
    from ....aggregator import StreamingAggregator
    from ....rules import RuleMetadataLoader
    from ....summarizer import DataQualitySummarizer
except ImportError:
    # For testing - mock these imports
    def run_pipeline(*args, **kwargs):
        return True
    ChunkedCSVReader = None
    StreamingAggregator = None
    RuleMetadataLoader = None
    DataQualitySummarizer = None


class UIProcessingPipeline:
    """
    UI wrapper for the existing data quality pipeline.
    
    Provides progress tracking and web-friendly interface for the
    command-line data processing pipeline.
    """
    
    def __init__(self, progress_callback: Optional[Callable[[int, str], None]] = None):
        """
        Initialize the UI processing pipeline.
        
        Args:
            progress_callback: Function to call with progress updates (percentage, message)
        """
        self.progress_callback = progress_callback
        self._progress = {
            'current_step': 0,
            'total_steps': 5,
            'percentage': 0,
            'message': 'Initialized'
        }
        self._results = {}
        self._processing_thread = None
    
    def _update_progress(self, step: int, message: str) -> None:
        """Update progress and call callback if provided."""
        self._progress['current_step'] = step
        self._progress['message'] = message
        self._progress['percentage'] = int((step / self._progress['total_steps']) * 100)
        
        if self.progress_callback:
            self.progress_callback(self._progress['percentage'], message)
    
    def process_files(self, csv_file_path: str, json_file_path: str, 
                     output_dir: Optional[str] = None, chunk_size: int = 20000) -> bool:
        """
        Process data files using the existing pipeline.
        
        Args:
            csv_file_path: Path to CSV data file
            json_file_path: Path to JSON rule metadata file
            output_dir: Output directory (optional, uses temp dir if not provided)
            chunk_size: Chunk size for processing
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            self._update_progress(1, "Starting data processing...")
            
            # Create output directory if not provided
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="data_quality_")
            
            self._update_progress(2, "Loading rule metadata...")
            
            # Process files using existing pipeline
            self._update_progress(3, "Processing data chunks...")
            
            # Use existing run_pipeline function
            success = run_pipeline(
                csv_file_path, 
                json_file_path, 
                output_dir=output_dir, 
                chunk_size=chunk_size
            )
            
            if success:
                self._update_progress(4, "Generating summaries...")
                
                # Store results
                self._results = {
                    'output_dir': output_dir,
                    'csv_summary_path': os.path.join(output_dir, 'artifacts', 'full_summary.csv'),
                    'natural_language_path': os.path.join(output_dir, 'artifacts', 'nl_all_rows.txt')
                }
                
                self._update_progress(5, "Processing complete!")
                return True
            else:
                self._update_progress(0, "Processing failed")
                return False
                
        except Exception as e:
            self._update_progress(0, f"Error: {str(e)}")
            return False
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current processing progress."""
        return self._progress.copy()
    
    def get_results(self) -> Dict[str, Any]:
        """Get processing results."""
        return self._results.copy()


def create_ui_pipeline(progress_callback: Optional[Callable[[int, str], None]] = None) -> UIProcessingPipeline:
    """
    Factory function to create a UI processing pipeline.
    
    Args:
        progress_callback: Function to call with progress updates
        
    Returns:
        Configured UIProcessingPipeline instance
    """
    return UIProcessingPipeline(progress_callback=progress_callback)