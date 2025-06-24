"""
Tests for backend integration component.

Tests Stage 1 completion requirement: Backend integration with existing pipeline.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


class TestBackendIntegration:
    """Test backend integration with existing pipeline classes."""
    
    def test_backend_integration_module_exists(self):
        """Test that backend integration module can be imported."""
        try:
            from data_quality_summarizer.ui.utils import backend_integration
            assert hasattr(backend_integration, 'UIProcessingPipeline')
        except ImportError:
            pytest.fail("Backend integration module should be importable")
    
    def test_ui_processing_pipeline_class_exists(self):
        """Test that UIProcessingPipeline wrapper class exists."""
        from data_quality_summarizer.ui.utils.backend_integration import UIProcessingPipeline
        
        # Should be a class that wraps the existing pipeline
        assert isinstance(UIProcessingPipeline, type)
    
    def test_ui_pipeline_has_required_methods(self):
        """Test that UI pipeline has required methods for web interface."""
        from data_quality_summarizer.ui.utils.backend_integration import UIProcessingPipeline
        
        # Required methods for UI integration
        assert hasattr(UIProcessingPipeline, 'process_files')
        assert hasattr(UIProcessingPipeline, 'get_progress')
        assert hasattr(UIProcessingPipeline, 'get_results')
    
    def test_ui_pipeline_initialization(self):
        """Test UI pipeline can be initialized with progress callback."""
        from data_quality_summarizer.ui.utils.backend_integration import UIProcessingPipeline
        
        # Mock progress callback
        progress_callback = Mock()
        
        # Should initialize successfully
        pipeline = UIProcessingPipeline(progress_callback=progress_callback)
        assert pipeline is not None
    
    @patch('data_quality_summarizer.ui.utils.backend_integration.run_pipeline')
    def test_ui_pipeline_process_files(self, mock_run_pipeline):
        """Test that UI pipeline processes files using existing backend."""
        from data_quality_summarizer.ui.utils.backend_integration import UIProcessingPipeline
        
        # Mock successful pipeline run
        mock_run_pipeline.return_value = True
        
        # Setup pipeline
        progress_callback = Mock()
        pipeline = UIProcessingPipeline(progress_callback=progress_callback)
        
        # Test file processing
        with tempfile.NamedTemporaryFile(suffix='.csv') as csv_file, \
             tempfile.NamedTemporaryFile(suffix='.json') as json_file:
            
            csv_file.write(b"test,data")
            json_file.write(b'{"R001": {"name": "test"}}')
            csv_file.flush()
            json_file.flush()
            
            result = pipeline.process_files(csv_file.name, json_file.name)
            
            # Should call existing pipeline
            mock_run_pipeline.assert_called_once()
            assert result is True
    
    def test_ui_pipeline_progress_tracking(self):
        """Test that UI pipeline tracks and reports progress."""
        from data_quality_summarizer.ui.utils.backend_integration import UIProcessingPipeline
        
        # Mock progress callback
        progress_callback = Mock()
        pipeline = UIProcessingPipeline(progress_callback=progress_callback)
        
        # Progress should be trackable
        progress = pipeline.get_progress()
        assert isinstance(progress, dict)
        assert 'current_step' in progress
        assert 'total_steps' in progress
        assert 'percentage' in progress


class TestProgressTracking:
    """Test progress tracking functionality."""
    
    def test_progress_tracker_module_exists(self):
        """Test that progress tracker component exists."""
        try:
            from data_quality_summarizer.ui.components import progress_tracker
            assert hasattr(progress_tracker, 'ProgressTracker')
        except ImportError:
            pytest.fail("Progress tracker component should be importable")
    
    def test_progress_tracker_initialization(self):
        """Test progress tracker can be initialized."""
        from data_quality_summarizer.ui.components.progress_tracker import ProgressTracker
        
        tracker = ProgressTracker()
        assert tracker is not None
    
    def test_progress_tracker_update_progress(self):
        """Test progress tracker can update progress."""
        from data_quality_summarizer.ui.components.progress_tracker import ProgressTracker
        
        tracker = ProgressTracker()
        
        # Should be able to update progress
        tracker.update_progress(25, "Processing data...")
        
        # Should be able to get current progress
        progress = tracker.get_progress()
        assert progress['percentage'] == 25
        assert progress['message'] == "Processing data..."
    
    def test_progress_tracker_creates_ui_elements(self):
        """Test progress tracker creates UI elements."""
        from data_quality_summarizer.ui.components.progress_tracker import ProgressTracker
        
        tracker = ProgressTracker()
        elements = tracker.create_ui_elements()
        
        # Should return UI element structure (even if None for testing)
        assert isinstance(elements, dict)
        assert 'progress_bar' in elements
        assert 'status_text' in elements


class TestDownloadFunctionality:
    """Test download functionality for processed results."""
    
    def test_download_module_exists(self):
        """Test that download functionality module exists."""
        try:
            from data_quality_summarizer.ui.components import download_manager
            assert hasattr(download_manager, 'create_download_buttons')
        except ImportError:
            pytest.fail("Download manager component should be importable")
    
    def test_download_manager_has_required_functions(self):
        """Test download manager has required functions."""
        from data_quality_summarizer.ui.components import download_manager
        
        assert hasattr(download_manager, 'create_download_buttons')
        assert hasattr(download_manager, 'prepare_download_data')
        assert callable(download_manager.create_download_buttons)
        assert callable(download_manager.prepare_download_data)
    
    def test_create_download_buttons(self):
        """Test download buttons creation."""
        from data_quality_summarizer.ui.components.download_manager import create_download_buttons
        
        # Mock results data
        results_data = {
            'csv_data': 'test,data\n1,2',
            'txt_data': 'Test summary text'
        }
        
        # Should create download buttons (returns structure even without streamlit)
        result = create_download_buttons(results_data)
        
        # Should return download button state structure
        assert isinstance(result, dict)
        assert 'buttons_created' in result
        assert result['buttons_created'] is True
    
    def test_prepare_download_data(self):
        """Test download data preparation."""
        from data_quality_summarizer.ui.components.download_manager import prepare_download_data
        
        # Mock processed results
        mock_results = {
            'summary_csv': 'test,data\n1,2',
            'natural_language': 'Test summary'
        }
        
        # Should prepare data for download
        download_data = prepare_download_data(mock_results)
        
        assert isinstance(download_data, dict)
        assert 'csv_data' in download_data
        assert 'txt_data' in download_data