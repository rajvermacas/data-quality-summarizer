"""
Tests for the main Streamlit application.

These tests cover Stage 1 requirements:
- Basic web application launch
- Main page accessibility  
- Core navigation structure
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


class TestMainApplication:
    """Test the main Streamlit application functionality."""
    
    def test_app_module_exists(self):
        """Test that the app module can be imported."""
        # This test will fail initially since app.py doesn't exist yet
        try:
            import data_quality_summarizer.ui.app
            assert hasattr(data_quality_summarizer.ui.app, 'main')
        except ImportError as e:
            pytest.fail(f"Could not import app module: {e}")
    
    @patch('streamlit.set_page_config')
    @patch('streamlit.title')
    @patch('streamlit.sidebar')
    def test_app_initialization(self, mock_sidebar, mock_title, mock_config):
        """Test that the app initializes with proper configuration."""
        from data_quality_summarizer.ui.app import main
        
        # Call main function
        main()
        
        # Verify Streamlit configuration is set
        mock_config.assert_called_once()
        config_call = mock_config.call_args
        assert 'page_title' in config_call.kwargs
        assert 'Data Quality Summarizer' in config_call.kwargs['page_title']
        
        # Verify main title is set
        mock_title.assert_called_once_with('Data Quality Summarizer')
    
    @patch('streamlit.sidebar')
    def test_navigation_menu_exists(self, mock_sidebar):
        """Test that navigation menu is created in sidebar."""
        from data_quality_summarizer.ui.app import main
        
        main()
        
        # Verify sidebar is accessed (navigation menu should be created)
        mock_sidebar.assert_called()
    
    def test_app_has_required_functions(self):
        """Test that app module has all required functions."""
        from data_quality_summarizer.ui import app
        
        # Required functions for Stage 1
        assert hasattr(app, 'main'), "App must have main() function"
        assert hasattr(app, 'setup_navigation'), "App must have setup_navigation() function"
        assert callable(app.main), "main() must be callable"
        assert callable(app.setup_navigation), "setup_navigation() must be callable"


class TestApplicationLaunch:
    """Test application launch mechanisms."""
    
    def test_app_can_be_launched_via_module(self):
        """Test that app can be launched via python -m command."""
        # This tests the module structure for launch compatibility
        try:
            import data_quality_summarizer.ui.app
            # If we can import it, the module structure is correct
            assert True
        except ImportError:
            pytest.fail("UI app module should be importable for launch")
    
    def test_streamlit_dependencies_available(self):
        """Test that required Streamlit dependencies are available."""
        try:
            import streamlit
            assert hasattr(streamlit, 'set_page_config')
            assert hasattr(streamlit, 'title')
            assert hasattr(streamlit, 'sidebar')
        except ImportError:
            pytest.fail("Streamlit dependencies must be available")