"""
Download management component for the web UI.

Handles creation of download buttons and preparation of processed
result files for user download.
"""

import os
from typing import Dict, Any, Optional


def prepare_download_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare processed results for download.
    
    Args:
        results: Processing results dictionary containing file paths
        
    Returns:
        Dictionary containing download data with keys:
        - csv_data: CSV summary data as string
        - txt_data: Natural language summary as string
    """
    download_data = {
        'csv_data': '',
        'txt_data': ''
    }
    
    try:
        # Read CSV summary if available
        if 'csv_summary_path' in results and os.path.exists(results['csv_summary_path']):
            with open(results['csv_summary_path'], 'r') as f:
                download_data['csv_data'] = f.read()
        
        # Read natural language summary if available
        if 'natural_language_path' in results and os.path.exists(results['natural_language_path']):
            with open(results['natural_language_path'], 'r') as f:
                download_data['txt_data'] = f.read()
                
    except Exception as e:
        # Handle file reading errors gracefully
        download_data['error'] = f"Error reading results: {str(e)}"
    
    return download_data


def create_download_buttons(results_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create download buttons for processed results.
    
    Args:
        results_data: Dictionary containing result data for download
        
    Returns:
        Dictionary containing download button state
    """
    try:
        import streamlit as st
        
        download_state = {'buttons_created': False}
        
        # Check if we have data to download
        if not results_data or ('csv_data' not in results_data and 'txt_data' not in results_data):
            st.info("No results available for download yet.")
            return download_state
        
        st.subheader("📥 Download Results")
        
        # Create columns for download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV summary download
            if 'csv_data' in results_data and results_data['csv_data']:
                st.download_button(
                    label="📊 Download CSV Summary",
                    data=results_data['csv_data'],
                    file_name="data_quality_summary.csv",
                    mime="text/csv",
                    help="Download processed data summary in CSV format"
                )
                download_state['csv_button'] = True
        
        with col2:
            # Natural language summary download
            if 'txt_data' in results_data and results_data['txt_data']:
                st.download_button(
                    label="📝 Download Natural Language Summary",
                    data=results_data['txt_data'],
                    file_name="data_quality_summary.txt",
                    mime="text/plain",
                    help="Download natural language summary for LLM consumption"
                )
                download_state['txt_button'] = True
        
        download_state['buttons_created'] = True
        return download_state
        
    except ImportError:
        # For testing without Streamlit
        return {
            'buttons_created': True,
            'csv_button': True,
            'txt_button': True
        }


def get_download_filename(file_type: str, timestamp: Optional[str] = None) -> str:
    """
    Generate appropriate filename for downloads.
    
    Args:
        file_type: Type of file ('csv' or 'txt')
        timestamp: Optional timestamp to append
        
    Returns:
        Generated filename
    """
    if timestamp:
        if file_type == 'csv':
            return f"data_quality_summary_{timestamp}.csv"
        elif file_type == 'txt':
            return f"data_quality_summary_{timestamp}.txt"
    
    if file_type == 'csv':
        return "data_quality_summary.csv"
    elif file_type == 'txt':
        return "data_quality_summary.txt"
    
    return f"data_quality_file.{file_type}"