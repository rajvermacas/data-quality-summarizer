"""
Main Streamlit application for Data Quality Summarizer Web UI.

This module provides the main entry point for the web interface,
implementing Stage 1 completion requirements with backend integration,
progress tracking, and download functionality.
"""

import tempfile
import os
from typing import Dict, Any, Optional

# Import UI components
from .components.file_uploader import create_file_uploader, validate_csv_file, validate_json_file
from .components.progress_tracker import ProgressTracker
from .components.download_manager import create_download_buttons, prepare_download_data
from .utils.backend_integration import UIProcessingPipeline


def setup_navigation() -> Dict[str, Any]:
    """
    Setup navigation menu in sidebar.
    
    Returns:
        Dict containing navigation state and selected page.
    """
    try:
        import streamlit as st
        
        with st.sidebar:
            st.header("Navigation")
            
            # Main navigation options for Stage 1
            page = st.selectbox(
                "Select Page",
                [
                    "Data Upload",
                    "Dashboard", 
                    "Rule Performance",
                    "Dataset Insights"
                ],
                index=0
            )
            
            return {"current_page": page}
    except ImportError:
        # Return default for testing without Streamlit
        return {"current_page": "Data Upload"}


def display_data_upload_page() -> None:
    """Display the main data upload and processing page."""
    try:
        import streamlit as st
        
        st.header("📊 Data Quality Analysis")
        st.write("Upload your data files to begin processing.")
        
        # File upload section
        uploader_state = create_file_uploader()
        
        # Initialize session state for processing
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = {}
        
        # Process files if both are uploaded
        if uploader_state['csv_file'] is not None and uploader_state['json_file'] is not None:
            st.success("✅ Both files uploaded successfully!")
            
            # Validate files
            csv_validation = validate_csv_file(uploader_state['csv_file'].read())
            json_validation = validate_json_file(uploader_state['json_file'].read())
            
            # Reset file pointer after reading for validation
            uploader_state['csv_file'].seek(0)
            uploader_state['json_file'].seek(0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if csv_validation['is_valid']:
                    st.success(f"✅ CSV Valid: {csv_validation['row_count']} rows")
                else:
                    st.error(f"❌ CSV Error: {csv_validation['error']}")
            
            with col2:
                if json_validation['is_valid']:
                    st.success(f"✅ JSON Valid: {json_validation['rule_count']} rules")
                else:
                    st.error(f"❌ JSON Error: {json_validation['error']}")
            
            # Processing section
            if csv_validation['is_valid'] and json_validation['is_valid']:
                st.subheader("🔄 Data Processing")
                
                # Processing configuration
                with st.expander("⚙️ Processing Configuration"):
                    chunk_size = st.slider("Chunk Size", min_value=10000, max_value=100000, value=20000, step=5000)
                    st.info(f"Selected chunk size: {chunk_size:,} rows")
                
                # Process button
                if st.button("🚀 Start Analysis", type="primary"):
                    process_data_files(uploader_state['csv_file'], uploader_state['json_file'], chunk_size)
        
        # Display results if processing is complete
        if st.session_state.processing_complete and st.session_state.processing_results:
            st.subheader("📋 Processing Results")
            
            # Prepare download data
            download_data = prepare_download_data(st.session_state.processing_results)
            
            # Create download buttons
            create_download_buttons(download_data)
            
            # Show summary
            if 'csv_data' in download_data and download_data['csv_data']:
                st.success(f"✅ Processing completed successfully!")
                st.info("Your processed data is ready for download above.")
        
    except ImportError:
        # For testing without Streamlit
        print("Data Upload Page (Testing Mode)")


def process_data_files(csv_file, json_file, chunk_size: int = 20000) -> None:
    """Process uploaded data files using the backend pipeline."""
    try:
        import streamlit as st
        
        # Create progress tracker
        progress_tracker = ProgressTracker()
        progress_elements = progress_tracker.create_ui_elements()
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_csv, \
             tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_json:
            
            # Write uploaded files to temporary files
            tmp_csv.write(csv_file.read())
            tmp_json.write(json_file.read())
            tmp_csv.flush()
            tmp_json.flush()
            
            # Progress callback function
            def update_progress(percentage: int, message: str):
                progress_tracker.update_progress(percentage, message)
            
            # Create and run pipeline
            pipeline = UIProcessingPipeline(progress_callback=update_progress)
            
            try:
                success = pipeline.process_files(
                    tmp_csv.name, 
                    tmp_json.name, 
                    chunk_size=chunk_size
                )
                
                if success:
                    # Store results in session state
                    st.session_state.processing_results = pipeline.get_results()
                    st.session_state.processing_complete = True
                    st.rerun()
                else:
                    st.error("❌ Processing failed. Please check your files and try again.")
                    
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(tmp_csv.name)
                    os.unlink(tmp_json.name)
                except:
                    pass
                    
    except ImportError:
        # For testing without Streamlit
        print("Processing files (Testing Mode)")


def main() -> None:
    """
    Main application function that initializes the Streamlit app.
    
    Sets up page configuration, navigation, and displays the main interface.
    """
    try:
        import streamlit as st
        
        # Configure Streamlit page
        st.set_page_config(
            page_title="Data Quality Summarizer",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main title
        st.title("Data Quality Summarizer")
        st.markdown("Transform your data quality analysis with our comprehensive web interface.")
        
        # Setup navigation
        nav_state = setup_navigation()
        
        # Display appropriate page based on navigation
        if nav_state['current_page'] == "Data Upload":
            display_data_upload_page()
        else:
            # Placeholder for other pages (Stage 2)
            st.subheader(f"🚧 {nav_state['current_page']} Page")
            st.info(f"The {nav_state['current_page']} page will be implemented in Stage 2: Visualization Dashboard.")
            st.markdown("**Stage 1 Complete Features:**")
            st.markdown("- ✅ File upload with validation")
            st.markdown("- ✅ Backend integration")
            st.markdown("- ✅ Progress tracking")
            st.markdown("- ✅ Download functionality")
        
    except ImportError:
        # For testing without Streamlit
        print("Data Quality Summarizer Web UI (Testing Mode)")
        nav_state = setup_navigation()
        print(f"Current Page: {nav_state['current_page']}")


if __name__ == "__main__":
    main()