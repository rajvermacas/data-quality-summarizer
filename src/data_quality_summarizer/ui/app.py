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
try:
    from src.data_quality_summarizer.ui.components.file_uploader import create_file_uploader, validate_csv_file, validate_json_file
    from src.data_quality_summarizer.ui.components.progress_tracker import ProgressTracker
    from src.data_quality_summarizer.ui.components.download_manager import create_download_buttons, prepare_download_data
    from src.data_quality_summarizer.ui.utils.backend_integration import UIProcessingPipeline
except ImportError:
    try:
        from components.file_uploader import create_file_uploader, validate_csv_file, validate_json_file
        from components.progress_tracker import ProgressTracker
        from components.download_manager import create_download_buttons, prepare_download_data
        from utils.backend_integration import UIProcessingPipeline
    except ImportError:
        # For testing without components
        create_file_uploader = None
        validate_csv_file = None
        validate_json_file = None
        ProgressTracker = None
        create_download_buttons = None
        prepare_download_data = None
        UIProcessingPipeline = None

# Import visualization page components
try:
    from src.data_quality_summarizer.ui.pages.dashboard import display_dashboard_page
    from src.data_quality_summarizer.ui.pages.rule_performance import display_rule_performance_page
    from src.data_quality_summarizer.ui.pages.dataset_insights import display_dataset_insights_page
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    try:
        from pages.dashboard import display_dashboard_page
        from pages.rule_performance import display_rule_performance_page
        from pages.dataset_insights import display_dataset_insights_page
        VISUALIZATIONS_AVAILABLE = True
    except ImportError:
        # For testing without visualization dependencies
        VISUALIZATIONS_AVAILABLE = False


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


def display_dashboard_visualization_page() -> None:
    """Display the dashboard visualization page with sample or processed data."""
    try:
        import streamlit as st
        
        if not VISUALIZATIONS_AVAILABLE:
            st.error("❌ Visualization components not available. Please install required dependencies.")
            return
        
        # Check if we have processed data in session state
        if st.session_state.get('processing_complete', False) and st.session_state.get('processing_results', {}):
            # Use actual processed data
            dashboard_data = st.session_state.processing_results
            display_dashboard_page(dashboard_data)
        else:
            # Show sample dashboard with placeholder data
            st.subheader("📊 Executive Dashboard")
            st.info("🔄 Upload and process data files to view live dashboard visualizations.")
            st.markdown("**Dashboard Features:**")
            st.markdown("- 📈 Quality score gauges and key metrics")
            st.markdown("- 🥧 Rule category distribution charts")
            st.markdown("- 📊 Quality trend analysis over time")
            st.markdown("- 🎯 Executive summary cards")
            
            # Sample metrics display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rules", "Sample: 25")
            with col2:
                st.metric("Total Datasets", "Sample: 5")
            with col3:
                st.metric("Overall Pass Rate", "Sample: 94.6%")
            with col4:
                st.metric("Latest Execution", "Sample: Today")
                
    except ImportError:
        print("Dashboard Visualization Page (Testing Mode)")


def display_rule_performance_visualization_page() -> None:
    """Display the rule performance visualization page."""
    try:
        import streamlit as st
        
        if not VISUALIZATIONS_AVAILABLE:
            st.error("❌ Visualization components not available. Please install required dependencies.")
            return
        
        # Check if we have processed data in session state
        if st.session_state.get('processing_complete', False) and st.session_state.get('processing_results', {}):
            # Use actual processed data
            dashboard_data = st.session_state.processing_results
            display_rule_performance_page(dashboard_data)
        else:
            # Show placeholder for rule performance
            st.subheader("📈 Rule Performance Analysis")
            st.info("🔄 Upload and process data files to view rule performance analytics.")
            st.markdown("**Rule Performance Features:**")
            st.markdown("- 📊 Rule rankings by fail rate")
            st.markdown("- 🔥 Performance heatmaps")
            st.markdown("- 📈 Time series comparisons")
            st.markdown("- 🎯 Correlation analysis")
            
    except ImportError:
        print("Rule Performance Visualization Page (Testing Mode)")


def display_dataset_insights_visualization_page() -> None:
    """Display the dataset insights visualization page."""
    try:
        import streamlit as st
        
        if not VISUALIZATIONS_AVAILABLE:
            st.error("❌ Visualization components not available. Please install required dependencies.")
            return
        
        # Check if we have processed data in session state
        if st.session_state.get('processing_complete', False) and st.session_state.get('processing_results', {}):
            # Use actual processed data
            dashboard_data = st.session_state.processing_results
            display_dataset_insights_page(dashboard_data)
        else:
            # Show placeholder for dataset insights
            st.subheader("🗂️ Dataset Quality Insights")
            st.info("🔄 Upload and process data files to view dataset quality insights.")
            st.markdown("**Dataset Insights Features:**")
            st.markdown("- 🚦 Health status indicators")
            st.markdown("- 📊 Quality distribution analysis")
            st.markdown("- 🏢 Tenant comparison views")
            st.markdown("- 🎯 Multi-dimensional quality assessment")
            
    except ImportError:
        print("Dataset Insights Visualization Page (Testing Mode)")


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
        elif nav_state['current_page'] == "Dashboard":
            display_dashboard_visualization_page()
        elif nav_state['current_page'] == "Rule Performance":
            display_rule_performance_visualization_page()
        elif nav_state['current_page'] == "Dataset Insights":
            display_dataset_insights_visualization_page()
        else:
            # Fallback for unknown pages
            st.subheader(f"🚧 {nav_state['current_page']} Page")
            st.info("This page is under construction.")
        
    except ImportError:
        # For testing without Streamlit
        print("Data Quality Summarizer Web UI (Testing Mode)")
        nav_state = setup_navigation()
        print(f"Current Page: {nav_state['current_page']}")


if __name__ == "__main__":
    main()