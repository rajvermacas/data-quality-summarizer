"""
Dataset insights page component for dataset quality analysis.

This module provides the dataset insights page with health indicators,
quality distributions, and dataset-centric visualizations.
"""

from typing import Dict, Any

try:
    import streamlit as st
except ImportError:
    # For testing without Streamlit
    st = None

from ..visualizations.dataset_analysis import (
    create_dataset_health_indicators,
    create_quality_distribution_histogram
)


def display_dataset_insights_page(dashboard_data: Dict[str, Any]) -> None:
    """
    Display the dataset insights page with health indicators and analysis.
    
    Args:
        dashboard_data: Processed data containing dataset quality metrics
    """
    if st is None:
        # Testing mode - minimal implementation
        print("Dataset Insights Page (Testing Mode)")
        return
    
    st.header("🗂️ Dataset Quality Insights")
    st.write("Comprehensive analysis of dataset health and quality metrics.")
    
    # Dataset health indicators
    st.subheader("Dataset Health Status")
    health_data = create_dataset_health_indicators(dashboard_data)
    
    if health_data['status'] == 'success':
        for indicator in health_data['indicators'][:5]:  # Show top 5
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{indicator['dataset']}** - {indicator['rule']}")
            
            with col2:
                st.metric("Pass Rate", f"{indicator['pass_rate']:.1%}")
            
            with col3:
                color = {
                    'green': '🟢',
                    'yellow': '🟡', 
                    'red': '🔴'
                }.get(indicator['status'], '⚪')
                st.write(f"{color} {indicator['status'].title()}")
    else:
        st.warning("No health indicator data available.")
    
    # Quality distribution histogram
    st.subheader("Quality Score Distribution")
    hist_fig = create_quality_distribution_histogram(dashboard_data)
    st.plotly_chart(hist_fig, use_container_width=True)
    
    # Summary insights
    st.subheader("Dataset Insights Summary")
    st.info("📈 Dataset quality analysis provides visibility into data health across your organization.")