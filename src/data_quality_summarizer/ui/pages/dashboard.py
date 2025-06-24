"""
Dashboard page component for executive summary visualization.

This module provides the main dashboard page with executive overview
including quality metrics, trend analysis, and key performance indicators.
"""

from typing import Dict, Any

try:
    import streamlit as st
except ImportError:
    # For testing without Streamlit
    st = None

from ..visualizations.dashboard_charts import (
    create_quality_score_gauge,
    create_rule_category_pie_chart,
    create_trend_line_chart,
    create_key_metrics_cards
)


def display_dashboard_page(dashboard_data: Dict[str, Any]) -> None:
    """
    Display the executive dashboard page with key metrics and visualizations.
    
    Args:
        dashboard_data: Processed data containing metrics and summary information
    """
    if st is None:
        # Testing mode - minimal implementation
        print("Dashboard Page (Testing Mode)")
        return
    
    st.header("📊 Executive Dashboard")
    st.write("High-level overview of data quality metrics and trends.")
    
    # Key metrics cards
    metrics = create_key_metrics_cards(dashboard_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rules", metrics['total_rules'])
    
    with col2:
        st.metric("Total Datasets", metrics['total_datasets'])
    
    with col3:
        st.metric("Overall Pass Rate", f"{metrics['overall_pass_rate']:.1%}")
    
    with col4:
        st.metric("Latest Execution", metrics['latest_execution'])
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Quality score gauge
        gauge_fig = create_quality_score_gauge(dashboard_data)
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col2:
        # Rule category pie chart
        pie_fig = create_rule_category_pie_chart(dashboard_data)
        st.plotly_chart(pie_fig, use_container_width=True)
    
    # Trend line chart
    trend_fig = create_trend_line_chart(dashboard_data)
    st.plotly_chart(trend_fig, use_container_width=True)