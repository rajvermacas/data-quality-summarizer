"""
Rule performance page component for detailed rule analysis.

This module provides the rule performance analysis page with interactive
charts for rule rankings, performance trends, and correlation analysis.
"""

from typing import Dict, Any

try:
    import streamlit as st
except ImportError:
    # For testing without Streamlit
    st = None

from ..visualizations.rule_analytics import (
    create_rule_ranking_bar_chart,
    create_performance_heatmap
)


def display_rule_performance_page(dashboard_data: Dict[str, Any]) -> None:
    """
    Display the rule performance analysis page with detailed charts.
    
    Args:
        dashboard_data: Processed data containing rule performance metrics
    """
    if st is None:
        # Testing mode - minimal implementation
        print("Rule Performance Page (Testing Mode)")
        return
    
    st.header("📈 Rule Performance Analysis")
    st.write("Detailed analysis of rule performance metrics and trends.")
    
    # Rule ranking bar chart
    st.subheader("Rule Performance Rankings")
    ranking_fig = create_rule_ranking_bar_chart(dashboard_data)
    st.plotly_chart(ranking_fig, use_container_width=True)
    
    # Performance heatmap
    st.subheader("Dataset vs Rule Performance Matrix")
    heatmap_fig = create_performance_heatmap(dashboard_data)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Additional analysis section
    st.subheader("Performance Insights")
    st.info("📊 Rule performance analysis helps identify problematic rules and optimization opportunities.")