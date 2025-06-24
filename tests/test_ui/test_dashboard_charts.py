"""
Tests for dashboard chart components following TDD methodology.

These tests validate Stage 2 dashboard visualizations including
executive summary charts, rule performance analytics, and dataset quality insights.
Now in GREEN phase - testing implemented functionality.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Mock Plotly to avoid import errors in testing
@pytest.fixture
def mock_plotly():
    with patch('plotly.express') as mock_px, \
         patch('plotly.graph_objects') as mock_go:
        mock_px.pie.return_value = Mock()
        mock_px.line.return_value = Mock()
        mock_px.bar.return_value = Mock()
        mock_px.imshow.return_value = Mock()
        mock_px.histogram.return_value = Mock()
        mock_go.Indicator.return_value = Mock()
        mock_go.Figure.return_value = Mock()
        yield mock_px, mock_go


# Sample test data for dashboard charts
@pytest.fixture
def sample_dashboard_data():
    """Sample processed data for dashboard testing."""
    return {
        'summary_data': pd.DataFrame({
            'source': ['S1', 'S1', 'S2', 'S2'],
            'rule_code': ['R001', 'R002', 'R001', 'R003'],
            'rule_name': ['Data Completeness', 'Format Validation', 'Data Completeness', 'Range Check'],
            'rule_category': ['completeness', 'format', 'completeness', 'validity'],
            'total_records': [1000, 800, 1200, 900],
            'pass_count': [950, 780, 1150, 850],
            'fail_count': [50, 20, 50, 50],
            'pass_rate': [0.95, 0.975, 0.958, 0.944],
            'fail_rate': [0.05, 0.025, 0.042, 0.056],
            'business_date': ['2024-01-15', '2024-01-15', '2024-01-15', '2024-01-15']
        }),
        'aggregated_metrics': {
            'total_rules': 4,
            'total_datasets': 2,
            'overall_pass_rate': 0.946,
            'overall_fail_rate': 0.054,
            'latest_execution': '2024-01-15'
        }
    }


class TestDashboardCharts:
    """Test suite for dashboard chart generation components."""
    
    def test_create_quality_score_gauge_returns_figure(self, mock_plotly, sample_dashboard_data):
        """Test that quality score gauge creation returns a valid figure."""
        # GREEN: Test that function exists and returns expected structure
        from src.data_quality_summarizer.ui.visualizations.dashboard_charts import create_quality_score_gauge
        
        result = create_quality_score_gauge(sample_dashboard_data)
        assert result is not None
        # Verify it's a plotly figure-like object with traces
        assert hasattr(result, 'add_trace')
    
    def test_create_rule_category_pie_chart_returns_figure(self, mock_plotly, sample_dashboard_data):
        """Test that rule category pie chart creation returns a valid figure."""
        # GREEN: Test that function exists and returns expected structure
        from src.data_quality_summarizer.ui.visualizations.dashboard_charts import create_rule_category_pie_chart
        
        result = create_rule_category_pie_chart(sample_dashboard_data)
        assert result is not None
        # Mock plotly should return Mock object
        mock_plotly[0].pie.assert_called_once()
    
    def test_create_trend_line_chart_returns_figure(self, mock_plotly, sample_dashboard_data):
        """Test that trend line chart creation returns a valid figure."""
        # GREEN: Test that function exists and returns expected structure
        from src.data_quality_summarizer.ui.visualizations.dashboard_charts import create_trend_line_chart
        
        result = create_trend_line_chart(sample_dashboard_data)
        assert result is not None
        # Mock plotly should return Mock object
        mock_plotly[0].line.assert_called_once()
    
    def test_create_key_metrics_cards_returns_dict(self, sample_dashboard_data):
        """Test that key metrics cards creation returns a valid dictionary."""
        # GREEN: Test that function exists and returns expected structure
        from src.data_quality_summarizer.ui.visualizations.dashboard_charts import create_key_metrics_cards
        
        result = create_key_metrics_cards(sample_dashboard_data)
        assert isinstance(result, dict)
        # Verify expected keys are present
        expected_keys = ['total_rules', 'total_datasets', 'overall_pass_rate', 'latest_execution']
        for key in expected_keys:
            assert key in result
        
        # Verify values match input data
        assert result['total_rules'] == 4
        assert result['total_datasets'] == 2
        assert result['overall_pass_rate'] == 0.946


class TestRulePerformanceCharts:
    """Test suite for rule performance analytics charts."""
    
    def test_create_rule_ranking_bar_chart_returns_figure(self, mock_plotly, sample_dashboard_data):
        """Test that rule ranking bar chart creation returns a valid figure."""
        # GREEN: Test that function exists and returns expected structure
        from src.data_quality_summarizer.ui.visualizations.rule_analytics import create_rule_ranking_bar_chart
        
        result = create_rule_ranking_bar_chart(sample_dashboard_data)
        assert result is not None
        # Mock plotly should be called
        mock_plotly[0].bar.assert_called_once()
    
    def test_create_performance_heatmap_returns_figure(self, mock_plotly, sample_dashboard_data):
        """Test that performance heatmap creation returns a valid figure."""
        # GREEN: Test that function exists and returns expected structure
        from src.data_quality_summarizer.ui.visualizations.rule_analytics import create_performance_heatmap
        
        result = create_performance_heatmap(sample_dashboard_data)
        assert result is not None
        # Mock plotly should be called
        mock_plotly[0].imshow.assert_called_once()


class TestDatasetAnalysisCharts:
    """Test suite for dataset quality analysis charts."""
    
    def test_create_dataset_health_indicators_returns_dict(self, sample_dashboard_data):
        """Test that dataset health indicators creation returns a valid dictionary."""
        # GREEN: Test that function exists and returns expected structure
        from src.data_quality_summarizer.ui.visualizations.dataset_analysis import create_dataset_health_indicators
        
        result = create_dataset_health_indicators(sample_dashboard_data)
        assert isinstance(result, dict)
        assert 'indicators' in result
        assert 'status' in result
        assert result['status'] == 'success'
        assert isinstance(result['indicators'], list)
    
    def test_create_quality_distribution_histogram_returns_figure(self, mock_plotly, sample_dashboard_data):
        """Test that quality distribution histogram creation returns a valid figure."""
        # GREEN: Test that function exists and returns expected structure
        from src.data_quality_summarizer.ui.visualizations.dataset_analysis import create_quality_distribution_histogram
        
        result = create_quality_distribution_histogram(sample_dashboard_data)
        assert result is not None
        # Mock plotly should be called
        mock_plotly[0].histogram.assert_called_once()


class TestDashboardPages:
    """Test suite for complete dashboard page components."""
    
    def test_dashboard_page_display_exists_and_callable(self, sample_dashboard_data):
        """Test that dashboard page display function exists and is callable."""
        # GREEN: Test that function exists and can be called
        from src.data_quality_summarizer.ui.pages.dashboard import display_dashboard_page
        
        # Should not raise an exception when called
        try:
            display_dashboard_page(sample_dashboard_data)
        except Exception as e:
            # Allow ImportError for Streamlit in testing, but not other errors
            if "streamlit" not in str(e).lower():
                pytest.fail(f"Unexpected error: {e}")
    
    def test_rule_performance_page_display_exists_and_callable(self, sample_dashboard_data):
        """Test that rule performance page display function exists and is callable."""
        # GREEN: Test that function exists and can be called
        from src.data_quality_summarizer.ui.pages.rule_performance import display_rule_performance_page
        
        # Should not raise an exception when called
        try:
            display_rule_performance_page(sample_dashboard_data)
        except Exception as e:
            # Allow ImportError for Streamlit in testing, but not other errors
            if "streamlit" not in str(e).lower():
                pytest.fail(f"Unexpected error: {e}")
    
    def test_dataset_insights_page_display_exists_and_callable(self, sample_dashboard_data):
        """Test that dataset insights page display function exists and is callable."""
        # GREEN: Test that function exists and can be called
        from src.data_quality_summarizer.ui.pages.dataset_insights import display_dataset_insights_page
        
        # Should not raise an exception when called
        try:
            display_dataset_insights_page(sample_dashboard_data)
        except Exception as e:
            # Allow ImportError for Streamlit in testing, but not other errors
            if "streamlit" not in str(e).lower():
                pytest.fail(f"Unexpected error: {e}")


class TestDataTransformation:
    """Test suite for data transformation utilities for visualizations."""
    
    def test_transform_data_for_charts_returns_dict(self, sample_dashboard_data):
        """Test that data transformation for charts returns a valid dictionary."""
        # GREEN: Test that function exists and returns expected structure
        from src.data_quality_summarizer.ui.visualizations.chart_utils import transform_data_for_charts
        
        result = transform_data_for_charts(sample_dashboard_data)
        assert isinstance(result, dict)
        assert 'transformed' in result
        assert result['transformed'] is True
    
    def test_calculate_dashboard_metrics_returns_dict(self, sample_dashboard_data):
        """Test that dashboard metrics calculation returns a valid dictionary."""
        # GREEN: Test that function exists and returns expected structure
        from src.data_quality_summarizer.ui.visualizations.chart_utils import calculate_dashboard_metrics
        
        result = calculate_dashboard_metrics(sample_dashboard_data)
        assert isinstance(result, dict)
        assert 'calculated' in result
        assert result['calculated'] is True
        assert 'metrics' in result
    
    def test_functions_handle_empty_data_gracefully(self):
        """Test that all functions handle empty data without crashing."""
        from src.data_quality_summarizer.ui.visualizations.chart_utils import (
            transform_data_for_charts, calculate_dashboard_metrics
        )
        
        # Test with empty data
        empty_data = {}
        
        result1 = transform_data_for_charts(empty_data)
        assert isinstance(result1, dict)
        
        result2 = calculate_dashboard_metrics(empty_data)
        assert isinstance(result2, dict)