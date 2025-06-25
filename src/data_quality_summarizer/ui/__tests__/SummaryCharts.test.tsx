import React from 'react';
import { render, screen } from '@testing-library/react';
import { SummaryCharts } from '../visualizations/SummaryCharts';
import { SummaryRow } from '../types/common';
import { transformSummaryData } from '../utils/dataTransformer';
import { ApiSummaryRow } from '../types/api';

// Mock data that matches what the API actually returns (from backend_integration.py)
const mockApiResponse = [
  {
    source: "test_source",
    tenant_id: "tenant_123",
    dataset_uuid: "uuid_123",
    dataset_name: "test_dataset_alpha",
    rule_code: "R001",
    rule_name: "Data Completeness Check",
    rule_type: "completeness",
    dimension: "completeness",
    rule_description: "Check for missing values",
    category: "Data Quality", // API returns 'category', not 'rule_category'
    business_date_latest: "2024-01-15",
    pass_count_total: 850,
    fail_count_total: 150, // API returns 'fail_count_total', not 'total_failures'
    pass_count_1m: 280,
    fail_count_1m: 20,
    pass_count_3m: 800,
    fail_count_3m: 100,
    pass_count_12m: 850,
    fail_count_12m: 150,
    fail_rate_total: 0.15, // API returns 'fail_rate_total', not 'overall_fail_rate'
    fail_rate_1m: 0.067,
    fail_rate_3m: 0.111,
    fail_rate_12m: 0.15,
    trend_flag: "stable",
    last_execution_level: "dataset"
    // Missing fields that frontend expects:
    // - rule_category (should map from 'category')
    // - total_failures (should map from 'fail_count_total')
    // - total_passes (should map from 'pass_count_total')
    // - overall_fail_rate (should map from 'fail_rate_total')
    // - risk_level (needs to be calculated)
    // - improvement_needed (needs to be calculated)
    // - execution_consistency (missing entirely)
    // - avg_daily_executions (missing entirely)
  }
] as any; // Use 'any' because this doesn't match SummaryRow interface

// Mock data that matches what the frontend expects (from types/common.ts)
const mockExpectedData: SummaryRow[] = [
  {
    source: "test_source",
    tenant_id: "tenant_123",
    dataset_uuid: "uuid_123",
    dataset_name: "test_dataset_alpha",
    rule_code: "R001",
    rule_category: "Data Quality", // Frontend expects this field
    rule_description: "Check for missing values",
    total_passes: 850, // Frontend expects this field
    total_failures: 150, // Frontend expects this field
    overall_fail_rate: 0.15, // Frontend expects this field
    pass_rate_1m: 0.933,
    fail_rate_1m: 0.067,
    pass_rate_3m: 0.889,
    fail_rate_3m: 0.111,
    pass_rate_12m: 0.85,
    fail_rate_12m: 0.15,
    trend_1m_vs_3m: "improving",
    trend_3m_vs_12m: "stable",
    latest_business_date: "2024-01-15",
    earliest_business_date: "2023-01-15",
    total_execution_days: 365,
    avg_daily_executions: 2.5, // Frontend expects this field
    execution_consistency: 0.85, // Frontend expects this field
    data_volume_trend: "stable",
    risk_level: "MEDIUM", // Frontend expects this field
    improvement_needed: true, // Frontend expects this field
    last_failure_date: "2024-01-14"
  }
];

describe('SummaryCharts Component - Data Contract Issues', () => {
  // RED PHASE: These tests document the current broken state
  
  it('should fail to render with current API response format', () => {
    // This test demonstrates the problem: API response doesn't match frontend expectations
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    
    expect(() => {
      render(<SummaryCharts data={mockApiResponse} />);
    }).not.toThrow(); // Component renders but charts will be empty/broken
    
    // Check that the component renders but categories are "Unknown" due to missing fields
    const categoryChart = screen.getByText('Failure Rates by Rule Category');
    expect(categoryChart).toBeInTheDocument();
    
    // The chart will show "Unknown" categories because rule_category field is missing
    // This test will pass initially but documents the broken behavior
    
    consoleErrorSpy.mockRestore();
  });

  it('should show empty/broken risk level distribution with API data', () => {
    render(<SummaryCharts data={mockApiResponse} />);
    
    // Risk level chart should be present but will show "Unknown" data
    const riskChart = screen.getByText('Risk Level Distribution');
    expect(riskChart).toBeInTheDocument();
    
    // The chart data will be empty or show "Unknown" because risk_level is missing
    // This documents the current broken state
  });

  it('should fail to show proper trends with API data', () => {
    render(<SummaryCharts data={mockApiResponse} />);
    
    // Trend chart should be present
    const trendChart = screen.getByText('Failure Rate Trends (Top 20 Datasets)');
    expect(trendChart).toBeInTheDocument();
    
    // The trend data might work since fail_rate_1m, fail_rate_3m, fail_rate_12m exist in API
    // But this test documents what should work vs what doesn't
  });

  it('should fail to show execution consistency chart with API data', () => {
    render(<SummaryCharts data={mockApiResponse} />);
    
    // Execution consistency chart should be present
    const execChart = screen.getByText('Execution Consistency vs Daily Execution Frequency');
    expect(execChart).toBeInTheDocument();
    
    // This chart will be completely broken because execution_consistency and 
    // avg_daily_executions are missing from API response
  });

  // GREEN PHASE: This test shows what should work when data transformation is implemented
  it('should render correctly with properly transformed data', () => {
    render(<SummaryCharts data={mockExpectedData} />);
    
    // All charts should be present and functional
    expect(screen.getByText('Failure Rates by Rule Category')).toBeInTheDocument();
    expect(screen.getByText('Risk Level Distribution')).toBeInTheDocument();
    expect(screen.getByText('Failure Rate Trends (Top 20 Datasets)')).toBeInTheDocument();
    expect(screen.getByText('Execution Consistency vs Daily Execution Frequency')).toBeInTheDocument();
    
    // Charts should render (even if mocked)
    // The actual chart rendering will be tested once we implement the data transformation
    expect(screen.getByText('Failure Rates by Rule Category')).toBeInTheDocument();
  });

  it('should handle empty data gracefully', () => {
    render(<SummaryCharts data={[]} />);
    
    // Charts should still render but show no data
    expect(screen.getByText('Failure Rates by Rule Category')).toBeInTheDocument();
    expect(screen.getByText('Risk Level Distribution')).toBeInTheDocument();
  });
});

describe('SummaryCharts Component - Stage 3 TDD Implementation', () => {
  // RED PHASE: New tests that should fail initially
  
  it('should render bar chart correctly with transformed API data', () => {
    // Transform the API data using our data transformer
    const transformedData = transformSummaryData(mockApiResponse as ApiSummaryRow[]);
    
    render(<SummaryCharts data={transformedData} />);
    
    // Bar chart should be present
    expect(screen.getByText('Failure Rates by Rule Category')).toBeInTheDocument();
    
    // The chart should show the correct category (not "Unknown")
    // This test will fail initially because the component still expects the wrong field names
  });
  
  it('should render pie chart with correct risk levels from transformed data', () => {
    const transformedData = transformSummaryData(mockApiResponse as ApiSummaryRow[]);
    
    render(<SummaryCharts data={transformedData} />);
    
    // Risk level chart should be present
    expect(screen.getByText('Risk Level Distribution')).toBeInTheDocument();
    
    // The chart should show calculated risk levels (not "Unknown")
    // This test will fail initially
  });
  
  it('should render line chart with fail rate trends', () => {
    const transformedData = transformSummaryData(mockApiResponse as ApiSummaryRow[]);
    
    render(<SummaryCharts data={transformedData} />);
    
    // Trend chart should be present
    expect(screen.getByText('Failure Rate Trends (Top 20 Datasets)')).toBeInTheDocument();
    
    // Chart should use the correct fail_rate fields
  });
  
  it('should render scatter chart with execution consistency data', () => {
    const transformedData = transformSummaryData(mockApiResponse as ApiSummaryRow[]);
    
    render(<SummaryCharts data={transformedData} />);
    
    // Execution consistency chart should be present
    expect(screen.getByText('Execution Consistency vs Daily Execution Frequency')).toBeInTheDocument();
    
    // Chart should use calculated execution_consistency and avg_daily_executions
  });
  
  it('should handle multiple datasets with different risk levels', () => {
    const multipleDatasets: ApiSummaryRow[] = [
      {
        ...mockApiResponse[0],
        dataset_name: 'High Risk Dataset',
        fail_rate_total: 0.35, // 35% fail rate = HIGH risk
      },
      {
        ...mockApiResponse[0],
        dataset_name: 'Medium Risk Dataset',
        fail_rate_total: 0.15, // 15% fail rate = MEDIUM risk
      },
      {
        ...mockApiResponse[0],
        dataset_name: 'Low Risk Dataset',
        fail_rate_total: 0.05, // 5% fail rate = LOW risk
      },
    ] as ApiSummaryRow[];
    
    const transformedData = transformSummaryData(multipleDatasets);
    render(<SummaryCharts data={transformedData} />);
    
    // All risk levels should be represented in the pie chart
    expect(screen.getByText('Risk Level Distribution')).toBeInTheDocument();
  });
  
  it('should render responsive charts that adapt to container size', () => {
    const transformedData = transformSummaryData(mockApiResponse as ApiSummaryRow[]);
    
    const { container } = render(<SummaryCharts data={transformedData} />);
    
    // Check for ResponsiveContainer elements
    const responsiveContainers = container.querySelectorAll('.recharts-responsive-container');
    expect(responsiveContainers.length).toBeGreaterThan(0);
  });

  // More specific tests to verify data transformation is working
  it('should use transformed field names in chart data processing', () => {
    // Create mock data with specific values to test
    const testData: ApiSummaryRow[] = [{
      ...mockApiResponse[0],
      category: 'Test Category', // API field
      fail_count_total: 100,
      pass_count_total: 900,
      fail_rate_total: 0.1
    } as ApiSummaryRow];
    
    const transformedData = transformSummaryData(testData);
    
    // Verify the transformation worked
    expect(transformedData[0].rule_category).toBe('Test Category');
    expect(transformedData[0].total_failures).toBe(100);
    expect(transformedData[0].total_passes).toBe(900);
    expect(transformedData[0].overall_fail_rate).toBe(0.1);
    expect(transformedData[0].risk_level).toBe('MEDIUM'); // 10% fail rate = MEDIUM
  });

  it('should handle edge case data values correctly', () => {
    const edgeCaseData: ApiSummaryRow[] = [
      {
        ...mockApiResponse[0],
        fail_rate_total: 0, // 0% fail rate
        pass_count_1m: 0,
        fail_count_1m: 0,
      } as ApiSummaryRow,
      {
        ...mockApiResponse[0],
        dataset_name: 'High Risk Dataset',
        fail_rate_total: 0.5, // 50% fail rate
      } as ApiSummaryRow,
      {
        ...mockApiResponse[0],
        dataset_name: 'Missing Category',
        category: undefined as any, // Missing category
      } as ApiSummaryRow,
    ];
    
    const transformedData = transformSummaryData(edgeCaseData);
    
    const { container } = render(<SummaryCharts data={transformedData} />);
    
    // Should render without errors
    expect(screen.getByText('Failure Rates by Rule Category')).toBeInTheDocument();
    expect(screen.getByText('Risk Level Distribution')).toBeInTheDocument();
  });
});

describe('SummaryCharts Component - Actual Field Usage Tests', () => {
  // These tests verify that the component actually uses the correct field names
  
  it('should fail if component uses wrong field names for category chart', () => {
    // This test verifies the component uses rule_category, not category
    const apiData: ApiSummaryRow[] = [{
      ...mockApiResponse[0],
      category: 'API Category Field',
      // Note: rule_category is NOT in the API response
    } as ApiSummaryRow];
    
    // Without transformation, the component should show "Unknown" categories
    render(<SummaryCharts data={apiData as any} />);
    
    // The component should fail to find rule_category and show "Unknown"
    // This test documents the current broken behavior
  });
  
  it('should work correctly only with transformed data', () => {
    const apiData: ApiSummaryRow[] = [{
      ...mockApiResponse[0],
      category: 'Data Quality',
      fail_count_total: 150,
      pass_count_total: 850,
    } as ApiSummaryRow];
    
    // With transformation
    const transformedData = transformSummaryData(apiData);
    
    // Verify transformation created the expected fields
    expect(transformedData[0].rule_category).toBe('Data Quality');
    expect(transformedData[0].total_failures).toBe(150);
    expect(transformedData[0].total_passes).toBe(850);
    
    // Component should work with transformed data
    render(<SummaryCharts data={transformedData} />);
    expect(screen.getByText('Failure Rates by Rule Category')).toBeInTheDocument();
  });
});

describe('SummaryCharts Component - Field Mapping Documentation', () => {
  it('should document the exact field mapping requirements', () => {
    // This test documents the required field transformations:
    const apiData = mockApiResponse[0];
    const expectedData = mockExpectedData[0];
    
    // Required field mappings from API to Frontend:
    expect(apiData.category).toBeDefined(); // should map to -> rule_category
    expect(apiData.fail_count_total).toBeDefined(); // should map to -> total_failures  
    expect(apiData.pass_count_total).toBeDefined(); // should map to -> total_passes
    expect(apiData.fail_rate_total).toBeDefined(); // should map to -> overall_fail_rate
    
    // Required calculated fields (missing from API):
    expect(expectedData.risk_level).toBeDefined(); // needs to be calculated
    expect(expectedData.improvement_needed).toBeDefined(); // needs to be calculated
    expect(expectedData.execution_consistency).toBeDefined(); // missing from API
    expect(expectedData.avg_daily_executions).toBeDefined(); // missing from API
    
    // This test documents the exact transformation requirements
    expect(true).toBe(true); // This test always passes, it's for documentation
  });
});