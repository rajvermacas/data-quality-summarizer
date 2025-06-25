import React from 'react';
import { render, screen } from '@testing-library/react';
import { ResultsPage } from '../pages/ResultsPage';
import { ProcessingResult } from '../types/common';
import { transformSummaryData } from '../utils/dataTransformer';
import { ApiProcessingResult, ApiSummaryRow } from '../types/api';

describe('Integration Tests - Data Transformation Flow', () => {
  it('should render ResultsPage correctly with transformed API data', () => {
    // Mock API response (what backend actually returns)
    const mockApiData: ApiSummaryRow[] = [{
      source: "test_source",
      tenant_id: "tenant_123",
      dataset_uuid: "uuid_123",
      dataset_name: "test_dataset",
      rule_code: "R001",
      rule_name: "Data Completeness Check",
      rule_type: "completeness",
      dimension: "completeness",
      rule_description: "Check for missing values",
      category: "Data Quality", // API field
      business_date_latest: "2024-01-15",
      pass_count_total: 850,
      fail_count_total: 150, // API field
      pass_count_1m: 280,
      fail_count_1m: 20,
      pass_count_3m: 800,
      fail_count_3m: 100,
      pass_count_12m: 850,
      fail_count_12m: 150,
      fail_rate_total: 0.15, // API field
      fail_rate_1m: 0.067,
      fail_rate_3m: 0.111,
      fail_rate_12m: 0.15,
      trend_flag: "stable",
      last_execution_level: "dataset",
      dataset_record_count_latest: 1000,
      filtered_record_count_latest: 1000
    }];

    // Transform the data as App.tsx should do
    const transformedData = transformSummaryData(mockApiData);

    // Create the ProcessingResult with transformed data
    const mockResult: ProcessingResult = {
      summary_data: transformedData,
      nl_summary: ["Test natural language summary"],
      total_rows_processed: 1000,
      processing_time_seconds: 0.5,
      memory_usage_mb: 50,
      unique_datasets: 1,
      unique_rules: 1,
      time_range: {
        start_date: "2024-01-01",
        end_date: "2024-01-15"
      }
    };

    // Render ResultsPage with transformed data
    render(
      <ResultsPage 
        result={mockResult}
        onStartOver={() => {}}
        onViewMLPipeline={() => {}}
      />
    );

    // Verify the page renders without errors
    expect(screen.getByText(/Processing Results/i)).toBeInTheDocument();
    
    // Verify overview metrics work (these use transformed fields)
    expect(screen.getByText(/Health Score/i)).toBeInTheDocument();
    expect(screen.getByText(/High Risk Rules/i)).toBeInTheDocument();
    
    // The health score should be calculated as (1 - avgFailRate) * 100
    // With 15% fail rate, health score should be 85%
    expect(screen.getByText('85%')).toBeInTheDocument();
  });

  it('should calculate metrics correctly with transformed data', () => {
    // Create test data with specific values
    const mockApiData: ApiSummaryRow[] = [
      {
        source: "test",
        tenant_id: "123",
        dataset_uuid: "uuid-1",
        dataset_name: "High Risk Dataset",
        rule_code: "R001",
        rule_name: "Rule 1",
        rule_type: "quality",
        dimension: "completeness",
        rule_description: "Test",
        category: "Data Quality",
        business_date_latest: "2024-01-15",
        pass_count_total: 500,
        fail_count_total: 500, // 50% fail rate = HIGH risk
        pass_count_1m: 100,
        fail_count_1m: 100,
        pass_count_3m: 300,
        fail_count_3m: 300,
        pass_count_12m: 500,
        fail_count_12m: 500,
        fail_rate_total: 0.5,
        fail_rate_1m: 0.5,
        fail_rate_3m: 0.5,
        fail_rate_12m: 0.5,
        trend_flag: "stable",
        last_execution_level: "dataset",
        dataset_record_count_latest: 1000,
        filtered_record_count_latest: 1000
      },
      {
        source: "test",
        tenant_id: "123",
        dataset_uuid: "uuid-2",
        dataset_name: "Medium Risk Dataset",
        rule_code: "R002",
        rule_name: "Rule 2",
        rule_type: "quality",
        dimension: "validity",
        rule_description: "Test",
        category: "Data Validity",
        business_date_latest: "2024-01-15",
        pass_count_total: 850,
        fail_count_total: 150, // 15% fail rate = MEDIUM risk
        pass_count_1m: 280,
        fail_count_1m: 20,
        pass_count_3m: 800,
        fail_count_3m: 100,
        pass_count_12m: 850,
        fail_count_12m: 150,
        fail_rate_total: 0.15,
        fail_rate_1m: 0.067,
        fail_rate_3m: 0.111,
        fail_rate_12m: 0.15,
        trend_flag: "worsening", // MEDIUM + worsening = improvement needed
        last_execution_level: "dataset",
        dataset_record_count_latest: 1000,
        filtered_record_count_latest: 1000
      }
    ];

    const transformedData = transformSummaryData(mockApiData);
    
    const mockResult: ProcessingResult = {
      summary_data: transformedData,
      nl_summary: [],
      total_rows_processed: 2000,
      processing_time_seconds: 1.0,
      memory_usage_mb: 100,
      unique_datasets: 2,
      unique_rules: 2,
      time_range: {
        start_date: "2024-01-01",
        end_date: "2024-01-15"
      }
    };

    render(
      <ResultsPage 
        result={mockResult}
        onStartOver={() => {}}
        onViewMLPipeline={() => {}}
      />
    );

    // Verify specific metric calculations
    // High risk rules: 1 (the 50% fail rate dataset)
    const highRiskMetric = screen.getByText('High Risk Rules').parentElement;
    expect(highRiskMetric?.querySelector('h3')?.textContent).toBe('1');
    
    // Average fail rate: (0.5 + 0.15) / 2 = 0.325 = 32.5%
    // Health score: (1 - 0.325) * 100 = 67.5% ≈ 68%
    const healthScore = screen.getByText('Health Score').parentElement;
    expect(healthScore?.querySelector('h3')?.textContent).toBe('68%');
  });
});