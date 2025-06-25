/**
 * Type definitions for API responses.
 * These match the actual structure returned by the backend.
 */

export interface ApiSummaryRow {
  // Identity fields
  source: string
  tenant_id: string
  dataset_uuid: string
  dataset_name: string
  rule_code: string
  
  // Rule metadata (from backend conversion)
  rule_name: string
  rule_type: string
  dimension: string
  rule_description: string
  category: string  // Maps to -> rule_category
  
  // Counts (from aggregation)
  pass_count_total: number  // Maps to -> total_passes
  fail_count_total: number  // Maps to -> total_failures
  pass_count_1m: number
  fail_count_1m: number
  pass_count_3m: number
  fail_count_3m: number
  pass_count_12m: number
  fail_count_12m: number
  
  // Rates (calculated)
  fail_rate_total: number  // Maps to -> overall_fail_rate
  fail_rate_1m: number
  fail_rate_3m: number
  fail_rate_12m: number
  
  // Status
  trend_flag: string
  business_date_latest: string  // Maps to -> latest_business_date
  last_execution_level: string
  
  // Additional fields from backend
  dataset_record_count_latest: number
  filtered_record_count_latest: number
}

export interface ApiProcessingResult {
  summary_data: ApiSummaryRow[]
  nl_summary: string[]
  total_rows_processed: number
  processing_time_seconds: number
  memory_usage_mb: number
  unique_datasets: number
  unique_rules: number
  time_range: {
    start_date: string
    end_date: string
  }
}