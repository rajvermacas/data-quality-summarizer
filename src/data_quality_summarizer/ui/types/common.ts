import { ApiSummaryRow } from './api'

export interface ProcessingStatus {
  status: 'idle' | 'processing' | 'completed' | 'error'
  progress: number
  message: string
}

export interface ProcessingResult {
  summary_data: SummaryRow[]
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

/**
 * Enhanced SummaryRow interface for UI consumption.
 * This extends the API response with calculated fields and UI-friendly names.
 */
export interface SummaryRow extends ApiSummaryRow {
  // UI-friendly field mappings (for backward compatibility)
  rule_category: string            // = category
  total_passes: number             // = pass_count_total
  total_failures: number           // = fail_count_total
  overall_fail_rate: number        // = fail_rate_total
  latest_business_date: string     // = business_date_latest
  
  // Calculated fields for UI
  risk_level: 'HIGH' | 'MEDIUM' | 'LOW' | 'UNKNOWN'
  improvement_needed: boolean
  avg_daily_executions: number
  execution_consistency: number
  
  // Additional UI fields (to be calculated or defaulted)
  pass_rate_1m: number
  pass_rate_3m: number
  pass_rate_12m: number
  trend_1m_vs_3m: string
  trend_3m_vs_12m: string
  earliest_business_date: string
  total_execution_days: number
  data_volume_trend: string
  last_failure_date: string | null
}

export interface MLPipelineConfig {
  model_type: 'lightgbm' | 'xgboost'
  features: string[]
  target: string
  test_size: number
  random_state: number
  hyperparameters: Record<string, any>
}

export interface MLModelResult {
  model_id: string
  model_type: string
  training_score: number
  validation_score: number
  test_score: number
  feature_importance: Record<string, number>
  training_time_seconds: number
  model_path: string
}

export interface PredictionRequest {
  dataset_uuid: string
  rule_code: string
  business_date: string
  features?: Record<string, any>
}

export interface PredictionResult {
  prediction: number
  probability: number
  confidence_interval: [number, number]
  feature_contributions: Record<string, number>
}