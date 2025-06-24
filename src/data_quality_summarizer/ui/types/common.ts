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

export interface SummaryRow {
  source: string
  tenant_id: string
  dataset_uuid: string
  dataset_name: string
  rule_code: string
  rule_category: string
  rule_description: string
  total_passes: number
  total_failures: number
  overall_fail_rate: number
  pass_rate_1m: number
  fail_rate_1m: number
  pass_rate_3m: number
  fail_rate_3m: number
  pass_rate_12m: number
  fail_rate_12m: number
  trend_1m_vs_3m: string
  trend_3m_vs_12m: string
  latest_business_date: string
  earliest_business_date: string
  total_execution_days: number
  avg_daily_executions: number
  execution_consistency: number
  data_volume_trend: string
  risk_level: string
  improvement_needed: boolean
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