/**
 * Runtime validation utilities using Zod.
 * This module provides schema validation for API responses to ensure
 * data integrity and catch API contract violations at runtime.
 */

import { z } from 'zod'
import { ApiSummaryRow } from '../types/api'

// Define the Zod schema for ApiSummaryRow
export const ApiSummaryRowSchema = z.object({
  // Identity fields (required)
  source: z.string(),
  tenant_id: z.string(),
  dataset_uuid: z.string(),
  dataset_name: z.string(),
  rule_code: z.string(),
  
  // Rule metadata (optional with defaults)
  rule_name: z.string().default(''),
  rule_type: z.string().default(''),
  dimension: z.string().default(''),
  rule_description: z.string().default(''),
  category: z.string().default(''),
  
  // Counts (optional with defaults)
  pass_count_total: z.number().default(0),
  fail_count_total: z.number().default(0),
  pass_count_1m: z.number().default(0),
  fail_count_1m: z.number().default(0),
  pass_count_3m: z.number().default(0),
  fail_count_3m: z.number().default(0),
  pass_count_12m: z.number().default(0),
  fail_count_12m: z.number().default(0),
  
  // Rates (optional with defaults)
  fail_rate_total: z.number().default(0),
  fail_rate_1m: z.number().default(0),
  fail_rate_3m: z.number().default(0),
  fail_rate_12m: z.number().default(0),
  
  // Status (optional with defaults)
  trend_flag: z.string().default('unknown'),
  business_date_latest: z.string().default(''),
  last_execution_level: z.string().default(''),
  
  // Additional fields (optional)
  dataset_record_count_latest: z.number().optional(),
  filtered_record_count_latest: z.number().optional()
})

// Schema for the complete API response
export const ApiProcessingResultSchema = z.object({
  summary_data: z.array(ApiSummaryRowSchema),
  nl_summary: z.array(z.string()),
  total_rows_processed: z.number(),
  processing_time_seconds: z.number(),
  memory_usage_mb: z.number(),
  unique_datasets: z.number(),
  unique_rules: z.number(),
  time_range: z.object({
    start_date: z.string(),
    end_date: z.string()
  })
})

/**
 * Validate a single API summary row.
 * Returns the validated and typed data or throws a validation error.
 */
export function validateApiSummaryRow(data: unknown): ApiSummaryRow {
  try {
    return ApiSummaryRowSchema.parse(data)
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error('API response validation failed:', error.errors)
      throw new Error(`Invalid API response format: ${error.errors.map(e => `${e.path.join('.')}: ${e.message}`).join(', ')}`)
    }
    throw error
  }
}

/**
 * Validate an array of API summary rows.
 * Returns the validated data or throws if any row is invalid.
 */
export function validateApiSummaryData(data: unknown[]): ApiSummaryRow[] {
  // Handle null/undefined/non-array input
  if (!data || !Array.isArray(data)) {
    console.warn('validateApiSummaryData: Invalid input, expected array but got:', data)
    return []
  }
  
  // Handle empty array
  if (data.length === 0) {
    console.log('validateApiSummaryData: Empty array provided')
    return []
  }
  
  return data.map((row, index) => {
    try {
      return validateApiSummaryRow(row)
    } catch (error) {
      console.error(`Validation failed for row ${index}:`, error)
      throw new Error(`Validation failed for row ${index}: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  })
}

/**
 * Validate the complete API processing result.
 * Returns the validated result or throws a validation error.
 */
export function validateApiProcessingResult(data: unknown): z.infer<typeof ApiProcessingResultSchema> {
  try {
    return ApiProcessingResultSchema.parse(data)
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error('API processing result validation failed:', error.errors)
      throw new Error(`Invalid API response: ${error.errors.map(e => `${e.path.join('.')}: ${e.message}`).join(', ')}`)
    }
    throw error
  }
}

/**
 * Safe validation that returns a result object instead of throwing.
 * Useful for graceful error handling in UI components.
 */
export function safeValidateApiSummaryRow(data: unknown): { success: true; data: ApiSummaryRow } | { success: false; error: string } {
  try {
    const validated = validateApiSummaryRow(data)
    return { success: true, data: validated }
  } catch (error) {
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown validation error' 
    }
  }
}

/**
 * Type guard to check if data matches ApiSummaryRow structure.
 * This is a lightweight check without full validation.
 */
export function isApiSummaryRow(data: unknown): data is ApiSummaryRow {
  if (!data || typeof data !== 'object') return false
  
  const obj = data as Record<string, unknown>
  
  // Check required fields exist
  return (
    typeof obj.source === 'string' &&
    typeof obj.tenant_id === 'string' &&
    typeof obj.dataset_uuid === 'string' &&
    typeof obj.dataset_name === 'string' &&
    typeof obj.rule_code === 'string'
  )
}