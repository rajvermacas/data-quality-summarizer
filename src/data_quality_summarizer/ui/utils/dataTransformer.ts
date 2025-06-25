/**
 * Data transformation utilities for converting API responses to UI-friendly format.
 * This module handles the field mapping and calculated field generation required
 * for the chart components and data tables.
 */

import { ApiSummaryRow } from '../types/api'
import { SummaryRow } from '../types/common'
import { validateApiSummaryData, safeValidateApiSummaryRow } from './validation'

// Re-export types for convenience
export type { ApiSummaryRow } from '../types/api'
export type EnhancedSummaryRow = SummaryRow

/**
 * Calculate risk level based on fail rate.
 * LOW: < 10% fail rate
 * MEDIUM: 10-20% fail rate  
 * HIGH: > 20% fail rate
 * UNKNOWN: Invalid or missing data
 */
export function calculateRiskLevel(failRate: number | undefined | null): 'HIGH' | 'MEDIUM' | 'LOW' | 'UNKNOWN' {
  if (failRate === undefined || failRate === null || isNaN(failRate)) {
    return 'UNKNOWN'
  }
  
  if (failRate < 0.1) return 'LOW'
  if (failRate <= 0.2) return 'MEDIUM'
  return 'HIGH'
}

/**
 * Determine if improvement is needed based on fail rate and trend.
 * Improvement is needed if:
 * - Fail rate is HIGH (> 20%) regardless of trend
 * - Fail rate is MEDIUM (10-20%) and trend is worsening
 * - Never needed if trend is improving
 */
export function calculateImprovementNeeded(
  failRate: number | undefined | null,
  trend: string | undefined | null
): boolean {
  if (failRate === undefined || failRate === null) {
    return false
  }
  
  // If trend is improving, no improvement needed
  if (trend === 'improving') {
    return false
  }
  
  // High fail rate always needs improvement (unless improving)
  if (failRate > 0.2) {
    return true
  }
  
  // Medium fail rate with worsening trend needs improvement
  if (failRate >= 0.1 && trend === 'worsening') {
    return true
  }
  
  return false
}

/**
 * Calculate execution consistency based on monthly execution counts.
 * Returns a value between 0 and 1, where 1 means perfectly consistent.
 * Uses coefficient of variation (standard deviation / mean).
 */
export function calculateExecutionConsistency(
  count1m: number | undefined | null,
  count3m: number | undefined | null,
  count12m: number | undefined | null
): number {
  // Handle missing values
  if (count1m === undefined || count1m === null ||
      count3m === undefined || count3m === null ||
      count12m === undefined || count12m === null) {
    return 0
  }
  
  // If all are zero, return 0 consistency
  if (count1m === 0 && count3m === 0 && count12m === 0) {
    return 0
  }
  
  // For the 1m and 3m counts, we need to interpret them correctly:
  // - count1m is the total for 1 month
  // - count3m is the total for 3 months
  // - count12m is the total for 12 months
  
  // If the counts are all the same, it means perfect consistency
  // (e.g., 100 per month for all periods)
  if (count1m === count3m && count3m === count12m) {
    return 1
  }
  
  // Calculate monthly averages from the totals
  const avg1m = count1m  // 1-month total is already monthly
  const avg3m = count3m / 3  // 3-month total divided by 3
  const avg12m = count12m / 12  // 12-month total divided by 12
  
  // If any period has data but others don't, it's inconsistent
  const nonZeroCount = [avg1m, avg3m, avg12m].filter(v => v > 0).length
  if (nonZeroCount === 1) {
    return 0.3  // Low consistency if only one period has data
  }
  
  // Calculate mean and standard deviation
  const values = [avg1m, avg3m, avg12m]
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  
  if (mean === 0) return 0
  
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length
  const stdDev = Math.sqrt(variance)
  const coefficientOfVariation = stdDev / mean
  
  // Convert to consistency score (1 - CV, bounded between 0 and 1)
  // Using a scaling factor to make the score more intuitive
  const consistencyScore = Math.exp(-coefficientOfVariation * 2)
  return Math.max(0, Math.min(1, consistencyScore))
}

/**
 * Transform API summary data to UI-friendly format.
 * This is the main transformation function that handles:
 * - Field name mapping
 * - Calculated field generation
 * - Default value handling
 * - Runtime validation
 */
export function transformSummaryData(apiData: ApiSummaryRow[], skipValidation = false): SummaryRow[] {
  // Validate the data if not skipped (useful for testing)
  const validatedData = skipValidation ? apiData : validateApiSummaryData(apiData)
  
  return validatedData.map(row => {
    // Calculate derived fields
    const riskLevel = calculateRiskLevel(row.fail_rate_total)
    const improvementNeeded = calculateImprovementNeeded(row.fail_rate_total, row.trend_flag)
    
    // Calculate average daily executions (based on 1-month data)
    const totalExecutions1m = (row.pass_count_1m || 0) + (row.fail_count_1m || 0)
    const avgDailyExecutions = totalExecutions1m / 30  // Assuming 30-day month
    
    // Calculate execution consistency
    const totalExecutions3m = (row.pass_count_3m || 0) + (row.fail_count_3m || 0)
    const totalExecutions12m = (row.pass_count_12m || 0) + (row.fail_count_12m || 0)
    const executionConsistency = calculateExecutionConsistency(
      totalExecutions1m,
      totalExecutions3m,
      totalExecutions12m
    )
    
    // Calculate pass rates (inverse of fail rates)
    const passRate1m = row.fail_rate_1m !== undefined ? 1 - row.fail_rate_1m : 0
    const passRate3m = row.fail_rate_3m !== undefined ? 1 - row.fail_rate_3m : 0
    const passRate12m = row.fail_rate_12m !== undefined ? 1 - row.fail_rate_12m : 0
    
    // Calculate trends
    const trend1mVs3m = calculateTrend(row.fail_rate_1m, row.fail_rate_3m)
    const trend3mVs12m = calculateTrend(row.fail_rate_3m, row.fail_rate_12m)
    
    // Create the enhanced summary row
    const enhancedRow: SummaryRow = {
      // Spread all original API fields
      ...row,
      
      // UI-friendly field mappings
      rule_category: row.category || '',
      total_passes: row.pass_count_total || 0,
      total_failures: row.fail_count_total || 0,
      overall_fail_rate: row.fail_rate_total || 0,
      latest_business_date: row.business_date_latest || '',
      
      // Calculated fields
      risk_level: riskLevel,
      improvement_needed: improvementNeeded,
      avg_daily_executions: avgDailyExecutions,
      execution_consistency: executionConsistency,
      
      // Additional UI fields
      pass_rate_1m: passRate1m,
      pass_rate_3m: passRate3m,
      pass_rate_12m: passRate12m,
      trend_1m_vs_3m: trend1mVs3m,
      trend_3m_vs_12m: trend3mVs12m,
      earliest_business_date: row.business_date_latest || '', // TODO: Calculate from data
      total_execution_days: 30, // TODO: Calculate from actual date range
      data_volume_trend: 'stable', // TODO: Calculate from historical data
      last_failure_date: row.fail_count_total > 0 ? row.business_date_latest : null
    }
    
    return enhancedRow
  })
}

/**
 * Calculate trend between two periods.
 * Returns: 'improving' if fail rate decreased, 'worsening' if increased, 'stable' if same
 */
function calculateTrend(recentRate: number | undefined, olderRate: number | undefined): string {
  if (recentRate === undefined || olderRate === undefined) {
    return 'unknown'
  }
  
  const threshold = 0.05 // 5% change threshold
  const change = recentRate - olderRate
  
  if (Math.abs(change) < threshold) {
    return 'stable'
  }
  
  return change < 0 ? 'improving' : 'worsening'
}