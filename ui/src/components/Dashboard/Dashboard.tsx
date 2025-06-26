import React from 'react'
import { 
  CheckCircleIcon, 
  XCircleIcon, 
  ChartBarIcon, 
  ClockIcon 
} from '@heroicons/react/24/outline'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { ResultsResponse } from '../../lib/api'

interface DashboardProps {
  results: ResultsResponse
}

export const Dashboard: React.FC<DashboardProps> = ({ results }) => {
  const { summary, by_rule, trends } = results

  // Prepare trend data for chart
  const trendData = [
    { period: '1 Month', pass_rate: trends['1_month'].pass_rate * 100 },
    { period: '3 Months', pass_rate: trends['3_month'].pass_rate * 100 },
    { period: '12 Months', pass_rate: trends['12_month'].pass_rate * 100 },
  ]

  // Prepare rule data for chart
  const ruleData = by_rule.map(rule => ({
    rule: rule.rule_code,
    pass_rate: rule.pass_rate * 100,
  }))

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Total Checks
              </p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {summary.total_checks.toLocaleString()}
              </p>
            </div>
            <ChartBarIcon className="h-10 w-10 text-gray-400" />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Passed
              </p>
              <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                {summary.passed.toLocaleString()}
              </p>
            </div>
            <CheckCircleIcon className="h-10 w-10 text-green-500" />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Failed
              </p>
              <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                {summary.failed.toLocaleString()}
              </p>
            </div>
            <XCircleIcon className="h-10 w-10 text-red-500" />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                Pass Rate
              </p>
              <p className="text-2xl font-bold text-primary-600 dark:text-primary-400">
                {(summary.pass_rate * 100).toFixed(1)}%
              </p>
            </div>
            <div className="relative">
              <svg className="h-10 w-10 transform -rotate-90">
                <circle
                  cx="20"
                  cy="20"
                  r="16"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                  className="text-gray-300 dark:text-gray-600"
                />
                <circle
                  cx="20"
                  cy="20"
                  r="16"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                  strokeDasharray={`${summary.pass_rate * 100} ${100 - summary.pass_rate * 100}`}
                  className="text-primary-500"
                />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trend Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Pass Rate Trends
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="period" 
                stroke="#9CA3AF"
                style={{ fontSize: '12px' }}
              />
              <YAxis 
                stroke="#9CA3AF"
                style={{ fontSize: '12px' }}
                domain={[0, 100]}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: 'none',
                  borderRadius: '8px',
                  color: '#F9FAFB'
                }}
                formatter={(value: any) => `${value.toFixed(1)}%`}
              />
              <Line 
                type="monotone" 
                dataKey="pass_rate" 
                stroke="#3B82F6" 
                strokeWidth={3}
                dot={{ fill: '#3B82F6', r: 6 }}
                activeDot={{ r: 8 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Rule Performance Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Pass Rate by Rule
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={ruleData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="rule" 
                stroke="#9CA3AF"
                style={{ fontSize: '12px' }}
              />
              <YAxis 
                stroke="#9CA3AF"
                style={{ fontSize: '12px' }}
                domain={[0, 100]}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: 'none',
                  borderRadius: '8px',
                  color: '#F9FAFB'
                }}
                formatter={(value: any) => `${value.toFixed(1)}%`}
              />
              <Bar 
                dataKey="pass_rate" 
                fill="#8B5CF6"
                radius={[8, 8, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}