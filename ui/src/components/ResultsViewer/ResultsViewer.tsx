import React from 'react'
import { 
  ArrowDownTrayIcon, 
  ArrowPathIcon,
  DocumentTextIcon 
} from '@heroicons/react/24/outline'
import { ResultsResponse } from '../../lib/api'

interface ResultsViewerProps {
  results: ResultsResponse
  onExport: () => void
  onReset: () => void
}

export const ResultsViewer: React.FC<ResultsViewerProps> = ({
  results,
  onExport,
  onReset,
}) => {
  const { summary, by_rule } = results

  return (
    <div className="space-y-6">
      {/* Actions */}
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Processing Results
        </h2>
        <div className="flex space-x-3">
          <button
            onClick={onExport}
            className="inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors"
          >
            <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
            Export CSV
          </button>
          <button
            onClick={onReset}
            className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors"
          >
            <ArrowPathIcon className="h-4 w-4 mr-2" />
            Process Another File
          </button>
        </div>
      </div>

      {/* Summary Table */}
      <div className="bg-white dark:bg-gray-800 shadow rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Summary Statistics
          </h3>
        </div>
        <div className="px-6 py-4">
          <dl className="grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2">
            <div>
              <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Total Quality Checks
              </dt>
              <dd className="mt-1 text-2xl font-semibold text-gray-900 dark:text-white">
                {summary.total_checks.toLocaleString()}
              </dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Overall Pass Rate
              </dt>
              <dd className="mt-1 text-2xl font-semibold text-primary-600 dark:text-primary-400">
                {(summary.pass_rate * 100).toFixed(2)}%
              </dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Checks Passed
              </dt>
              <dd className="mt-1 text-2xl font-semibold text-green-600 dark:text-green-400">
                {summary.passed.toLocaleString()}
              </dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Checks Failed
              </dt>
              <dd className="mt-1 text-2xl font-semibold text-red-600 dark:text-red-400">
                {summary.failed.toLocaleString()}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      {/* Rule Performance Table */}
      <div className="bg-white dark:bg-gray-800 shadow rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Performance by Rule
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Rule Code
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Pass Rate
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {by_rule.map((rule) => {
                const passRate = rule.pass_rate * 100
                const status = passRate >= 95 ? 'excellent' : passRate >= 80 ? 'good' : 'needs attention'
                
                return (
                  <tr key={rule.rule_code}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                      {rule.rule_code}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      <div className="flex items-center">
                        <div className="flex-1 mr-4">
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${
                                status === 'excellent' ? 'bg-green-500' :
                                status === 'good' ? 'bg-yellow-500' : 'bg-red-500'
                              }`}
                              style={{ width: `${passRate}%` }}
                            />
                          </div>
                        </div>
                        <span className="text-sm font-medium">
                          {passRate.toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        status === 'excellent' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                        status === 'good' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400' :
                        'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                      }`}>
                        {status}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Natural Language Summary Preview */}
      <div className="bg-white dark:bg-gray-800 shadow rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center">
          <DocumentTextIcon className="h-5 w-5 text-gray-400 mr-2" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Natural Language Summary (Preview)
          </h3>
        </div>
        <div className="px-6 py-4">
          <p className="text-sm text-gray-600 dark:text-gray-400 italic">
            The full natural language summary will be included in the exported CSV file. 
            This summary is optimized for LLM consumption and includes detailed breakdowns 
            by dataset, rule, and time period.
          </p>
        </div>
      </div>
    </div>
  )
}