import React from 'react'
import { ClockIcon } from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface ProcessingStatusProps {
  status: 'uploading' | 'processing' | 'completed' | 'failed'
  progress: number
  message: string
}

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({
  status,
  progress,
  message,
}) => {
  const getStatusColor = () => {
    switch (status) {
      case 'uploading':
        return 'blue'
      case 'processing':
        return 'indigo'
      case 'completed':
        return 'green'
      case 'failed':
        return 'red'
      default:
        return 'gray'
    }
  }

  const color = getStatusColor()

  return (
    <div className="max-w-2xl mx-auto">
      <div className={`bg-${color}-50 dark:bg-${color}-900/20 border border-${color}-200 dark:border-${color}-800 rounded-lg p-8`}>
        <div className="flex items-center justify-center mb-6">
          <div className="relative">
            <ClockIcon className={`h-16 w-16 text-${color}-500 animate-pulse-subtle`} />
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-xs font-bold text-${color}-700 dark:text-${color}-300">
                {progress}%
              </span>
            </div>
          </div>
        </div>

        <h2 className={`text-xl font-semibold text-${color}-900 dark:text-${color}-100 text-center mb-4`}>
          {status === 'uploading' && 'Uploading Files'}
          {status === 'processing' && 'Processing Data'}
          {status === 'completed' && 'Processing Complete'}
          {status === 'failed' && 'Processing Failed'}
        </h2>

        <p className={`text-${color}-700 dark:text-${color}-300 text-center mb-6`}>
          {message}
        </p>

        <div className="w-full">
          <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
            <span>Progress</span>
            <span>{progress}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
            <div
              className={clsx(
                `h-full rounded-full transition-all duration-500 ease-out`,
                {
                  'bg-blue-500': color === 'blue',
                  'bg-indigo-500': color === 'indigo',
                  'bg-green-500': color === 'green',
                  'bg-red-500': color === 'red',
                  'bg-gray-500': color === 'gray',
                }
              )}
              style={{ width: `${progress}%` }}
              role="progressbar"
              aria-valuenow={progress}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          </div>
        </div>

        {status === 'processing' && (
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              This may take a few minutes depending on file size...
            </p>
          </div>
        )}
      </div>
    </div>
  )
}