import React, { useState, useEffect } from 'react'
import { Toaster } from 'react-hot-toast'
import { SunIcon, MoonIcon } from '@heroicons/react/24/outline'
import { FileUpload } from './components/FileUpload'
import { ProcessingStatus } from './components/ProcessingStatus'
import { Dashboard } from './components/Dashboard'
import { ResultsViewer } from './components/ResultsViewer'
import { api } from './lib/api'
import './index.css'

export interface ProcessingState {
  processingId: string | null
  status: 'idle' | 'uploading' | 'processing' | 'completed' | 'failed'
  progress: number
  message: string
  results: any | null
}

function App() {
  const [darkMode, setDarkMode] = useState(false)
  const [processing, setProcessing] = useState<ProcessingState>({
    processingId: null,
    status: 'idle',
    progress: 0,
    message: '',
    results: null,
  })

  useEffect(() => {
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme')
    if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      setDarkMode(true)
      document.documentElement.classList.add('dark')
    }
  }, [])

  const toggleDarkMode = () => {
    const newMode = !darkMode
    setDarkMode(newMode)
    if (newMode) {
      document.documentElement.classList.add('dark')
      localStorage.setItem('theme', 'dark')
    } else {
      document.documentElement.classList.remove('dark')
      localStorage.setItem('theme', 'light')
    }
  }

  const handleFileUpload = async (csvFile: File, rulesFile: File) => {
    setProcessing({
      processingId: null,
      status: 'uploading',
      progress: 0,
      message: 'Uploading files...',
      results: null,
    })

    try {
      const response = await api.uploadFiles(csvFile, rulesFile)
      const { processing_id } = response

      setProcessing({
        processingId: processing_id,
        status: 'processing',
        progress: 10,
        message: 'Processing started...',
        results: null,
      })

      // Start polling for status
      pollProcessingStatus(processing_id)
    } catch (error: any) {
      setProcessing({
        processingId: null,
        status: 'failed',
        progress: 0,
        message: error.message || 'Upload failed',
        results: null,
      })
    }
  }

  const pollProcessingStatus = async (processingId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const status = await api.getProcessingStatus(processingId)
        
        setProcessing(prev => ({
          ...prev,
          progress: status.progress,
          message: status.message,
        }))

        if (status.status === 'completed') {
          clearInterval(pollInterval)
          const results = await api.getResults(processingId)
          setProcessing({
            processingId,
            status: 'completed',
            progress: 100,
            message: 'Processing completed successfully',
            results,
          })
        } else if (status.status === 'failed') {
          clearInterval(pollInterval)
          setProcessing({
            processingId,
            status: 'failed',
            progress: 0,
            message: status.error || 'Processing failed',
            results: null,
          })
        }
      } catch (error) {
        clearInterval(pollInterval)
        setProcessing(prev => ({
          ...prev,
          status: 'failed',
          message: 'Failed to get status',
        }))
      }
    }, 2000) // Poll every 2 seconds
  }

  const handleExport = async () => {
    if (!processing.processingId) return
    
    try {
      await api.exportResults(processing.processingId, 'csv')
    } catch (error) {
      console.error('Export failed:', error)
    }
  }

  const resetProcessing = () => {
    setProcessing({
      processingId: null,
      status: 'idle',
      progress: 0,
      message: '',
      results: null,
    })
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      <Toaster position="top-right" />
      
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                Data Quality Summarizer
              </h1>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                v0.1.0
              </span>
            </div>
            
            <button
              onClick={toggleDarkMode}
              data-testid="dark-mode-toggle"
              className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              aria-label="Toggle dark mode"
            >
              {darkMode ? (
                <SunIcon className="h-5 w-5 text-yellow-500" />
              ) : (
                <MoonIcon className="h-5 w-5 text-gray-700" />
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {processing.status === 'idle' && (
          <div className="space-y-8">
            <section>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                Upload CSV File
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                Upload your data quality check results CSV file along with the rule metadata JSON file to begin processing.
              </p>
              <FileUpload onUpload={handleFileUpload} />
            </section>
          </div>
        )}

        {(processing.status === 'uploading' || processing.status === 'processing') && (
          <ProcessingStatus
            status={processing.status}
            progress={processing.progress}
            message={processing.message}
          />
        )}

        {processing.status === 'completed' && processing.results && (
          <div className="space-y-8">
            <Dashboard results={processing.results} />
            <ResultsViewer
              results={processing.results}
              onExport={handleExport}
              onReset={resetProcessing}
            />
          </div>
        )}

        {processing.status === 'failed' && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-red-800 dark:text-red-200 mb-2">
              Processing Failed
            </h3>
            <p className="text-red-700 dark:text-red-300 mb-4">{processing.message}</p>
            <button
              onClick={resetProcessing}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

export default App