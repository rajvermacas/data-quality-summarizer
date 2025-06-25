import React, { useState } from 'react'
import { FileUploadPage } from './pages/FileUploadPage'
import { ProcessingPage } from './pages/ProcessingPage'
import { ResultsPage } from './pages/ResultsPage'
import { MLPipelinePage } from './pages/MLPipelinePage'
import { ProcessingStatus, ProcessingResult } from './types/common'
import { transformSummaryData } from './utils/dataTransformer'
import { ApiProcessingResult } from './types/api'

type CurrentPage = 'upload' | 'processing' | 'results' | 'ml-pipeline'

function App() {
  const [currentPage, setCurrentPage] = useState<CurrentPage>('upload')
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus>({
    status: 'idle',
    progress: 0,
    message: ''
  })
  const [processingResult, setProcessingResult] = useState<ProcessingResult | null>(null)

  const handleFileUpload = async (csvFile: File, rulesFile: File) => {
    setCurrentPage('processing')
    setProcessingStatus({
      status: 'processing',
      progress: 0,
      message: 'Starting data processing...'
    })

    try {
      // Create form data for file upload
      const formData = new FormData()
      formData.append('csv_file', csvFile)
      formData.append('rules_file', rulesFile)

      // Start processing
      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Processing failed: ${response.statusText}`)
      }

      const apiResult: ApiProcessingResult = await response.json()
      
      // Transform the API response to UI-friendly format
      const transformedResult: ProcessingResult = {
        ...apiResult,
        summary_data: transformSummaryData(apiResult.summary_data)
      }
      
      setProcessingResult(transformedResult)
      setProcessingStatus({
        status: 'completed',
        progress: 100,
        message: 'Processing completed successfully!'
      })
      
      setTimeout(() => {
        setCurrentPage('results')
      }, 1000)

    } catch (error) {
      setProcessingStatus({
        status: 'error',
        progress: 0,
        message: error instanceof Error ? error.message : 'An error occurred during processing'
      })
    }
  }

  const handleStartOver = () => {
    setCurrentPage('upload')
    setProcessingStatus({
      status: 'idle',
      progress: 0,
      message: ''
    })
    setProcessingResult(null)
  }

  const handleViewMLPipeline = () => {
    setCurrentPage('ml-pipeline')
  }

  const handleBackToResults = () => {
    setCurrentPage('results')
  }

  return (
    <div className="container">
      <header style={{ marginBottom: '40px' }}>
        <h1>Data Quality Summarizer</h1>
        <p>Offline data processing system for LLM-optimized summary artifacts and ML pipeline</p>
      </header>

      {currentPage === 'upload' && (
        <FileUploadPage onFileUpload={handleFileUpload} />
      )}

      {currentPage === 'processing' && (
        <ProcessingPage 
          status={processingStatus}
          onStartOver={handleStartOver}
        />
      )}

      {currentPage === 'results' && processingResult && (
        <ResultsPage 
          result={processingResult}
          onStartOver={handleStartOver}
          onViewMLPipeline={handleViewMLPipeline}
        />
      )}

      {currentPage === 'ml-pipeline' && (
        <MLPipelinePage 
          onBackToResults={handleBackToResults}
          csvData={processingResult?.summary_data}
        />
      )}
    </div>
  )
}

export default App