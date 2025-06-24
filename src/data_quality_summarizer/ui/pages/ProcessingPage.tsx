import React, { useEffect, useState } from 'react'
import { ProgressBar } from '../components/ProgressBar'
import { ProcessingStatus } from '../types/common'
import { RefreshCw, Database, FileText, BarChart3, Cpu } from 'lucide-react'

interface ProcessingPageProps {
  status: ProcessingStatus
  onStartOver: () => void
}

export const ProcessingPage: React.FC<ProcessingPageProps> = ({ status, onStartOver }) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [elapsedTime, setElapsedTime] = useState(0)

  const processingSteps = [
    { name: 'File Validation', icon: FileText, description: 'Validating file formats and structure' },
    { name: 'Data Ingestion', icon: Database, description: 'Reading CSV data in chunks' },
    { name: 'Rule Processing', icon: Cpu, description: 'Processing data quality rules' },
    { name: 'Aggregation', icon: BarChart3, description: 'Computing summary statistics' },
    { name: 'Report Generation', icon: FileText, description: 'Creating summary artifacts' }
  ]

  useEffect(() => {
    if (status.status === 'processing') {
      const timer = setInterval(() => {
        setElapsedTime(prev => prev + 1)
      }, 1000)

      return () => clearInterval(timer)
    }
  }, [status.status])

  useEffect(() => {
    // Simulate step progression based on progress
    const stepIndex = Math.min(
      Math.floor((status.progress / 100) * processingSteps.length),
      processingSteps.length - 1
    )
    setCurrentStep(stepIndex)
  }, [status.progress, processingSteps.length])

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="processing-page">
      <div className="processing-header">
        <h2>Processing Data</h2>
        <p>Your files are being processed using the streaming aggregation pipeline.</p>
      </div>

      <div className="processing-content">
        <div className="progress-section">
          <ProgressBar 
            progress={status.progress}
            status={status.status}
            message={status.message}
          />
          
          {status.status === 'processing' && (
            <div className="processing-stats">
              <div className="stat">
                <span className="stat-label">Elapsed Time:</span>
                <span className="stat-value">{formatTime(elapsedTime)}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Estimated Total:</span>
                <span className="stat-value">~2 minutes</span>
              </div>
            </div>
          )}
        </div>

        <div className="steps-section">
          <h3>Processing Steps</h3>
          <div className="steps-list">
            {processingSteps.map((step, index) => {
              const StepIcon = step.icon
              const isActive = index === currentStep && status.status === 'processing'
              const isCompleted = index < currentStep || status.status === 'completed'
              const isError = status.status === 'error' && index === currentStep

              return (
                <div 
                  key={index}
                  className={`step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''} ${isError ? 'error' : ''}`}
                >
                  <div className="step-icon">
                    <StepIcon size={20} />
                  </div>
                  <div className="step-content">
                    <h4>{step.name}</h4>
                    <p>{step.description}</p>
                  </div>
                  <div className="step-status">
                    {isCompleted && <span className="status-completed">✓</span>}
                    {isActive && <span className="status-active">●</span>}
                    {isError && <span className="status-error">✗</span>}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {status.status === 'error' && (
          <div className="error-section">
            <div className="error-message">
              <h3>Processing Failed</h3>
              <p>{status.message}</p>
            </div>
            <button className="button" onClick={onStartOver}>
              <RefreshCw size={16} />
              Start Over
            </button>
          </div>
        )}

        <div className="processing-info">
          <h3>System Information</h3>
          <div className="info-grid">
            <div className="info-item">
              <strong>Memory Optimization:</strong>
              <span>Streaming processing keeps RAM usage under 1GB</span>
            </div>
            <div className="info-item">
              <strong>Chunk Size:</strong>
              <span>20,000 rows per chunk for efficient processing</span>
            </div>
            <div className="info-item">
              <strong>Output Format:</strong>
              <span>CSV summary + Natural language artifacts</span>
            </div>
            <div className="info-item">
              <strong>Time Windows:</strong>
              <span>1-month, 3-month, and 12-month aggregations</span>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .processing-page {
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
        }

        .processing-header {
          text-align: center;
          margin-bottom: 40px;
        }

        .processing-header h2 {
          color: #333;
          margin-bottom: 10px;
        }

        .processing-header p {
          color: #666;
          font-size: 16px;
        }

        .processing-content {
          display: flex;
          flex-direction: column;
          gap: 30px;
        }

        .progress-section {
          background: white;
          padding: 30px;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border: 1px solid #e9ecef;
        }

        .processing-stats {
          display: flex;
          justify-content: space-around;
          margin-top: 20px;
          padding-top: 20px;
          border-top: 1px solid #e9ecef;
        }

        .stat {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 5px;
        }

        .stat-label {
          font-size: 14px;
          color: #666;
        }

        .stat-value {
          font-size: 18px;
          font-weight: bold;
          color: #333;
        }

        .steps-section {
          background: white;
          padding: 30px;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border: 1px solid #e9ecef;
        }

        .steps-section h3 {
          margin: 0 0 20px 0;
          color: #333;
        }

        .steps-list {
          display: flex;
          flex-direction: column;
          gap: 15px;
        }

        .step {
          display: flex;
          align-items: center;
          gap: 15px;
          padding: 15px;
          border-radius: 6px;
          background: #f8f9fa;
          transition: all 0.3s ease;
        }

        .step.active {
          background: #e3f2fd;
          border-left: 4px solid #007bff;
        }

        .step.completed {
          background: #e8f5e8;
          border-left: 4px solid #28a745;
        }

        .step.error {
          background: #ffeaea;
          border-left: 4px solid #dc3545;
        }

        .step-icon {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 40px;
          height: 40px;
          background: white;
          border-radius: 50%;
          color: #666;
        }

        .step.active .step-icon {
          color: #007bff;
          background: #e3f2fd;
        }

        .step.completed .step-icon {
          color: #28a745;
          background: #e8f5e8;
        }

        .step.error .step-icon {
          color: #dc3545;
          background: #ffeaea;
        }

        .step-content {
          flex: 1;
        }

        .step-content h4 {
          margin: 0 0 5px 0;
          color: #333;
        }

        .step-content p {
          margin: 0;
          color: #666;
          font-size: 14px;
        }

        .step-status {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 30px;
          height: 30px;
          border-radius: 50%;
          font-weight: bold;
        }

        .status-completed {
          color: #28a745;
          background: #e8f5e8;
        }

        .status-active {
          color: #007bff;
          background: #e3f2fd;
        }

        .status-error {
          color: #dc3545;
          background: #ffeaea;
        }

        .error-section {
          background: white;
          padding: 30px;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          border: 1px solid #dc3545;
          text-align: center;
        }

        .error-message {
          margin-bottom: 20px;
        }

        .error-message h3 {
          color: #dc3545;
          margin-bottom: 10px;
        }

        .processing-info {
          background: #f8f9fa;
          padding: 30px;
          border-radius: 8px;
          border: 1px solid #e9ecef;
        }

        .processing-info h3 {
          margin: 0 0 20px 0;
          color: #333;
        }

        .info-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 15px;
        }

        .info-item {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }

        .info-item strong {
          color: #495057;
          font-size: 14px;
        }

        .info-item span {
          color: #666;
          font-size: 13px;
        }

        .button {
          display: flex;
          align-items: center;
          gap: 8px;
          justify-content: center;
        }
      `}</style>
    </div>
  )
}