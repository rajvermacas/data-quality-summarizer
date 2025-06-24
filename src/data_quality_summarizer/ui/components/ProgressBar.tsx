import React from 'react'
import { CheckCircle, AlertCircle, Loader } from 'lucide-react'

interface ProgressBarProps {
  progress: number
  status: 'idle' | 'processing' | 'completed' | 'error'
  message: string
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ progress, status, message }) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'processing':
        return <Loader className="spinning" size={24} />
      case 'completed':
        return <CheckCircle size={24} color="#28a745" />
      case 'error':
        return <AlertCircle size={24} color="#dc3545" />
      default:
        return null
    }
  }

  const getStatusColor = () => {
    switch (status) {
      case 'processing':
        return '#007bff'
      case 'completed':
        return '#28a745'
      case 'error':
        return '#dc3545'
      default:
        return '#6c757d'
    }
  }

  return (
    <div className="progress-container">
      <div className="progress-header">
        {getStatusIcon()}
        <span className="progress-message">{message}</span>
      </div>
      
      <div className="progress-bar">
        <div 
          className="progress-fill"
          style={{ 
            width: `${progress}%`,
            backgroundColor: getStatusColor()
          }}
        />
      </div>
      
      <div className="progress-text">
        {progress}% Complete
      </div>

      <style jsx>{`
        .progress-container {
          width: 100%;
          margin: 20px 0;
        }

        .progress-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 10px;
        }

        .progress-message {
          font-weight: 500;
          color: #333;
        }

        .progress-bar {
          width: 100%;
          height: 20px;
          background-color: #e9ecef;
          border-radius: 10px;
          overflow: hidden;
          margin-bottom: 8px;
        }

        .progress-fill {
          height: 100%;
          transition: width 0.3s ease;
          border-radius: 10px;
        }

        .progress-text {
          text-align: center;
          font-size: 14px;
          color: #6c757d;
        }

        :global(.spinning) {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}