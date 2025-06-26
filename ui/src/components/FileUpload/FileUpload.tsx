import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { CloudArrowUpIcon, DocumentIcon, XCircleIcon } from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface FileUploadProps {
  onUpload: (csvFile: File, rulesFile: File) => void
  isUploading?: boolean
  uploadProgress?: number
  maxSize?: number
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onUpload,
  isUploading = false,
  uploadProgress = 0,
  maxSize = 100 * 1024 * 1024, // 100MB default
}) => {
  const [error, setError] = useState<string | null>(null)
  const [csvFile, setCsvFile] = useState<File | null>(null)
  const [rulesFile, setRulesFile] = useState<File | null>(null)

  const onDropCsv = useCallback((acceptedFiles: File[], fileRejections: any[]) => {
    setError(null)
    
    if (fileRejections.length > 0) {
      setError('Only CSV files are allowed')
      return
    }

    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0]
      
      if (file.size > maxSize) {
        setError(`File size must be less than ${maxSize / 1024 / 1024}MB`)
        return
      }

      setCsvFile(file)
      if (rulesFile) {
        onUpload(file, rulesFile)
      }
    }
  }, [onUpload, maxSize, rulesFile])

  const onDropRules = useCallback((acceptedFiles: File[]) => {
    setError(null)
    
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0]
      setRulesFile(file)
      if (csvFile) {
        onUpload(csvFile, file)
      }
    }
  }, [onUpload, csvFile])

  const csvDropzone = useDropzone({
    onDrop: onDropCsv,
    accept: {
      'text/csv': ['.csv'],
    },
    multiple: false,
    disabled: isUploading,
  })

  const rulesDropzone = useDropzone({
    onDrop: onDropRules,
    accept: {
      'application/json': ['.json'],
    },
    multiple: false,
    disabled: isUploading,
  })

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} bytes`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className="w-full max-w-4xl mx-auto space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* CSV File Upload */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Data Quality CSV File
          </h3>
          <div
            {...csvDropzone.getRootProps()}
            data-testid="dropzone"
            className={clsx(
              'relative border-2 border-dashed rounded-lg p-6 transition-all duration-200',
              {
                'border-gray-300 hover:border-gray-400': !csvDropzone.isDragActive && !error,
                'border-primary-500 bg-primary-50': csvDropzone.isDragActive,
                'border-green-500 bg-green-50': csvFile,
                'border-red-300 bg-red-50': error,
                'opacity-50 cursor-not-allowed': isUploading,
              }
            )}
          >
            <input {...csvDropzone.getInputProps()} data-testid="file-input" />
            
            <div className="text-center">
              <CloudArrowUpIcon className={clsx(
                'mx-auto h-10 w-10 mb-3',
                {
                  'text-gray-400': !csvDropzone.isDragActive && !csvFile,
                  'text-primary-500': csvDropzone.isDragActive,
                  'text-green-500': csvFile,
                  'text-red-400': error,
                }
              )} />
              
              {csvFile ? (
                <div>
                  <p className="text-sm font-medium text-green-700">{csvFile.name}</p>
                  <p className="text-xs text-gray-500">{formatFileSize(csvFile.size)}</p>
                </div>
              ) : (
                <>
                  <p className="text-sm text-gray-700 mb-1">
                    Drag and drop CSV file
                  </p>
                  <p className="text-xs text-gray-500">
                    or click to select
                  </p>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Rules File Upload */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Rule Metadata JSON File
          </h3>
          <div
            {...rulesDropzone.getRootProps()}
            data-testid="rules-dropzone"
            className={clsx(
              'relative border-2 border-dashed rounded-lg p-6 transition-all duration-200',
              {
                'border-gray-300 hover:border-gray-400': !rulesDropzone.isDragActive && !rulesFile,
                'border-primary-500 bg-primary-50': rulesDropzone.isDragActive,
                'border-green-500 bg-green-50': rulesFile,
                'opacity-50 cursor-not-allowed': isUploading,
              }
            )}
          >
            <input {...rulesDropzone.getInputProps()} data-testid="rules-input" />
            
            <div className="text-center">
              <DocumentIcon className={clsx(
                'mx-auto h-10 w-10 mb-3',
                {
                  'text-gray-400': !rulesDropzone.isDragActive && !rulesFile,
                  'text-primary-500': rulesDropzone.isDragActive,
                  'text-green-500': rulesFile,
                }
              )} />
              
              {rulesFile ? (
                <div>
                  <p className="text-sm font-medium text-green-700">{rulesFile.name}</p>
                  <p className="text-xs text-gray-500">{formatFileSize(rulesFile.size)}</p>
                </div>
              ) : (
                <>
                  <p className="text-sm text-gray-700 mb-1">
                    Drag and drop JSON file
                  </p>
                  <p className="text-xs text-gray-500">
                    or click to select
                  </p>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Upload Progress */}
      {isUploading && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="space-y-3">
            <p className="text-sm text-blue-800 font-medium">Uploading files...</p>
            <div className="w-full bg-blue-200 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
                role="progressbar"
                aria-valuenow={uploadProgress}
                aria-valuemin={0}
                aria-valuemax={100}
              />
            </div>
            <p className="text-sm text-blue-700">{uploadProgress}%</p>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 flex items-center">
          <XCircleIcon className="h-5 w-5 text-red-400 mr-2" />
          <p className="text-sm text-red-800">{error}</p>
        </div>
      )}

      {/* Ready to Process */}
      {csvFile && rulesFile && !error && !isUploading && (
        <div className="bg-green-50 border border-green-200 rounded-md p-4">
          <p className="text-sm text-green-800 font-medium text-center">
            Both files uploaded successfully. Processing will begin automatically.
          </p>
        </div>
      )}
    </div>
  )
}