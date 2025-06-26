import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { FileUpload } from './FileUpload'

describe('FileUpload', () => {
  it('should render upload areas for both CSV and JSON files', () => {
    render(<FileUpload onUpload={vi.fn()} />)
    
    expect(screen.getByText(/data quality csv file/i)).toBeInTheDocument()
    expect(screen.getByText(/rule metadata json file/i)).toBeInTheDocument()
  })

  it('should accept only CSV files', () => {
    render(<FileUpload onUpload={vi.fn()} />)
    
    const input = screen.getByTestId('file-input')
    expect(input).toHaveAttribute('accept', '.csv')
  })

  it('should call onUpload when both CSV and JSON files are uploaded', async () => {
    const onUpload = vi.fn()
    render(<FileUpload onUpload={onUpload} />)
    
    const csvFile = new File(['col1,col2\nval1,val2'], 'test.csv', { type: 'text/csv' })
    const jsonFile = new File(['{}'], 'rules.json', { type: 'application/json' })
    
    const csvDropzone = screen.getByTestId('dropzone')
    const rulesDropzone = screen.getByTestId('rules-dropzone')
    
    // Upload CSV file
    const csvInput = screen.getByTestId('file-input')
    await userEvent.upload(csvInput, csvFile)
    
    // Upload JSON file
    const rulesInput = screen.getByTestId('rules-input')
    await userEvent.upload(rulesInput, jsonFile)
    
    await waitFor(() => {
      expect(onUpload).toHaveBeenCalledWith(csvFile, jsonFile)
    })
  })

  it('should show file names after upload', async () => {
    const onUpload = vi.fn()
    render(<FileUpload onUpload={onUpload} />)
    
    const csvFile = new File(['col1,col2\nval1,val2'], 'test.csv', { type: 'text/csv' })
    const jsonFile = new File(['{}'], 'rules.json', { type: 'application/json' })
    
    const csvInput = screen.getByTestId('file-input')
    const rulesInput = screen.getByTestId('rules-input')
    
    await userEvent.upload(csvInput, csvFile)
    await userEvent.upload(rulesInput, jsonFile)
    
    expect(screen.getByText('test.csv')).toBeInTheDocument()
    expect(screen.getByText('rules.json')).toBeInTheDocument()
  })

  it('should display success message when both files are uploaded', async () => {
    const onUpload = vi.fn()
    render(<FileUpload onUpload={onUpload} />)
    
    const csvFile = new File(['col1,col2\nval1,val2'], 'test.csv', { type: 'text/csv' })
    const jsonFile = new File(['{}'], 'rules.json', { type: 'application/json' })
    
    const csvInput = screen.getByTestId('file-input')
    const rulesInput = screen.getByTestId('rules-input')
    
    await userEvent.upload(csvInput, csvFile)
    await userEvent.upload(rulesInput, jsonFile)
    
    expect(screen.getByText(/both files uploaded successfully/i)).toBeInTheDocument()
  })

  it('should show upload progress', async () => {
    const onUpload = vi.fn()
    render(<FileUpload onUpload={onUpload} isUploading uploadProgress={45} />)
    
    expect(screen.getByText('45%')).toBeInTheDocument()
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '45')
  })
})