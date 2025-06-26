import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from './App'

describe('App', () => {
  it('should render the main application', () => {
    render(<App />)
    
    expect(screen.getByText(/data quality summarizer/i)).toBeInTheDocument()
  })

  it('should have a file upload section', () => {
    render(<App />)
    
    expect(screen.getByText(/upload csv file/i)).toBeInTheDocument()
  })

  it('should have a dark mode toggle', () => {
    render(<App />)
    
    const darkModeButton = screen.getByTestId('dark-mode-toggle')
    expect(darkModeButton).toBeInTheDocument()
  })
})