"""
Progress tracking component for the web UI.

Provides real-time progress updates during data processing operations.
"""

from typing import Dict, Any, Optional


class ProgressTracker:
    """
    Progress tracking component for UI operations.
    
    Manages progress state and provides UI elements for displaying
    processing progress to users.
    """
    
    def __init__(self):
        """Initialize progress tracker."""
        self._progress = {
            'percentage': 0,
            'message': 'Ready',
            'current_step': 0,
            'total_steps': 1
        }
        self._progress_bar = None
        self._status_text = None
    
    def update_progress(self, percentage: int, message: str) -> None:
        """
        Update progress state.
        
        Args:
            percentage: Progress percentage (0-100)
            message: Status message to display
        """
        self._progress['percentage'] = max(0, min(100, percentage))
        self._progress['message'] = message
        
        # Update UI elements if they exist
        if self._progress_bar is not None:
            try:
                import streamlit as st
                self._progress_bar.progress(self._progress['percentage'] / 100)
                self._status_text.text(message)
            except ImportError:
                # For testing without Streamlit
                pass
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress state."""
        return self._progress.copy()
    
    def create_ui_elements(self) -> Dict[str, Any]:
        """
        Create Streamlit UI elements for progress display.
        
        Returns:
            Dictionary containing created UI elements
        """
        try:
            import streamlit as st
            
            # Create progress bar
            self._progress_bar = st.progress(0)
            
            # Create status text placeholder
            self._status_text = st.empty()
            
            return {
                'progress_bar': self._progress_bar,
                'status_text': self._status_text
            }
            
        except ImportError:
            # For testing without Streamlit
            return {
                'progress_bar': None,
                'status_text': None
            }
    
    def reset(self) -> None:
        """Reset progress to initial state."""
        self.update_progress(0, "Ready")


def create_progress_tracker() -> ProgressTracker:
    """
    Factory function to create a progress tracker.
    
    Returns:
        Configured ProgressTracker instance
    """
    return ProgressTracker()