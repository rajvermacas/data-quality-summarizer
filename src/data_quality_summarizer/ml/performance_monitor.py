"""
Performance monitoring for ML pipeline operations.

This module provides comprehensive performance tracking including memory usage,
execution time, and resource optimization recommendations.
"""

import psutil
import time
import logging
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    duration_seconds: float
    memory_used_mb: float
    peak_memory_mb: float
    start_time: float
    end_time: float


class PerformanceMonitor:
    """Monitor pipeline performance and resource usage."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict[str, OperationMetrics] = {}
        self.process = psutil.Process()
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """
        Context manager for monitoring operation performance.
        
        Args:
            operation_name: Name of the operation being monitored
            
        Yields:
            None
            
        Example:
            monitor = PerformanceMonitor()
            with monitor.monitor_operation("training"):
                # Perform training operation
                model = train_model(data)
        """
        start_time = time.time()
        start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        
        self.logger.info(f"Starting monitoring for operation: {operation_name}")
        
        try:
            yield
        except Exception as e:
            self.logger.error(f"Operation {operation_name} failed: {e}")
            raise
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            
            metrics = OperationMetrics(
                duration_seconds=end_time - start_time,
                memory_used_mb=end_memory - start_memory,
                peak_memory_mb=end_memory,
                start_time=start_time,
                end_time=end_time
            )
            
            self.metrics[operation_name] = metrics
            
            self.logger.info(
                f"Operation {operation_name} completed: "
                f"duration={metrics.duration_seconds:.2f}s, "
                f"memory_used={metrics.memory_used_mb:.1f}MB, "
                f"peak_memory={metrics.peak_memory_mb:.1f}MB"
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns:
            Dictionary containing performance metrics and recommendations
        """
        if not self.metrics:
            return {
                'individual_operations': {},
                'total_duration': 0.0,
                'peak_memory': 0.0,
                'recommendations': ['No operations monitored yet']
            }
        
        # Convert metrics to serializable format
        individual_ops = {}
        for op_name, metrics in self.metrics.items():
            individual_ops[op_name] = {
                'duration_seconds': metrics.duration_seconds,
                'memory_used_mb': metrics.memory_used_mb,
                'peak_memory_mb': metrics.peak_memory_mb,
                'start_time': metrics.start_time,
                'end_time': metrics.end_time
            }
        
        total_duration = sum(m.duration_seconds for m in self.metrics.values())
        peak_memory = max(m.peak_memory_mb for m in self.metrics.values())
        
        return {
            'individual_operations': individual_ops,
            'total_duration': total_duration,
            'peak_memory': peak_memory,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate performance optimization recommendations.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not self.metrics:
            return ['No performance data available for recommendations']
        
        # Memory usage recommendations
        peak_memory = max(m.peak_memory_mb for m in self.metrics.values())
        if peak_memory > 800:  # Close to 1GB limit
            recommendations.append(
                f"Memory usage is high ({peak_memory:.1f}MB). "
                "Consider reducing chunk size or optimizing data types."
            )
        
        # Processing time recommendations
        slow_operations = [
            name for name, metrics in self.metrics.items()
            if metrics.duration_seconds > 60
        ]
        if slow_operations:
            recommendations.append(
                f"Slow operations detected: {', '.join(slow_operations)}. "
                "Consider performance optimization or parallel processing."
            )
        
        # Memory efficiency recommendations
        high_memory_ops = [
            name for name, metrics in self.metrics.items()
            if metrics.memory_used_mb > 200
        ]
        if high_memory_ops:
            recommendations.append(
                f"High memory usage operations: {', '.join(high_memory_ops)}. "
                "Consider streaming or chunked processing."
            )
        
        # Success message if no issues
        if not recommendations:
            recommendations.append("Performance is within acceptable limits.")
        
        return recommendations
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        self.metrics.clear()
        self.logger.info("Performance metrics reset")
    
    def save_report(self, filepath: str):
        """
        Save performance report to file.
        
        Args:
            filepath: Path to save the report
        """
        import json
        
        report = self.get_performance_report()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Performance report saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save performance report: {e}")
            raise