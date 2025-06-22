#!/usr/bin/env python3
"""
Create comprehensive graphs showing model prediction accuracy and deviations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_prediction_analysis_graphs():
    """Create multiple graphs to analyze prediction accuracy."""
    
    # Check if validation report exists
    if not Path("validation_report.csv").exists():
        print("❌ validation_report.csv not found. Please run simple_validation.py first.")
        return
    
    # Load validation results
    df = pd.read_csv("validation_report.csv")
    print(f"📊 Loaded {len(df)} prediction results")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Scatter Plot: Predicted vs Actual
    plt.subplot(3, 3, 1)
    plt.scatter(df['actual'], df['predicted'], alpha=0.6, s=30)
    plt.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Pass Percentage (%)')
    plt.ylabel('Predicted Pass Percentage (%)')
    plt.title('Predicted vs Actual Values\n(Perfect predictions would lie on red line)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Error Distribution Histogram
    plt.subplot(3, 3, 2)
    plt.hist(df['error'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error (%)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Prediction Errors\nMean Error: {df["error"].mean():.1f}%')
    plt.grid(True, alpha=0.3)
    plt.axvline(df['error'].mean(), color='red', linestyle='--', label=f'Mean: {df["error"].mean():.1f}%')
    plt.legend()
    
    # 3. Error by Rule Code
    plt.subplot(3, 3, 3)
    rule_errors = df.groupby('rule_code')['error'].agg(['mean', 'std']).reset_index()
    plt.bar(rule_errors['rule_code'].astype(str), rule_errors['mean'], 
            yerr=rule_errors['std'], capsize=5, alpha=0.7)
    plt.xlabel('Rule Code')
    plt.ylabel('Mean Absolute Error (%)')
    plt.title('Prediction Error by Rule Code')
    plt.grid(True, alpha=0.3)
    
    # 4. Actual vs Predicted Distribution Comparison
    plt.subplot(3, 3, 4)
    plt.hist(df['actual'], bins=20, alpha=0.5, label='Actual', density=True)
    plt.hist(df['predicted'], bins=20, alpha=0.5, label='Predicted', density=True)
    plt.xlabel('Pass Percentage (%)')
    plt.ylabel('Density')
    plt.title('Distribution Comparison\nActual vs Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Time Series of Errors (by date)
    plt.subplot(3, 3, 5)
    df['date'] = pd.to_datetime(df['date'])
    daily_errors = df.groupby('date')['error'].mean().sort_index()
    plt.plot(daily_errors.index, daily_errors.values, marker='o', markersize=3, linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Mean Absolute Error (%)')
    plt.title('Prediction Error Over Time')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6. Box Plot: Error by Dataset UUID (top 10)
    plt.subplot(3, 3, 6)
    top_datasets = df['dataset_uuid'].value_counts().head(10).index
    df_top = df[df['dataset_uuid'].isin(top_datasets)]
    df_top['dataset_short'] = df_top['dataset_uuid'].str[:8] + '...'
    
    box_data = [df_top[df_top['dataset_uuid'] == uuid]['error'].values 
                for uuid in top_datasets]
    labels = [uuid[:8] + '...' for uuid in top_datasets]
    
    plt.boxplot(box_data, labels=labels)
    plt.xlabel('Dataset UUID (truncated)')
    plt.ylabel('Absolute Error (%)')
    plt.title('Error Distribution by Dataset\n(Top 10 datasets)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 7. Model Performance Metrics Summary
    plt.subplot(3, 3, 7)
    metrics = {
        'MAE': df['error'].mean(),
        'MSE': (df['error'] ** 2).mean(), 
        'RMSE': np.sqrt((df['error'] ** 2).mean()),
        'Max Error': df['error'].max(),
        'Min Error': df['error'].min()
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = plt.bar(metric_names, metric_values, alpha=0.7)
    plt.ylabel('Error Value (%)')
    plt.title('Model Performance Metrics')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 8. Residuals Plot
    plt.subplot(3, 3, 8)
    residuals = df['predicted'] - df['actual']
    plt.scatter(df['actual'], residuals, alpha=0.6, s=30)
    plt.axhline(y=0, color='r', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Actual Pass Percentage (%)')
    plt.ylabel('Residuals (Predicted - Actual)')
    plt.title('Residuals Plot\n(Points should be randomly scattered around 0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Model Quality Assessment
    plt.subplot(3, 3, 9)
    
    # Calculate quality metrics
    mae = df['error'].mean()
    unique_predictions = df['predicted'].nunique()
    total_predictions = len(df)
    prediction_variance = df['predicted'].var()
    actual_variance = df['actual'].var()
    
    # Create text summary
    plt.text(0.1, 0.9, f"🎯 MODEL ASSESSMENT SUMMARY", fontsize=14, fontweight='bold', 
             transform=plt.gca().transAxes)
    
    plt.text(0.1, 0.8, f"Total Predictions: {total_predictions:,}", fontsize=10,
             transform=plt.gca().transAxes)
    
    plt.text(0.1, 0.7, f"Unique Predictions: {unique_predictions:,}", fontsize=10,
             transform=plt.gca().transAxes)
    
    plt.text(0.1, 0.6, f"Mean Absolute Error: {mae:.1f}%", fontsize=10,
             transform=plt.gca().transAxes)
    
    # Quality assessment
    if unique_predictions == 1:
        quality = "❌ CRITICAL: All predictions identical"
        color = 'red'
    elif mae < 5:
        quality = "✅ EXCELLENT: Very accurate"
        color = 'green'
    elif mae < 15:
        quality = "✅ GOOD: Reasonably accurate"  
        color = 'orange'
    elif mae < 30:
        quality = "⚠️ FAIR: Moderate accuracy"
        color = 'yellow'
    else:
        quality = "❌ POOR: Needs improvement"
        color = 'red'
    
    plt.text(0.1, 0.5, f"Quality: {quality}", fontsize=10, color=color, fontweight='bold',
             transform=plt.gca().transAxes)
    
    plt.text(0.1, 0.4, f"Prediction Variance: {prediction_variance:.2f}", fontsize=10,
             transform=plt.gca().transAxes)
    
    plt.text(0.1, 0.3, f"Actual Variance: {actual_variance:.2f}", fontsize=10,
             transform=plt.gca().transAxes)
    
    # Key issues
    plt.text(0.1, 0.2, "🚨 KEY ISSUES:", fontsize=12, fontweight='bold',
             transform=plt.gca().transAxes)
    
    if unique_predictions == 1:
        plt.text(0.1, 0.1, "• Model predicts same value for all inputs", fontsize=9, color='red',
                 transform=plt.gca().transAxes)
        plt.text(0.1, 0.05, "• No learning from features detected", fontsize=9, color='red',
                 transform=plt.gca().transAxes)
    
    plt.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('model_prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a second figure for detailed error analysis
    create_detailed_error_analysis(df)
    
    print(f"\n📊 GRAPHS CREATED:")
    print(f"   📁 Main analysis: model_prediction_analysis.png")
    print(f"   📁 Detailed errors: detailed_error_analysis.png")

def create_detailed_error_analysis(df):
    """Create detailed error analysis graphs."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Error Heatmap by Rule and Date
    ax1 = axes[0, 0]
    
    # Prepare data for heatmap
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.strftime('%Y-%m')
    
    heatmap_data = df.groupby(['rule_code', 'month'])['error'].mean().unstack(fill_value=0)
    
    if not heatmap_data.empty:
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='Reds', ax=ax1)
        ax1.set_title('Average Error by Rule Code and Month')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Rule Code')
    else:
        ax1.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center')
        ax1.set_title('Error Heatmap (Insufficient Data)')
    
    # 2. Cumulative Error Distribution
    ax2 = axes[0, 1]
    sorted_errors = np.sort(df['error'])
    cumulative_pct = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    
    ax2.plot(sorted_errors, cumulative_pct, linewidth=2)
    ax2.set_xlabel('Absolute Error (%)')
    ax2.set_ylabel('Cumulative Percentage of Predictions (%)')
    ax2.set_title('Cumulative Error Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add percentile lines
    percentiles = [50, 75, 90, 95]
    for p in percentiles:
        error_at_percentile = np.percentile(df['error'], p)
        ax2.axvline(error_at_percentile, color='red', linestyle='--', alpha=0.7)
        ax2.text(error_at_percentile, p, f'P{p}: {error_at_percentile:.1f}%', 
                rotation=90, va='bottom')
    
    # 3. Actual vs Predicted Scatter with Error Coloring
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['actual'], df['predicted'], c=df['error'], 
                         cmap='Reds', alpha=0.6, s=30)
    ax3.plot([0, 100], [0, 100], 'black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Actual Pass Percentage (%)')
    ax3.set_ylabel('Predicted Pass Percentage (%)')
    ax3.set_title('Predictions Colored by Error Magnitude')
    plt.colorbar(scatter, ax=ax3, label='Absolute Error (%)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error Statistics Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate detailed statistics
    stats = {
        'Total Predictions': len(df),
        'Mean Error': f"{df['error'].mean():.2f}%",
        'Median Error': f"{df['error'].median():.2f}%", 
        'Std Dev Error': f"{df['error'].std():.2f}%",
        'Min Error': f"{df['error'].min():.2f}%",
        'Max Error': f"{df['error'].max():.2f}%",
        '25th Percentile': f"{df['error'].quantile(0.25):.2f}%",
        '75th Percentile': f"{df['error'].quantile(0.75):.2f}%",
        '90th Percentile': f"{df['error'].quantile(0.90):.2f}%",
        '95th Percentile': f"{df['error'].quantile(0.95):.2f}%",
        'Predictions = 0%': f"{(df['predicted'] == 0).sum():,}",
        'Perfect Predictions': f"{(df['error'] == 0).sum():,}",
        'Errors > 50%': f"{(df['error'] > 50).sum():,}",
        'Errors > 90%': f"{(df['error'] > 90).sum():,}"
    }
    
    # Create table
    table_data = [[key, value] for key, value in stats.items()]
    table = ax4.table(cellText=table_data, 
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(stats) + 1):
        table[(i, 0)].set_facecolor('#E6E6FA')
        table[(i, 1)].set_facecolor('#F0F8FF')
    
    ax4.set_title('Detailed Error Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('detailed_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_prediction_analysis_graphs()