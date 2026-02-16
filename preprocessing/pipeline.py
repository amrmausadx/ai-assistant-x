"""
Main preprocessing pipeline
"""
import os
import re
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

#from seaborn import load_dataset
from .core import (
    load_chosen_dataset,
    load_gutenberg, 
    load_bookcorpus, 
    load_poetry, 
    create_preprocessing_report
)
from utils.status import update_preprocessing_status


def count_tokens(text):
    """Simple token counter using whitespace splitting"""
    return len(text.split())


def evaluate_dataset(texts, output_dir="static", min_tokens=30):
    """
    Evaluate dataset and generate statistics
    
    Args:
        texts: List of text samples
        output_dir: Directory to save evaluation outputs
        min_tokens: Minimum token threshold for filtering
    
    Returns:
        dict: Evaluation metrics
        list: Filtered texts (with short samples removed)
    """
    # Count tokens for each document
    token_counts = [count_tokens(text) for text in texts]
    
    # Calculate statistics
    total_documents = len(texts)
    total_tokens = sum(token_counts)
    avg_length = total_tokens / total_documents if total_documents > 0 else 0
    median_length = np.median(token_counts) if token_counts else 0
    std_length = np.std(token_counts) if token_counts else 0
    min_length = min(token_counts) if token_counts else 0
    max_length = max(token_counts) if token_counts else 0
    
    # Count short documents
    short_docs = sum(1 for count in token_counts if count < min_tokens)
    
    # Filter out short documents
    filtered_texts = [text for text, count in zip(texts, token_counts) if count >= min_tokens]
    filtered_token_counts = [count for count in token_counts if count >= min_tokens]
    
    # Calculate filtered statistics
    filtered_total_documents = len(filtered_texts)
    filtered_total_tokens = sum(filtered_token_counts)
    filtered_avg_length = filtered_total_tokens / filtered_total_documents if filtered_total_documents > 0 else 0
    
    # Create evaluation metrics dictionary
    metrics = {
        'total_documents_before_filter': total_documents,
        'total_tokens_before_filter': total_tokens,
        'avg_document_length_before_filter': avg_length,
        'median_document_length_before_filter': median_length,
        'std_document_length_before_filter': std_length,
        'min_document_length_before_filter': min_length,
        'max_document_length_before_filter': max_length,
        'documents_removed': short_docs,
        'total_documents_after_filter': filtered_total_documents,
        'total_tokens_after_filter': filtered_total_tokens,
        'avg_document_length_after_filter': filtered_avg_length,
        'removal_percentage': (short_docs / total_documents * 100) if total_documents > 0 else 0
    }
    
    # Create length distribution plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "length_distribution.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Document Length Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Histogram of all documents
    axes[0, 0].hist(token_counts, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(avg_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_length:.1f}')
    axes[0, 0].axvline(median_length, color='green', linestyle='--', linewidth=2, label=f'Median: {median_length:.1f}')
    axes[0, 0].axvline(min_tokens, color='orange', linestyle='--', linewidth=2, label=f'Min threshold: {min_tokens}')
    axes[0, 0].set_xlabel('Document Length (tokens)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Before Filtering', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of filtered documents
    if filtered_token_counts:
        axes[0, 1].hist(filtered_token_counts, bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(filtered_avg_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {filtered_avg_length:.1f}')
        axes[0, 1].set_xlabel('Document Length (tokens)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title(f'After Filtering (>={min_tokens} tokens)', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Box plot comparison
    box_data = [token_counts, filtered_token_counts] if filtered_token_counts else [token_counts]
    box_labels = ['Before Filter', 'After Filter'] if filtered_token_counts else ['Before Filter']
    bp = axes[1, 0].boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['steelblue', 'darkgreen']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_ylabel('Document Length (tokens)', fontsize=11)
    axes[1, 0].set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Statistics summary table
    axes[1, 1].axis('off')
    summary_data = [
        ['Metric', 'Before Filter', 'After Filter'],
        ['Total Documents', f'{total_documents:,}', f'{filtered_total_documents:,}'],
        ['Total Tokens', f'{total_tokens:,}', f'{filtered_total_tokens:,}'],
        ['Avg Length', f'{avg_length:.1f}', f'{filtered_avg_length:.1f}'],
        ['Median Length', f'{median_length:.1f}', f'{np.median(filtered_token_counts):.1f}' if filtered_token_counts else 'N/A'],
        ['Min Length', f'{min_length}', f'{min(filtered_token_counts)}' if filtered_token_counts else 'N/A'],
        ['Max Length', f'{max_length}', f'{max(filtered_token_counts)}' if filtered_token_counts else 'N/A'],
        ['Documents Removed', '-', f'{short_docs:,} ({metrics["removal_percentage"]:.2f}%)']
    ]
    
    table = axes[1, 1].table(cellText=summary_data, cellLoc='left', loc='center',
                             colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    axes[1, 1].set_title('Statistics Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create evaluation report text file
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("DATASET EVALUATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Minimum token threshold: {min_tokens}\n")
        f.write("="*70 + "\n\n")
        
        f.write("BEFORE FILTERING:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Total Documents:        {total_documents:,}\n")
        f.write(f"  Total Tokens:           {total_tokens:,}\n")
        f.write(f"  Average Length:         {avg_length:.2f} tokens\n")
        f.write(f"  Median Length:          {median_length:.2f} tokens\n")
        f.write(f"  Std Deviation:          {std_length:.2f} tokens\n")
        f.write(f"  Min Length:             {min_length} tokens\n")
        f.write(f"  Max Length:             {max_length} tokens\n")
        f.write("\n")
        
        f.write("FILTERING RESULTS:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Documents Removed:      {short_docs:,} ({metrics['removal_percentage']:.2f}%)\n")
        f.write(f"  Removal Criterion:      Documents with < {min_tokens} tokens\n")
        f.write("\n")
        
        f.write("AFTER FILTERING:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Total Documents:        {filtered_total_documents:,}\n")
        f.write(f"  Total Tokens:           {filtered_total_tokens:,}\n")
        f.write(f"  Average Length:         {filtered_avg_length:.2f} tokens\n")
        if filtered_token_counts:
            f.write(f"  Median Length:          {np.median(filtered_token_counts):.2f} tokens\n")
            f.write(f"  Std Deviation:          {np.std(filtered_token_counts):.2f} tokens\n")
            f.write(f"  Min Length:             {min(filtered_token_counts)} tokens\n")
            f.write(f"  Max Length:             {max(filtered_token_counts)} tokens\n")
        f.write("\n")
        
        f.write("LENGTH DISTRIBUTION (Before Filtering):\n")
        f.write("-"*70 + "\n")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(token_counts, p)
            f.write(f"  {p}th percentile:        {value:.2f} tokens\n")
        
        f.write("\n" + "="*70 + "\n")
    
    return metrics, filtered_texts, plot_path, report_path


def run_preprocessing_pipeline(config, opt=None):
    """Execute the complete preprocessing pipeline"""
    update_preprocessing_status(
        running=True,
        progress=0,
        message='Starting preprocessing...',
        error=None,
        start_time=datetime.now(),
        experiment_name=config['experiment_name'],
    )
    
    try:
        mlflow.set_experiment(config["experiment_name"])
        mlflow.autolog()
        with mlflow.start_run(run_name="data_preprocessing") as run:
            update_preprocessing_status(run_id=run.info.run_id)
            mlflow.log_param("pipeline", "text_cleaning_and_tokenization")
            
            # Load datasets with progress updates
            update_preprocessing_status(progress=30, message='Collecting selected datasets '+str(config['selected_datasets'])+'...')
            
            # Load additional datasets if specified
            selected_datasets = config.get("selected_datasets", [])
            # Add this check:
            if not selected_datasets:
                raise ValueError("No datasets selected. Please select at least one dataset in the UI.")
            
            count_other_datasets = 0
            datasets_texts = {}
            for dataset_name in selected_datasets:
                texts, count = load_chosen_dataset(dataset_name=dataset_name, config=config)
                if texts:  # Only add if not empty
                    datasets_texts[dataset_name] = texts
                    count_other_datasets += count
                    mlflow.log_metric(f"{dataset_name}_count", count)
            
            if not datasets_texts or all(len(texts) == 0 for texts in datasets_texts.values()):
                raise ValueError("No datasets were successfully loaded")
            
            update_preprocessing_status(progress=60, message='Combining data...')
            
            # Add additional datasets to DataFrame
            # Clean texts
            all_texts = []

            for idx, (dataset_name, texts) in enumerate(datasets_texts.items(), 1):
                all_texts.extend(texts)
                progress = 60 + (idx / len(datasets_texts)) * 10  # Progress from 60-70%
                update_preprocessing_status(progress=int(progress), 
                                            message=f'Combining texts ({idx}/{len(datasets_texts)} datasets)...')
            
            update_preprocessing_status(progress=70, message='Evaluating dataset quality...')
            
            # Evaluate dataset
            min_token_threshold = config.get("min_tokens", 30)
            metrics, filtered_texts, plot_path, report_path = evaluate_dataset(
                all_texts, 
                output_dir="static",
                min_tokens=min_token_threshold
            )
            
            # Log all metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log the plot and report
            mlflow.log_artifact(plot_path)
            mlflow.log_artifact(report_path)
            
            update_preprocessing_status(progress=85, message='Saving filtered data...')
            
            # Save filtered dataset
            output_file = os.path.join("static", "datasets.txt")           
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(text.strip() for text in filtered_texts if isinstance(text, str) and text.strip()))
            
            mlflow.log_artifact(output_file)
            
            # Log final metrics
            mlflow.log_metric("total_samples", len(all_texts))
            mlflow.log_metric("total_datasets_used", len(datasets_texts)) 
            mlflow.log_param("min_token_threshold", min_token_threshold)
            
            # Log preprocessing report
            report = create_preprocessing_report()
            mlflow.log_text(report, "data_preprocessing_report.txt")
            
            update_preprocessing_status(
                progress=100,
                message=f'✅ Preprocessing completed! Final dataset: {metrics["total_documents_after_filter"]:,} samples '
                        f'({metrics["total_tokens_after_filter"]:,} tokens, {metrics["documents_removed"]:,} removed)',
                running=False,
                end_time=datetime.now(),
                experiment_name=config["experiment_name"],
            )
            
            return True
            
    except Exception as e:
        error_message = str(e)
        update_preprocessing_status(
            running=False,
            error=error_message,
            message=f'❌ Error occurred: {error_message}',
            end_time=datetime.now(),
            experiment_name=config["experiment_name"],
        )
        mlflow.end_run(status="FAILED")
        
        # Log error to MLflow if possible
        try:
            mlflow.log_param("status", "failed")
            mlflow.log_text(str(e), "error_log.txt")
        except:
            pass
            
        return False
