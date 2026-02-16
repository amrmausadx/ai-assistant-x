"""
Main preprocessing pipeline
"""
import os
import re
import mlflow
import pandas as pd
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

def run_preprocessing_pipeline(config,opt=None):
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
            

            update_preprocessing_status(progress=70, message='Combining data...')
            
            # Add additional datasets to DataFrame
            # Clean texts
            all_texts = []

            for idx, (dataset_name, texts) in enumerate(datasets_texts.items(), 1):
                all_texts.extend(texts)
                progress = 70 + (idx / len(datasets_texts)) * 15  # Progress from 70-85%
                update_preprocessing_status(progress=int(progress), 
                                            message=f'Combining texts ({idx}/{len(datasets_texts)} datasets)...')
            
            update_preprocessing_status(progress=85, message='Saving Data...')
            
            # Save dataset
            output_file = os.path.join("static", "datasets.txt")           
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(text.strip() for text in all_texts if isinstance(text, str) and text.strip()))
            
            mlflow.log_artifact(output_file)
            
            # ADD THIS LINE:
            mlflow.log_metric("total_samples", len(all_texts))
            mlflow.log_metric("total_datasets_used", len(datasets_texts)) 

            
            # Log preprocessing report
            report = create_preprocessing_report()
            mlflow.log_text(report, "data_preprocessing_report.txt")
            
            update_preprocessing_status(
                progress=100,
                message=f'✅ Preprocessing completed! Dataset size: {len(all_texts)} samples',
                running=False,
                end_time=datetime.now(),
                experiment_name = config["experiment_name"],
            )
            
            return True
            
    except Exception as e:
        error_message = str(e)
        update_preprocessing_status(
            running=False,
            error=error_message,
            message=f'❌ Error occurred: {error_message}',
            end_time=datetime.now(),
            experiment_name = config["experiment_name"],
        )
        mlflow.end_run(status="FAILED")
        
        # Log error to MLflow if possible
        try:
            mlflow.log_param("status", "failed")
            mlflow.log_text(str(e), "error_log.txt")
        except:
            pass
            
        return False
