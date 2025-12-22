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
            update_preprocessing_status(progress=10, message='Loading Gutenberg texts...')
            gutenberg_texts,Gun_len = load_gutenberg(config)
            mlflow.log_metric("gutenberg_count", Gun_len)

            update_preprocessing_status(progress=30, message='Loading BookCorpus texts...')
            bookcorpus_texts,book_len = load_bookcorpus(config)
            mlflow.log_metric("bookcorpus_count", book_len)
            
            
            update_preprocessing_status(progress=50, message='Loading Poetry texts...')
            poetry_texts,poetry_len = load_poetry(config)
            mlflow.log_metric("poetry_count", poetry_len)

            update_preprocessing_status(progress=60, message='Loading selected datasets '+str(config['selected_datasets'])+'...')
            
            # Load additional datasets if specified
            selected_datasets = config.get("selected_datasets", [])
            count_other_datasets = 0
            datasets_texts = {}
            for dataset_name in selected_datasets:
                texts, count = load_chosen_dataset(dataset_name=dataset_name, config=config)
                datasets_texts[dataset_name] = texts
                count_other_datasets += count
                mlflow.log_metric(f"{dataset_name}_count", count)

            

            update_preprocessing_status(progress=70, message='Creating DataFrame...')
            
            # Create combined DataFrame
#            df = pd.DataFrame({
                #"source": ["Gutenberg"] * len(gutenberg_texts) +
                 #         ["BookCorpus"] * len(bookcorpus_texts) +
                  #        ["Poetry"] * len(poetry_texts),
#                "text": gutenberg_texts + bookcorpus_texts + poetry_texts
#            })
            # Add additional datasets to DataFrame

#            for dataset_name, texts in datasets_texts.items():
#                temp_df = pd.DataFrame({
#                    #"source": [dataset_name] * len(texts),
#                    "text": texts
#                 })
      #           df = pd.concat([df, temp_df], ignore_index=True)    
            update_preprocessing_status(progress=75, message='Cleaning texts...')
            # Clean texts
            update_preprocessing_status(progress=85, message='Cleaning sample and saving...')
            all_texts = []
            all_texts.extend(gutenberg_texts)
            all_texts.extend(bookcorpus_texts)
            all_texts.extend(poetry_texts)

            for dataset_name, texts in datasets_texts.items():
                all_texts.extend(texts)

            # Save dataset
            
            output_file = os.path.join("static", "datasets.txt")           
            #df.to_csv(output_file, index=False)
            
            with open(output_file, "w", encoding="utf-8") as f:
                for text in all_texts:
                    if isinstance(text, str) and text.strip():
                        f.write(text.strip() + "\n")
            mlflow.log_artifact(output_file)

            #mlflow.log_metric("total_samples", len(df))
            
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
