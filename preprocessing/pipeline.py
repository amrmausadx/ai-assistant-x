"""
Main preprocessing pipeline
"""
import mlflow
import pandas as pd
from datetime import datetime
from .core import (
    load_gutenberg, 
    load_bookcorpus, 
    load_poetry, 
    tokenize_sentences,
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
            
            update_preprocessing_status(progress=30, message='Loading BookCorpus texts...')
            bookcorpus_texts,book_len = load_bookcorpus(config)
            
            update_preprocessing_status(progress=50, message='Loading Poetry texts...')
            poetry_texts,poetry_len = load_poetry(config)
            
            # Log dataset sizes
            mlflow.log_metric("gutenberg_count", len(gutenberg_texts))
            mlflow.log_metric("bookcorpus_count", len(bookcorpus_texts))
            mlflow.log_metric("poetry_count", len(poetry_texts))
            
            update_preprocessing_status(progress=70, message='Creating DataFrame...')
            
            # Create combined DataFrame
            df = pd.DataFrame({
                #"source": ["Gutenberg"] * len(gutenberg_texts) +
                 #         ["BookCorpus"] * len(bookcorpus_texts) +
                  #        ["Poetry"] * len(poetry_texts),
                "text": gutenberg_texts + bookcorpus_texts + poetry_texts
            })
            
            update_preprocessing_status(progress=85, message='Tokenizing sample and saving...')
            
            # Process sample for demonstration
            #if len(df) > 0:
            #    sample_sentences = tokenize_sentences(df.iloc[0]["text"])
            #    mlflow.log_text("\n".join(sample_sentences[:5]), "sample_sentences.txt")
            
            # Save dataset
            output_file = "creative_writing_dataset.csv"
            df.to_csv(output_file, index=False)
            mlflow.log_artifact(output_file)
            #mlflow.log_metric("total_samples", len(df))
            
            # Log preprocessing report
            report = create_preprocessing_report()
            mlflow.log_text(report, "data_preprocessing_report.txt")
            
            update_preprocessing_status(
                progress=100,
                message=f'✅ Preprocessing completed! Dataset size: {len(df)} samples of {Gun_len+book_len+poetry_len}',
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