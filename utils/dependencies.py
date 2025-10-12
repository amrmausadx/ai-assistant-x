"""
Dependency management and availability checking
"""

def check_dependencies():
    """Check which dependencies are available"""
    deps = {
        'preprocessing': True,
        'training': True,
        'generation': True,
        'errors': []
    }
    
    # Check preprocessing dependencies
    try:
        import nltk
        import spacy
        import pandas as pd
        from datasets import load_dataset
        from bs4 import BeautifulSoup
        import mlflow
    except ImportError as e:
        deps['preprocessing'] = False
        deps['errors'].append(f"Preprocessing: {str(e)}")
    
    # Check training dependencies
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
        from transformers.trainer_callback import TrainerCallback
        from evaluate import load as load_eval
        import torch
    except ImportError as e:
        deps['training'] = False
        deps['errors'].append(f"Training: {str(e)}")
    
    # Check spaCy model
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError) as e:
        deps['errors'].append("SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

    # Check generation dependencies
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        deps['generation'] = False
        deps['errors'].append(f"Generation: {str(e)}")

    
    return deps

def setup_nltk():
    """Download required NLTK data"""
    try:
        import nltk
        nltk.download('gutenberg', quiet=True)
        nltk.download("punkt", quiet=True)
        
        return True
    except Exception as e:
        print(f"Failed to setup NLTK: {e}")
        return False

def get_spacy_model():
    """Get spaCy model if available"""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except:
        return None
    
