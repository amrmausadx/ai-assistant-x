"""
Core preprocessing functions for text data
"""
import re
import pandas as pd
from datasets import load_dataset
from bs4 import BeautifulSoup
from datetime import datetime
from utils.dependencies import get_spacy_model

def clean_text(text):
    """Clean and normalize text for training."""
    if not text:
        return ""

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove Gutenberg headers/footers
    text = re.sub(r"\*\*\* START OF.*?\*\*\*", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*\* END OF.*?\*\*\*", "", text, flags=re.DOTALL)

    # Remove non-alphanumeric except basic punctuation
    text = re.sub(r"[^a-zA-Z0-9.,;:?!'\"\-\s]", " ", text)

    # Remove extra newlines and tabs
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Normalize multiple punctuation
    text = re.sub(r"([?.!]){2,}", r"\1", text)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Optional: lowercase normalization
    text = text.lower()

    return text


def tokenize_sentences(text):
    """Split into sentences using SpaCy or fallback to regex"""
    nlp = get_spacy_model()
    nlp.max_length = 5_000_000  # increase max length safely

    if nlp is None:
        # Fallback to simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [sent.strip() for sent in sentences if sent.strip()]
    
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def load_gutenberg(config):
     from nltk.corpus import gutenberg
     try:
         #texts = [clean_text(gutenberg.raw(fileid)) for fileid in gutenberg.fileids()] 
         texts = [clean_text(gutenberg.raw(fileid)) for fileid in gutenberg.fileids()] 
         return texts[:int(config["limit_load"])], len(texts)
     except Exception as e: 
        print(f"Failed to load Gutenberg corpus: {e}") 
        return e,1

def load_bookcorpus(config):
    """Load WikiText-103 as BookCorpus substitute"""
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        #texts = [clean_text(x["text"]) for x in dataset if x["text"].strip()]
        texts = [x["text"].strip() for x in dataset if x["text"].strip()]
        return texts[:int(config["limit_load"])],len(texts)  # Limit to first 1000 for demo
    except Exception as e:
        print(f"Failed to load WikiText dataset: {e}")
        return e,1

def load_poetry(config):
    """Load poetry dataset (using AG News as placeholder)"""
    try:
        dataset = load_dataset("DanFosing/public-domain-poetry")
        #texts = [clean_text(x["text"]) for x in dataset["train"]]
        texts = [x["text"].strip() for x in dataset["train"]]
        return texts[:int(config["limit_load"])],len(texts)
    except Exception as e:
        print(f"Failed to load poetry dataset: {e}")
        return e,1


def load_chosen_dataset(dataset_name, config):
    """Dynamically load a Hugging Face dataset and extract text intelligently."""
    try:
        dataset = load_dataset(dataset_name, streaming=True)
        split_name = next(iter(dataset.keys()))
        ds = dataset[split_name]
        limit = int(config.get("limit_load", 100))
        # Find text column: check common patterns, then fall back to first string column
        text_col = next(
            (c for c in ds.column_names 
             if any(kw in c.lower() for kw in ["text", "content", "sentence", "poem", "body","story"])),
            next((c for c in ds.column_names if isinstance(ds[0][c], str)), None)
        )
        
        if not text_col:
            raise ValueError(f"No text column found in {dataset_name}. Columns: {ds.column_names}")
        
        print(f"Using column '{text_col}' for text extraction from {dataset_name}")
        texts=[]
        for i, sample in enumerate(ds):
            if i >= limit:
                break
            text = clean_text(sample.get(text_col, ""))
            if text.strip():
                texts.append(text)
                
        return texts[:limit], len(texts)
    
    except Exception as e:
        print(f"❌ Failed to load chosen dataset '{dataset_name}': {e}")
        return e, 0

def create_preprocessing_report():
    """Generate preprocessing report"""
    report = f"""
Data Preprocessing Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
-------------------------
1. Removed HTML tags using BeautifulSoup.
2. Removed headers and footers using regex.
3. Removed non-alphabetic characters except basic punctuation.
4. Normalized whitespace.
5. Tokenized sentences using {'SpaCy' if get_spacy_model() else 'Regex fallback'}.
6. Loaded datasets
8. Saved the cleaned dataset as CSV.

Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return report