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
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove Gutenberg headers/footers
    text = re.sub(r"\*\*\* START OF.*?\*\*\*", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*\* END OF.*?\*\*\*", "", text, flags=re.DOTALL)
    
    # Remove non-alphabetic chars except basic punctuation
    text = re.sub(r"[^a-zA-Z0-9.,;:?!'\"\-\s]", " ", text)
    
    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    
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
        return ["Sample Gutenberg text for demonstration purposes."],1

def load_bookcorpus(config):
    """Load WikiText-103 as BookCorpus substitute"""
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        #texts = [clean_text(x["text"]) for x in dataset if x["text"].strip()]
        texts = [x["text"].strip() for x in dataset if x["text"].strip()]
        return texts[:int(config["limit_load"])],len(texts)  # Limit to first 1000 for demo
    except Exception as e:
        print(f"Failed to load WikiText dataset: {e}")
        return ["Sample WikiText data for demonstration purposes."],1

def load_poetry(config):
    """Load poetry dataset (using AG News as placeholder)"""
    try:
        dataset = load_dataset("DanFosing/public-domain-poetry")
        #texts = [clean_text(x["text"]) for x in dataset["train"]]
        texts = [x["text"].strip() for x in dataset["train"]]
        return texts[:int(config["limit_load"])],len(texts)
    except Exception as e:
        print(f"Failed to load poetry dataset: {e}")
        return ["The woods are lovely, dark and deep, but I have promises to keep..."],1

def create_preprocessing_report():
    """Generate preprocessing report"""
    report = f"""
Data Preprocessing Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
-------------------------
1. Removed HTML tags using BeautifulSoup.
2. Removed Gutenberg headers and footers using regex.
3. Removed non-alphabetic characters except basic punctuation.
4. Normalized whitespace.
5. Tokenized sentences using {'SpaCy' if get_spacy_model() else 'Regex fallback'}.
6. Loaded datasets:
   - Gutenberg (via NLTK)
   - BookCorpus substitute (WikiText-103)
   - DanFosing/public-domain-poetry is a dataset
7. Merged all sources into a single DataFrame.
8. Saved the cleaned dataset as CSV.

Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return report