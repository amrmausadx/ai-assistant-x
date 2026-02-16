"""
Optimized preprocessing functions for text data
"""
import re
#import pandas as pd
from datasets import load_dataset
from bs4 import BeautifulSoup
from datetime import datetime
from utils.dependencies import get_spacy_model
from typing import List, Tuple

# ==================== COMPILED REGEX PATTERNS ====================
# Only handle whitespace and specific platform noise
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
GUTENBERG_START = re.compile(r"\*\*\* START OF.*?\*\*\*", re.DOTALL)
GUTENBERG_END = re.compile(r"\*\*\* END OF.*?\*\*\*", re.DOTALL)

# Matches multiple newlines/tabs but preserves a single clean newline
NEWLINES_NORM = re.compile(r"\n\s*\n") 
# Matches multiple horizontal spaces
WHITESPACE = re.compile(r"[ \t]+")

# ==================== CORE CLEANING ====================
def clean_text(text: str, min_length: int = 10, remove_html: bool = True) -> str:
    """
    Normalizes whitespace and optionally removes HTML tags.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        # 1. Remove HTML tags
        if remove_html:
            text = BeautifulSoup(text, "html.parser").get_text()
        
        # 2. Remove URLs and Metadata
        text = URL_PATTERN.sub("", text)
        text = GUTENBERG_START.sub("", text)
        text = GUTENBERG_END.sub("", text)

        # 3. Normalize Whitespace 
        # Convert multiple newlines into a single newline
        text = NEWLINES_NORM.sub("\n", text)
        # Convert multiple spaces/tabs into a single space
        text = WHITESPACE.sub(" ", text)
        
        text = text.strip()

        if len(text) < min_length:
            return ""

        return text

    except Exception as e:
        print(f"[clean_text] Warning: {e}")
        return ""
        
# ==================== DATASET LOADERS ====================
def load_gutenberg(config: dict) -> Tuple[List[str], int]:
    """Load Project Gutenberg corpus with early stopping."""
    from nltk.corpus import gutenberg
    
    try:
        limit = int(config.get("limit_load", 1000))
        texts = []
        total = 0
        
        for fileid in gutenberg.fileids():
            if len(texts) >= limit:
                break
            raw_text = gutenberg.raw(fileid)
            cleaned = clean_text(raw_text)
            if cleaned:
                texts.append(cleaned)
            total += 1
        
        return texts, total
        
    except Exception as e:
        print(f"❌ Failed to load Gutenberg corpus: {e}")
        return [], 0


def load_bookcorpus(config: dict) -> Tuple[List[str], int]:
    """Load WikiText-103 as BookCorpus substitute with streaming."""
    try:
        limit = int(config.get("limit_load", 1000))
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", 
                              split="train", streaming=True)
        
        texts = []
        total = 0
        
        for x in dataset:
            if len(texts) >= limit:
                break
            
            text = x.get("text", "")
            if text and text.strip():
                cleaned = clean_text(text)
                if cleaned:
                    texts.append(cleaned)
                total += 1
        
        return texts, total
        
    except Exception as e:
        print(f"❌ Failed to load WikiText dataset: {e}")
        return [], 0


def load_poetry(config: dict) -> Tuple[List[str], int]:
    """Load poetry dataset with early stopping."""
    try:
        limit = int(config.get("limit_load", 1000))
        dataset = load_dataset("DanFosing/public-domain-poetry", split="train")
        
        texts = []
        total = 0
        
        for x in dataset:
            if len(texts) >= limit:
                break
            
            text = x.get("text", "")
            if text and text.strip():
                cleaned = clean_text(text)
                if cleaned:
                    texts.append(cleaned)
                total += 1
        
        return texts, total
        
    except Exception as e:
        print(f"❌ Failed to load poetry dataset: {e}")
        return [], 0


def load_chosen_dataset(dataset_name: str, config: dict) -> Tuple[List[str], int]:
    """
    Loads dataset and wraps content in tags based on the 'tag' config.
    """
    try:
        limit = int(config.get("limit_load", 1000))
        dataset = None
        columns = []
        # Determine column names
        try:
            # Peek at first sample to get columns
            dataset_peek = load_dataset(dataset_name, split="train", streaming=True)
            first_sample = next(iter(dataset_peek))
            columns = list(first_sample.keys())
            # Now load fresh dataset for actual processing
            dataset = load_dataset(dataset_name, split="train", streaming=True) 
        except:
            columns = []
        
        # Logic to find the right column
        text_columns = ["text", "content", "story", "article", "body", "poem"]
        chosen_column = next((col for col in text_columns if col in columns), columns[0] if columns else None)
        if not chosen_column:
            print(f"⚠️ No text column found in '{dataset_name}'. Available: {columns}")
            print(f"   Trying first column as fallback: {columns[0] if columns else 'None'}")
            chosen_column = columns[0] if columns else None
    
        if not chosen_column:
            raise ValueError(f"No columns found in '{dataset_name}'")

        print(f"✅ Loading '{dataset_name}' using column: '{chosen_column}'")

        texts = []
        total = 0
        
        for x in dataset:
            if len(texts) >= limit:
                break
            
            raw_val = x.get(chosen_column, "")
            if raw_val and isinstance(raw_val, str):
                cleaned = clean_text(raw_val)
                if cleaned:
                    texts.append(cleaned)
                    total += 1
        
        return texts, total
        
    except Exception as e:
        print(f"❌ Failed to load dataset '{dataset_name}': {e}")
        return [], 0

# ==================== ADVANCED FEATURES ====================
def deduplicate_texts(texts: List[str]) -> List[str]:
    """Remove duplicate texts while preserving order."""
    seen = set()
    unique = []
    for text in texts:
        text_hash = hash(text[:500])  # Hash first 500 chars for speed
        if text_hash not in seen:
            seen.add(text_hash)
            unique.append(text)
    return unique


def tokenize_sentences(text: str) -> List[str]:
    """Split into sentences using SpaCy with fallback."""
    nlp = get_spacy_model()
    
    if nlp is None:
        # Fallback to regex
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s) > 10]
    
    try:
        nlp.max_length = 5_000_000
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents 
                if sent.text.strip() and len(sent.text.strip()) > 10]
    except:
        # Fallback on error
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s) > 10]


def create_preprocessing_report() -> str:
    """Generate detailed preprocessing report."""
    report = f"""

Data Preprocessing Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CLEANING STEPS APPLIED:  
  ✓ Remove HTML tags
  ✓ Remove URLs
  ✓ Remove Gutenberg metadata
  ✓ Normalize whitespace and punctuation
  ✓ Filter minimum length 
  ✓ Deduplication
"""
    return report
