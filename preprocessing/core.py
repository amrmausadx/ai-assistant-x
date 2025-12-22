"""
Optimized preprocessing functions for text data
"""
import re
import pandas as pd
from datasets import load_dataset
from bs4 import BeautifulSoup
from datetime import datetime
from utils.dependencies import get_spacy_model
from typing import List, Tuple

# ==================== COMPILED REGEX PATTERNS ====================
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
GUTENBERG_START = re.compile(r"\*\*\* START OF.*?\*\*\*", re.DOTALL)
GUTENBERG_END = re.compile(r"\*\*\* END OF.*?\*\*\*", re.DOTALL)

# Keep only meaningful punctuation (no quotes)
NON_ALPHANUM = re.compile(r"[^a-zA-Z0-9.,;:?!\-—–…\s]")

NEWLINES = re.compile(r"[\r\n\t]+")
MULTI_PUNCT = re.compile(r"([?.!]){2,}")
WHITESPACE = re.compile(r"\s+")


# ==================== CORE CLEANING ====================
def clean_text(text: str, lowercase: bool = True, min_length: int = 10) -> str:
    """
    Clean and normalize text for language model training.

    Args:
        text: Raw text input
        lowercase: Whether to lowercase text
        min_length: Minimum valid text length

    Returns:
        Cleaned text or empty string
    """

    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        # -------- Remove HTML --------
        if "<" in text and ">" in text:
            text = BeautifulSoup(text, "html.parser").get_text()

        # -------- Remove URLs --------
        text = URL_PATTERN.sub("", text)

        # -------- Remove Gutenberg headers/footers --------
        text = GUTENBERG_START.sub("", text)
        text = GUTENBERG_END.sub("", text)

        # -------- Normalize characters --------
        text = NON_ALPHANUM.sub(" ", text)

        # -------- Normalize whitespace & punctuation --------
        text = NEWLINES.sub(" ", text)
        text = MULTI_PUNCT.sub(r"\1", text)

        # -------- Normalize contractions --------
        # e.g., plum'd → plum d , god's → god s
        text = re.sub(r"\b(\w+)'(\w+)\b", r"\1 \2", text)

        # -------- Collapse repeated words --------
        # e.g., alas, alas → alas | long, long → long
        text = re.sub(r"\b(\w+)(,\s*\1\b)+", r"\1", text)

        # -------- Reduce comma overload --------
        text = re.sub(r"(,\s*){2,}", ", ", text)
        text = re.sub(r",\s+(and|or|but)\b", r" \1", text)

        # -------- Final whitespace cleanup --------
        text = WHITESPACE.sub(" ", text).strip()

        # -------- Optional lowercase --------
        if lowercase:
            text = text.lower()

        return text if len(text) >= min_length else ""

    except Exception as e:
        print(f"[clean_text] Warning: {e}")
        return e 
    
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
            cleaned = clean_text(raw_text, lowercase=config.get("lowercase", True))
            if cleaned:
                texts.append(cleaned)
            total += 1
        
        return texts, total
        
    except Exception as e:
        print(f"❌ Failed to load Gutenberg corpus: {e}")
        return e, 0


def load_bookcorpus(config: dict) -> Tuple[List[str], int]:
    """Load WikiText-103 as BookCorpus substitute with streaming."""
    try:
        limit = int(config.get("limit_load", 1000))
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", 
                              split="train", streaming=True)
        
        texts = []
        total = 0
        
        for i, x in enumerate(dataset):
            if len(texts) >= limit:
                break
            
            text = x.get("text", "")
            if text and text.strip():
                cleaned = clean_text(text, lowercase=config.get("lowercase", True))
                if cleaned:
                    texts.append(cleaned)
                total += 1
        
        return texts, total
        
    except Exception as e:
        print(f"❌ Failed to load WikiText dataset: {e}")
        return e, 0


def load_poetry(config: dict) -> Tuple[List[str], int]:
    """Load poetry dataset with early stopping."""
    try:
        limit = int(config.get("limit_load", 1000))
        dataset = load_dataset("DanFosing/public-domain-poetry", split="train")
        
        texts = []
        total = 0
        
        for i, x in enumerate(dataset):
            if len(texts) >= limit:
                break
            
            text = x.get("text", "")
            if text and text.strip():
                cleaned = clean_text(text, lowercase=config.get("lowercase", True))
                if cleaned:
                    texts.append(cleaned)
                total += 1
        
        return texts, total
        
    except Exception as e:
        print(f"❌ Failed to load poetry dataset: {e}")
        return e, 0


def load_chosen_dataset(dataset_name: str, config: dict) -> Tuple[List[str], int]:
    """
    Dynamically load a Hugging Face dataset with intelligent text extraction.
    
    Args:
        dataset_name: Name of HuggingFace dataset
        config: Configuration dict with 'limit_load' and 'lowercase'
        
    Returns:
        Tuple of (cleaned_texts, total_count)
    """
    try:
        limit = int(config.get("limit_load", 1000))
        
        # Load dataset
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        # Try to get column names (not always available in streaming)
        try:
            if hasattr(dataset, 'column_names'):
                columns = dataset.column_names
            else:
                # Get first sample to inspect columns
                first_sample = next(iter(dataset))
                columns = list(first_sample.keys())
                dataset = load_dataset(dataset_name, split="train", streaming=True)  # Reset
        except:
            columns = []
        
        print(f"📊 Dataset '{dataset_name}' columns: {columns}")
        
        # Find text column
        text_columns = [
            "text", "content", "story", "article", "body", "poem",
            "Text", "Content", "Story", "Article", "Body", "Poem"
        ]
        
        chosen_column = None
        for col in text_columns:
            if col in columns:
                chosen_column = col
                break
        
        if chosen_column is None:
            # Try first string column as fallback
            for col in columns:
                chosen_column = col
                break
        
        if chosen_column is None:
            raise ValueError(f"No suitable text column found in '{dataset_name}'")
        
        print(f"✅ Using column: '{chosen_column}'")
        
        # Load and clean texts with early stopping
        texts = []
        total = 0
        
        for i, x in enumerate(dataset):
            if len(texts) >= limit:
                break
            
            try:
                text = x.get(chosen_column, "")
                if text and isinstance(text, str) and text.strip():
                    cleaned = clean_text(text, lowercase=config.get("lowercase", True))
                    if cleaned:
                        texts.append(cleaned)
                    total += 1
            except Exception as e:
                print(f"⚠️ Skipping sample {i}: {e}")
                continue
        
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
  ✓ Lowercase normalization
"""
    return report
