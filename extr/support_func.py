import os 
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import requests
from sentence_transformers import SentenceTransformer
import numpy as np  
import sys

import re
from typing import List







def chunksizing(chunk_size: int, reden: str) -> List[str]:
    """
    Split `reden` into chunks of (roughly) <= chunk_size characters,
    without ending a chunk in the middle of a sentence.

    - Sentences are detected by punctuation: ., !, ? (optionally followed by quotes/brackets)
    - Whitespace is normalized between sentences in chunks.

    Returns: List of chunk strings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not reden or not reden.strip():
        return []

    text = re.sub(r"\s+", " ", reden.strip())

    # Sentence split that keeps punctuation with the sentence.
    # Splits on whitespace that follows a sentence-ending punctuation.
    sentences = re.split(r'(?<=[.!?])(?:["\')\]]+)?\s+', text)

    chunks: List[str] = []
    current = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # If adding this sentence would exceed the chunk size, flush current first
        if current:
            candidate = current + " " + s
        else:
            candidate = s

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # flush current if it exists
            if current:
                chunks.append(current)
                current = ""

            # If a single sentence is longer than chunk_size, we can't avoid splitting
            # (otherwise we'd create an infinite loop). So we hard-split that sentence.
            if len(s) > chunk_size:
                for i in range(0, len(s), chunk_size):
                    part = s[i:i + chunk_size].strip()
                    if part:
                        chunks.append(part)
            else:
                current = s

    if current:
        chunks.append(current)

    return chunks
