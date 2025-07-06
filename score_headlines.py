#!/usr/bin/env python3

import sys
import os
import datetime
import joblib
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "svm.joblib"  
EMBEDDING_MODEL_PATH = "/opt/huggingface_models/all-MiniLM-L6-v2"
# EMBEDDING_MODEL_PATH = "all-MiniLM-L6-v2"


def load_headlines(input_file: str) -> list:
    """Load headlines from the specified file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Input file '{input_file}' not found.")
        sys.exit(1)

def vectorize_headlines(headlines: list, embedder: SentenceTransformer) -> list:
    """Convert headlines to sentence embeddings."""
    return embedder.encode(headlines)

def load_model(model_path: str):
    """Load the pre-trained SVM model."""
    if not os.path.exists(model_path):
        logger.error(f"SVM model not found at '{model_path}'")
        sys.exit(1)
    return joblib.load(model_path)

def write_output(predictions: list, headlines: list, source: str):
    """Write predictions to the output file."""
    today = datetime.datetime.now().strftime("%Y_%m_%d")
    filename = f"headline_scores_{source}_{today}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for label, headline in zip(predictions, headlines):
            f.write(f"{label},{headline}\n")
    logger.info(f"Predictions written to: {filename}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python score_headlines.py <input_file.txt> <source_name>")
        sys.exit(1)

    input_file, source = sys.argv[1], sys.argv[2]

    logger.info("Loading headlines...")
    headlines = load_headlines(input_file)

    logger.info("Loading sentence transformer model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_PATH)

    logger.info("Vectorizing headlines...")
    vectors = vectorize_headlines(headlines, embedder)

    logger.info("Loading SVM sentiment classifier...")
    model = load_model(MODEL_PATH)

    logger.info("Generating predictions...")
    predictions = model.predict(vectors)

    logger.info("Writing output...")
    write_output(predictions, headlines, source)

if __name__ == "__main__":
    main()
