# Headline Sentiment Scorer

This project uses a pre-trained SVM classifier and sentence embeddings from a transformer model to assign sentiment labels to news headlines. It outputs labeled results for downstream analytics used by hedge funds, governments, and media research teams.

---

## Project Overview

- **Input:** A `.txt` file with one headline per line
- **Output:** A new `.txt` file with sentiment label + original headline
- **Model:** SVM classifier trained on SentenceTransformer embeddings
- **Sentiment Classes:** `Optimistic`, `Pessimistic`, `Neutral`

---

## Usage Instructions

### 1. Clone the Repository
```bash
python score_headlines.py <input_file.txt> <source_name>
