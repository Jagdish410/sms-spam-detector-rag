# SMS Spam Detector with RAG-powered Explanations

A hybrid spam detection system:

1. **TF-IDF + Naive Bayes** gives a fast spam/ham prediction.
2. **Sentence embeddings + FAISS** retrieve the most similar labeled messages.
3. **An LLM (Claude)** uses those examples to explain *why* the message is spam or not.

```
New message -> Naive Bayes -> embed -> FAISS retrieval -> LLM explanation -> result
```

## Tech stack
Python, scikit-learn, sentence-transformers, FAISS, Claude API, Streamlit

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py                      # trains model, prints metrics, saves confusion matrix
streamlit run app.py                 # launches the demo app
python benchmark.py                  # (optional) Naive Bayes vs zero-shot LLM
```

The dataset downloads automatically on first run.

### Optional: enable LLM explanations
Without a key, the app still works and shows a simple rule-based explanation.

```bash
# Mac/Linux
export ANTHROPIC_API_KEY="your_key"
# Windows (cmd)
set ANTHROPIC_API_KEY=your_key
```

## Project structure

| File | Purpose |
|---|---|
| `data.py` | Downloads and loads the SMS Spam Collection dataset |
| `train.py` | Trains and evaluates TF-IDF + Naive Bayes |
| `rag.py` | Embeddings + FAISS similarity search |
| `explain.py` | LLM explanation (with offline fallback) |
| `app.py` | Streamlit web app |
| `benchmark.py` | Compares Naive Bayes with a zero-shot LLM |

## Results
Fill these in after running:

| Model | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|
| Naive Bayes + TF-IDF | | | |
| Zero-shot LLM | | | |

## Key learnings
- Accuracy alone is misleading on imbalanced data (~13% spam), so precision/recall/F1 matter.
- RAG grounds the LLM's explanation in real labeled examples instead of guesses.
- Classical ML is fast and cheap; the LLM adds explainability.
