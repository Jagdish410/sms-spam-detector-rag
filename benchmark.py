"""Compare Naive Bayes vs a zero-shot LLM classifier on the same test messages.

Run:  export ANTHROPIC_API_KEY=your_key   (Windows: set ANTHROPIC_API_KEY=your_key)
      python benchmark.py
"""
import os

import anthropic
from sklearn.metrics import classification_report

from data import load_data
from explain import LLM_MODEL
from train import load_model, split

N = 100  # number of test messages to evaluate (keeps API cost tiny)

df = load_data()
_, test_df = split(df)
sample = test_df.sample(N, random_state=42)

nb = load_model()
nb_preds = nb.predict(sample["text"])

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY to run the LLM part of the benchmark.")

client = anthropic.Anthropic()
llm_preds = []
for text in sample["text"]:
    resp = client.messages.create(
        model=LLM_MODEL, max_tokens=5,
        messages=[{"role": "user", "content":
                   f"Classify this SMS as spam or ham. Reply with one word only.\n\n{text}"}],
    )
    llm_preds.append(1 if "spam" in resp.content[0].text.lower() else 0)

print(f"\n=== Naive Bayes ({N} messages) ===")
print(classification_report(sample["y"], nb_preds, target_names=["ham", "spam"]))
print(f"=== Zero-shot LLM ({N} messages) ===")
print(classification_report(sample["y"], llm_preds, target_names=["ham", "spam"]))
