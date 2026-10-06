"""Train the TF-IDF + Naive Bayes baseline and evaluate it.

Run:  python train.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data import load_data

MODEL_PATH = Path("models/nb_pipeline.joblib")


def split(df):
    return train_test_split(df, test_size=0.2, random_state=42, stratify=df["y"])


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("nb", MultinomialNB(alpha=0.1)),
    ])


def load_model():
    if not MODEL_PATH.exists():
        main()
    return joblib.load(MODEL_PATH)


def predict(model, text: str):
    """Return (label, confidence) where label is 'spam' or 'ham'."""
    proba = model.predict_proba([text])[0]
    spam_p = float(proba[1])
    return ("spam", spam_p) if spam_p >= 0.5 else ("ham", 1 - spam_p)


def main():
    df = load_data()
    print(f"Loaded {len(df)} messages | spam ratio: {df['y'].mean():.1%}")
    train_df, test_df = split(df)

    model = build_pipeline()
    model.fit(train_df["text"], train_df["y"])
    preds = model.predict(test_df["text"])

    print("\n=== Test set results ===")
    print(classification_report(test_df["y"], preds, target_names=["ham", "spam"]))

    cm = confusion_matrix(test_df["y"], preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Naive Bayes - Confusion Matrix")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.close()

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH} and confusion_matrix.png")


if __name__ == "__main__":
    main()
