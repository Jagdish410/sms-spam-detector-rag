"""Generate a plain-English explanation using retrieved examples (the 'G' in RAG).

Uses the Claude API if ANTHROPIC_API_KEY is set; otherwise falls back to a
simple rule-based explanation so the app still works without any key.
"""
import os

LLM_MODEL = "claude-haiku-4-5-20251001"


def _fallback(label, neighbors):
    spam_n = sum(n["label"] == "spam" for n in neighbors)
    return (
        f"The classifier predicts **{label}**. Among the {len(neighbors)} most similar "
        f"known messages, {spam_n} were spam and {len(neighbors) - spam_n} were ham. "
        "(Set ANTHROPIC_API_KEY to get a full LLM-written explanation.)"
    )


def explain(message, label, confidence, neighbors):
    if not os.getenv("ANTHROPIC_API_KEY"):
        return _fallback(label, neighbors)

    import anthropic

    examples = "\n".join(
        f"- [{n['label'].upper()}] (similarity {n['similarity']:.2f}) {n['text']}"
        for n in neighbors
    )
    prompt = f"""You are an SMS spam analyst.

New message:
\"\"\"{message}\"\"\"

A Naive Bayes classifier predicts: {label.upper()} (confidence {confidence:.0%}).

Most similar labeled messages from our database:
{examples}

In 3-4 short sentences, say whether you agree with the prediction and why. Point out
specific scam patterns (urgency, prizes, links, premium numbers, etc.) if present, and
refer to the similar examples as evidence. Do not invent facts."""

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=LLM_MODEL, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except Exception as e:
        return _fallback(label, neighbors) + f"\n\n(LLM call failed: {e})"
