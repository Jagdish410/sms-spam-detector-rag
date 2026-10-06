"""Streamlit demo.  Run:  streamlit run app.py"""
import streamlit as st

from data import load_data
from explain import explain
from rag import Retriever
from train import load_model, predict

st.set_page_config(page_title="SMS Spam Detector (RAG)", page_icon="📩")


@st.cache_resource
def load_everything():
    df = load_data()
    return load_model(), Retriever(df)


st.title("📩 SMS Spam Detector with RAG Explanations")
st.caption("TF-IDF + Naive Bayes prediction, FAISS retrieval, LLM-written explanation")

model, retriever = load_everything()

samples = {
    "Spam example": "WINNER!! You have been selected for a free iPhone. Claim now: http://bit.ly/xyz",
    "Ham example": "Hey, are we still meeting for lunch at 1pm tomorrow?",
}
choice = st.selectbox("Try a sample (optional)", ["-"] + list(samples))
default = samples.get(choice, "")
message = st.text_area("Enter an SMS message", value=default, height=120)

if st.button("Analyze") and message.strip():
    label, conf = predict(model, message)
    neighbors = retriever.search(message, k=5)

    if label == "spam":
        st.error(f"🚨 SPAM  (confidence {conf:.0%})")
    else:
        st.success(f"✅ NOT SPAM  (confidence {conf:.0%})")

    with st.spinner("Generating explanation..."):
        st.subheader("Explanation")
        st.write(explain(message, label, conf, neighbors))

    st.subheader("Similar messages from the database")
    st.table(
        [{"label": n["label"], "similarity": round(n["similarity"], 2), "message": n["text"]}
         for n in neighbors]
    )
