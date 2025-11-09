# app.py
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title="SentimentScope", page_icon="💬", layout="centered")

@st.cache_resource
def load_model():
    model_path = Path("models/tfidf_logreg.joblib")
    return joblib.load(model_path)

model = load_model()

st.title("💬 SentimentScope")
st.caption("TF-IDF + Logistic Regression")

txt = st.text_area("Write a review / sentence:", height=140, placeholder="e.g., The movie was fantastic and I loved the acting!")
col1, col2 = st.columns([1,1])

if st.button("Analyze"):
    if txt.strip():
        pred = model.predict([txt])[0]
        proba = model.predict_proba([txt])[0]
        conf = float(max(proba)) * 100
        st.markdown(f"### Prediction: **{pred.upper()}**")
        st.progress(int(conf))
        st.write(f"Confidence: **{conf:.1f}%**")
    else:
        st.warning("Please enter some text.")

with st.sidebar:
    st.subheader("Examples")
    for ex in [
        "Absolutely wonderful experience!",
        "Not worth the money.",
        "It was okay, nothing special."
    ]:
        if st.button(ex, key=ex):
            st.session_state["text"] = ex
            st.experimental_rerun()
