import streamlit as st
import pickle
import numpy as np
from src.preprocess import clean_text

import os

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = BASE + "/models/svm.pkl"
TFIDF_PATH = BASE + "/models/tfidf.pkl"


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)


st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="centered"
)

# Title
st.title("Sentiment Analysis App")
st.write(
    "This model predicts **Positive**, **Neutral**, **Negative**, or **Irrelevant** sentiment."
)

st.markdown("---")

#Input
user_text = st.text_area(
    "Enter text to analyze:",
    placeholder="Type or paste a sentence...",
    height=150
)

#Button
if st.button("Analyze Sentiment"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess
        cleaned = clean_text(user_text)

        # Vectorize and Predict
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        # Probability (works only if model supports predict_proba)
        try:
            probs = model.predict_proba(vector)[0]
        except:
            probs = None

        #Prediction
        st.subheader("🔍 Prediction Result:")

        sentiment_text = label_names[prediction]
        color = label_colors[prediction]

        st.markdown(
            f"""
            <h3 style='text-align:center; color:{color};'>
                {sentiment_text}
            </h3>
            """,
            unsafe_allow_html=True
        )

       #Probabilty
        if probs is not None:
            st.markdown("### Prediction Confidence")
            for idx, p in enumerate(probs):
                st.progress(float(p))
                st.write(f"**{label_names[idx]}** → `{p:.3f}`")

        st.subheader(" Preprocessed Text")
        st.write(cleaned)




