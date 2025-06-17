
import streamlit as st
from transformers import pipeline

# Load sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_model()

# UI
st.title("📊 Sentiment Analysis Dashboard")

# Text input
user_input = st.text_area("Enter your text here:")

# Predict
if st.button("Analyze Sentiment"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            result = sentiment_model(user_input)[0]
            st.success(f"**Sentiment:** {result['label']} ({result['score']:.2f})")
    else:
        st.warning("Please enter some text.")
