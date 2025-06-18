import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

import pandas as pd
from transformers import pipeline

# Load sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_model()

st.title("💬 Sentiment Analyzer")
st.markdown("Enter **multiple texts (one per line)** or upload a file to analyze their sentiment.")

# File upload
uploaded_file = st.file_uploader("📎 Upload a .txt or .csv file", type=["txt", "csv"])

texts = []

if uploaded_file is not None:
    if uploaded_file.name.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8")
        texts = [line.strip() for line in content.split("\n") if line.strip()]
        st.text_area("📄 File Content", value="\n".join(texts), height=200)
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded CSV:")
        st.dataframe(df.head())
        text_column = st.selectbox("Choose column with text", df.columns)

        if st.button("🔍 Analyze Sentiment in CSV"):
            with st.spinner("Analyzing..."):
                df["Sentiment"] = df[text_column].apply(lambda x: sentiment_model(str(x))[0]['label'])
                df["Confidence"] = df[text_column].apply(lambda x: sentiment_model(str(x))[0]['score'])
                st.success("✅ Done!")
                st.dataframe(df)

                # Chart
                st.subheader("📈 Sentiment Distribution")
                st.bar_chart(df["Sentiment"].value_counts())

                # Download
                st.download_button("⬇️ Download Results", df.to_csv(index=False).encode('utf-8'), file_name="results.csv", mime="text/csv")
else:
    user_input = st.text_area("✍️ Enter multiple texts (one per line):", height=200, placeholder="I love it!\nThis is terrible.\nSo-so experience...")

    if st.button("🔍 Analyze Texts"):
        texts = [line.strip() for line in user_input.split("\n") if line.strip()]
        if texts:
            with st.spinner("Analyzing..."):
                results = [sentiment_model(text)[0] for text in texts]
                df = pd.DataFrame({
                    "Text": texts,
                    "Sentiment": [res["label"] for res in results],
                    "Confidence": [res["score"] for res in results]
                })

                st.success("✅ Sentiment analysis complete!")
                st.dataframe(df)

                st.subheader("📈 Sentiment Breakdown")
                st.bar_chart(df["Sentiment"].value_counts())

                # Optional: Download results
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", csv, file_name="sentiment_texts.csv", mime="text/csv")
        else:
            st.warning("⚠️ Please enter at least one line of text.")



