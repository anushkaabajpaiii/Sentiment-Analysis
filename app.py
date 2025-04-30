import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import movie_reviews
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Download NLTK data
nltk.download('movie_reviews')

# Load and shuffle data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

texts = [" ".join(words) for words, label in documents]
labels = [label for words, label in documents]

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

texts = [preprocess_text(text) for text in texts]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("🎭 Sentiment Analysis App")
st.write("This app predicts sentiment (Positive/Negative) based on your input text.")

# Sample sentences dropdown
sample_inputs = {
    "Select an example...": "",
    "This movie was absolutely fantastic, the acting was brilliant!": "This movie was absolutely fantastic, the acting was brilliant!",
    "The plot was boring and the characters were poorly written.": "The plot was boring and the characters were poorly written.",
    "I loved every second of the film, it kept me engaged throughout.": "I loved every second of the film, it kept me engaged throughout.",
    "A complete waste of time, I wouldn't recommend this movie to anyone.": "A complete waste of time, I wouldn't recommend this movie to anyone."
}
selected_sample = st.selectbox("Or choose an example sentence:", list(sample_inputs.keys()))

# User input
user_input = st.text_area("Enter a sentence or review:", value=sample_inputs[selected_sample], height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed = preprocess_text(user_input)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]

        st.success(f"Predicted Sentiment: **{prediction.upper()}**")

        # Confidence bar chart
        st.subheader("🔍 Confidence Scores")
        proba_df = pd.DataFrame({"Sentiment": model.classes_, "Probability": proba})
        fig, ax = plt.subplots()
        sns.barplot(x="Sentiment", y="Probability", data=proba_df, palette="viridis", ax=ax)
        ax.set_ylim(0, 1)
        st.pyplot(fig)

# Example reviews
st.markdown("---")
st.subheader("📋 Sample Test Reviews and Predictions")
examples = random.sample(list(zip(X_test, y_test)), 5)
for review, true_label in examples:
    pred = model.predict(vectorizer.transform([review]))[0]
    st.markdown(f"**Review:** {review[:150]}...")
    st.write(f"True: `{true_label}` | Predicted: `{pred}`")

# WordCloud
st.markdown("---")
st.subheader("🌥️ Word Cloud of Positive and Negative Words")
positive_text = " ".join([X_train[i] for i in range(len(X_train)) if y_train[i] == "pos"])
negative_text = " ".join([X_train[i] for i in range(len(X_train)) if y_train[i] == "neg"])

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Positive Reviews WordCloud**")
    wc_pos = WordCloud(width=400, height=200, background_color='white').generate(positive_text)
    st.image(wc_pos.to_array())

with col2:
    st.markdown("**Negative Reviews WordCloud**")
    wc_neg = WordCloud(width=400, height=200, background_color='white').generate(negative_text)
    st.image(wc_neg.to_array())

# Footer
st.markdown("---")
st.write("Model: Multinomial Naive Bayes | Features: TF-IDF (Top 5000 Words)")
