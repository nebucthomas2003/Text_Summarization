import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest
from summa import summarizer
from rouge_score import rouge_scorer
import numpy as np
import os

# Download NLTK data if not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load BART Model and Tokenizer
saved_model_directory = './saved_model'
tokenizer = BartTokenizer.from_pretrained(saved_model_directory)
model = BartForConditionalGeneration.from_pretrained(saved_model_directory)

def preprocess_text(text):
    """Preprocesses text by tokenizing sentences, removing stopwords, and cleaning each sentence."""
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    cleaned_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalnum()]
        words = [word for word in words if word not in stop_words]
        cleaned_sentences.append(' '.join(words))
    
    return sentences, cleaned_sentences

def compute_tfidf(sentences):
    """Computes TF-IDF scores for sentences."""
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=0.1, ngram_range=(1, 2))
    X = vectorizer.fit_transform(sentences)
    return vectorizer, X

def rank_sentences_with_scores(sentences, original_sentences, top_n=5):
    """Ranks sentences based on TF-IDF scores."""
    vectorizer, tfidf_matrix = compute_tfidf(sentences)
    scores = tfidf_matrix.sum(axis=1).flatten()
    ranked_indices = nlargest(top_n, range(len(scores)), scores.take)
    ranked_sentences = [original_sentences[i] for i in ranked_indices]
    ranked_scores = [scores[i] for i in ranked_indices]
    return ranked_sentences, ranked_scores, vectorizer

def summarize_with_tfidf(text, top_n=5):
    """Summarizes text using TF-IDF extractive summarization."""
    original_sentences, cleaned_sentences = preprocess_text(text)
    summary_sentences, _, vectorizer = rank_sentences_with_scores(cleaned_sentences, original_sentences, top_n)
    return " ".join(summary_sentences)

def textrank_summarize(text, ratio=0.3):
    """Summarizes text using TextRank extractive summarization."""
    summary = summarizer.summarize(text, ratio=ratio)
    return summary.split('\n')

def abstractive_summarize(article):
    """Summarizes text using abstractive BART model."""
    inputs = tokenizer(article, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=150,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize(article, summarization_type):
    """Routes text summarization based on selected type."""
    if summarization_type == "Abstractive":
        return abstractive_summarize(article)
    elif summarization_type == "Extractive (TF-IDF)":
        return summarize_with_tfidf(article)
    elif summarization_type == "Extractive (TextRank)":
        return " ".join(textrank_summarize(article))

def main():
    st.title('Text Summarization')
    st.markdown("Enter the article text to generate a summary.")

    article_text = st.text_area('Article Text', height=300)
    summarization_type = st.radio("Summarization Type", ["Abstractive", "Extractive (TF-IDF)", "Extractive (TextRank)"])

    if st.button('Generate Summary'):
        if article_text:
            summary = summarize(article_text, summarization_type)
            st.subheader('Summary')
            st.write(summary)

if __name__ == '__main__':
    main()
