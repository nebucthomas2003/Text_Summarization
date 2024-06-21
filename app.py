import gradio as gr
from transformers import BartForConditionalGeneration, BartTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest
import numpy as np
from summa import summarizer
from rouge_score import rouge_scorer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load saved model and tokenizer for abstractive summarization
saved_model_directory = './saved_model'
tokenizer = BartTokenizer.from_pretrained(saved_model_directory)
model = BartForConditionalGeneration.from_pretrained(saved_model_directory)

def preprocess_text(text):
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
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=0.1, ngram_range=(1, 2))
    X = vectorizer.fit_transform(sentences)
    return vectorizer, X

def rank_sentences_with_scores(sentences, original_sentences, top_n=5):
    vectorizer, tfidf_matrix = compute_tfidf(sentences)
    scores = tfidf_matrix.sum(axis=1).flatten()
    ranked_indices = nlargest(top_n, range(len(scores)), scores.take)
    ranked_sentences = [original_sentences[i] for i in ranked_indices]
    ranked_scores = [scores[i] for i in ranked_indices]
    return ranked_sentences, ranked_scores, vectorizer

def summarize_with_tfidf(text, top_n=5):
    original_sentences, cleaned_sentences = preprocess_text(text)
    summary_sentences, _, vectorizer = rank_sentences_with_scores(cleaned_sentences, original_sentences, top_n)
    return summary_sentences

def textrank_summarize(text, ratio=0.3):
    summary = summarizer.summarize(text, ratio=ratio)
    return summary.split('\n')

def abstractive_summarize(article):
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
    if summarization_type == "Abstractive":
        return abstractive_summarize(article)
    elif summarization_type == "Extractive (TF-IDF)":
        tfidf_summary = summarize_with_tfidf(article)
        return " ".join(tfidf_summary)
    elif summarization_type == "Extractive (TextRank)":
        textrank_summary = textrank_summarize(article)
        return " ".join(textrank_summary)

def evaluate_with_rouge(generated_summary, expected_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(generated_summary, expected_summary)
    return scores

# Create Gradio Interface
interface = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(lines=10, label="Article Text"),
        gr.Radio(["Abstractive", "Extractive (TF-IDF)", "Extractive (TextRank)"], label="Summarization Type")
    ],
    outputs=gr.Textbox(label="Summary"),
    title="Text Summarization",
    description="Enter the article text to generate a summary."
)

if __name__ == "__main__":
    interface.launch()
