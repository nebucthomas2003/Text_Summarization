import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest
import numpy as np
import networkx as nx
import pickle
from summa import summarizer
from rouge import Rouge

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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
    return summary_sentences, vectorizer

def textrank_summarize(text, ratio=0.3):
    summary = summarizer.summarize(text, ratio=ratio)
    return summary.split('\n')

def evaluate_with_rouge(generated_summary, expected_summary):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, expected_summary)
    return scores

def save_model(vectorizer, filename='tfidf_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(vectorizer, file)
        
def load_model(filename='tfidf_model.pkl'):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_similarity_matrix(matrix, filename='similarity_matrix.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(matrix, file)
        
def load_similarity_matrix(filename='similarity_matrix.pkl'):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Example usage
if __name__ == "__main__":
    text = """
    Infosys is a global leader in next-generation digital services and consulting. 
    We enable clients in more than 50 countries to navigate their digital transformation. 
    With over four decades of experience in managing the systems and workings of global enterprises, 
    we expertly steer our clients through their digital journey. We do it by enabling the enterprise 
    with an AI-powered core that helps prioritize the execution of change. We also empower the business 
    with agile digital at scale to deliver unprecedented levels of performance and customer delight. 
    Our always-on learning agenda drives their continuous improvement through building and 
    transferring digital skills, expertise, and ideas from our innovation ecosystem. 
    """

    # Generate TF-IDF summary
    tfidf_summary, vectorizer = summarize_with_tfidf(text)
    print("TF-IDF Summary:\n")
    print("\n".join(tfidf_summary))

    # Save TF-IDF model
    save_model(vectorizer, 'tfidf_model.pkl')

    # Load TF-IDF model
    loaded_vectorizer = load_model('tfidf_model.pkl')

    # Generate TextRank summary
    textrank_summary = textrank_summarize(text)
    print("\nTextRank Summary:\n")
    print("\n".join(textrank_summary))

    # Evaluate summaries with ROUGE
    expected_summary = "Infosys is a global leader in next-generation digital services and consulting."
    tfidf_scores = evaluate_with_rouge(" ".join(tfidf_summary), expected_summary)
    textrank_scores = evaluate_with_rouge(" ".join(textrank_summary), expected_summary)

    print("\nROUGE Evaluation:")
    print("TF-IDF Scores:", tfidf_scores)
    print("TextRank Scores:", textrank_scores)
