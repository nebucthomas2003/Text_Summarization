import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest
import numpy as np
from summa import summarizer
from rouge import Rouge  # Importing Rouge for ROUGE evaluation
import pickle

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocesses text by tokenizing sentences, removing stopwords, and cleaning each sentence."""
    stop_words = set(stopwords.words('english'))  # Get English stopwords
    sentences = sent_tokenize(text)  # Tokenize into sentences
    cleaned_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)  # Tokenize each sentence into words
        words = [word.lower() for word in words if word.isalnum()]  # Lowercase and remove non-alphanumeric tokens
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        cleaned_sentences.append(' '.join(words))  # Join words back into sentences
    
    return sentences, cleaned_sentences

def compute_tfidf(sentences):
    """Computes TF-IDF scores for sentences."""
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=0.1, ngram_range=(1, 2))
    X = vectorizer.fit_transform(sentences)  # Fit and transform sentences to TF-IDF matrix
    return vectorizer, X  # Return TF-IDF vectorizer and matrix

def rank_sentences_with_scores(sentences, original_sentences, top_n=5):
    """Ranks sentences based on TF-IDF scores."""
    vectorizer, tfidf_matrix = compute_tfidf(sentences)  # Compute TF-IDF matrix
    scores = tfidf_matrix.sum(axis=1).flatten()  # Calculate sum of TF-IDF scores for each sentence
    ranked_indices = nlargest(top_n, range(len(scores)), scores.take)  # Get indices of top-n sentences based on scores
    ranked_sentences = [original_sentences[i] for i in ranked_indices]  # Get top-n original sentences
    ranked_scores = [scores[i] for i in ranked_indices]  # Get scores corresponding to top-n sentences
    return ranked_sentences, ranked_scores, vectorizer  # Return top-n sentences, scores, and TF-IDF vectorizer

def summarize_with_tfidf(text, top_n=5):
    """Summarizes text using TF-IDF extractive summarization."""
    original_sentences, cleaned_sentences = preprocess_text(text)  # Preprocess text
    summary_sentences, _, vectorizer = rank_sentences_with_scores(cleaned_sentences, original_sentences, top_n)  # Get summary sentences
    return summary_sentences, vectorizer  # Return summary sentences and TF-IDF vectorizer

def textrank_summarize(text, ratio=0.3):
    """Summarizes text using TextRank extractive summarization."""
    summary = summarizer.summarize(text, ratio=ratio)  # Summarize text using TextRank
    return summary.split('\n')  # Split summary into sentences and return

def evaluate_with_rouge(generated_summary, expected_summary):
    """Evaluates generated summary against expected summary using ROUGE metrics."""
    rouge = Rouge()  # Initialize Rouge object
    scores = rouge.get_scores(generated_summary, expected_summary)  # Calculate ROUGE scores
    return scores  # Return ROUGE scores

def save_model(vectorizer, filename='tfidf_model.pkl'):
    """Saves TF-IDF vectorizer to a pickle file."""
    with open(filename, 'wb') as file:
        pickle.dump(vectorizer, file)  # Dump vectorizer object to file

def load_model(filename='tfidf_model.pkl'):
    """Loads TF-IDF vectorizer from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)  # Load vectorizer object from file

def save_similarity_matrix(matrix, filename='similarity_matrix.pkl'):
    """Saves similarity matrix to a pickle file."""
    with open(filename, 'wb') as file:
        pickle.dump(matrix, file)  # Dump similarity matrix object to file

def load_similarity_matrix(filename='similarity_matrix.pkl'):
    """Loads similarity matrix from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)  # Load similarity matrix object from file

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
