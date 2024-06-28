import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest
from rouge import Rouge
import pickle

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocesses text by tokenizing sentences, removing stopwords, and cleaning each sentence.
    Args:
    text (str): The text to be preprocessed.
    
    Returns:
    tuple: Original sentences and cleaned sentences.
    """
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
    """
    Computes TF-IDF scores for sentences.
    Args:
    sentences (list): List of cleaned sentences.
    
    Returns:
    tuple: TF-IDF vectorizer and matrix.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=0.1, ngram_range=(1, 2))
    X = vectorizer.fit_transform(sentences)  # Fit and transform sentences to TF-IDF matrix
    return vectorizer, X  # Return TF-IDF vectorizer and matrix

def rank_sentences_with_scores(sentences, original_sentences, top_n=5):
    """
    Ranks sentences based on TF-IDF scores.
    Args:
    sentences (list): List of cleaned sentences.
    original_sentences (list): List of original sentences.
    top_n (int): Number of top sentences to select.
    
    Returns:
    tuple: Top-n sentences, scores, and TF-IDF vectorizer.
    """
    vectorizer, tfidf_matrix = compute_tfidf(sentences)  # Compute TF-IDF matrix
    scores = tfidf_matrix.sum(axis=1).flatten()  # Calculate sum of TF-IDF scores for each sentence
    ranked_indices = nlargest(top_n, range(len(scores)), scores.take)  # Get indices of top-n sentences based on scores
    ranked_indices = sorted(ranked_indices)  # Ensure the selected sentences maintain the original order
    ranked_sentences = [original_sentences[i] for i in ranked_indices]  # Get top-n original sentences
    ranked_scores = [scores[i] for i in ranked_indices]  # Get scores corresponding to top-n sentences
    return ranked_sentences, ranked_scores, vectorizer  # Return top-n sentences, scores, and TF-IDF vectorizer

def summarize_with_tfidf(text, top_n=5):
    """
    Summarizes text using TF-IDF extractive summarization.
    Args:
    text (str): The text to be summarized.
    top_n (int): Number of top sentences to select.
    
    Returns:
    tuple: Summary sentences and TF-IDF vectorizer.
    """
    original_sentences, cleaned_sentences = preprocess_text(text)  # Preprocess text
    summary_sentences, _, vectorizer = rank_sentences_with_scores(cleaned_sentences, original_sentences, top_n)  # Get summary sentences
    return summary_sentences, vectorizer  # Return summary sentences and TF-IDF vectorizer

def textrank_summarize(text, ratio=0.3):
    """
    Summarizes text using TextRank extractive summarization.
    Args:
    text (str): The text to be summarized.
    ratio (float): Ratio of sentences to include in the summary.
    
    Returns:
    list: Summary sentences.
    """
    summary = summarizer.summarize(text, ratio=ratio)  # Summarize text using TextRank
    return summary.split('\n')  # Split summary into sentences and return

def evaluate_with_rouge(generated_summary, expected_summary):
    """
    Evaluates generated summary against expected summary using ROUGE metrics.
    Args:
    generated_summary (str): The generated summary.
    expected_summary (str): The expected summary.
    
    Returns:
    dict: ROUGE scores.
    """
    rouge = Rouge()  # Initialize Rouge object
    scores = rouge.get_scores(generated_summary, expected_summary)  # Calculate ROUGE scores
    return scores  # Return ROUGE scores

def save_model(vectorizer, filename='tfidf_model.pkl'):
    """
    Saves TF-IDF vectorizer to a pickle file.
    Args:
    vectorizer (TfidfVectorizer): The TF-IDF vectorizer.
    filename (str): The filename for the pickle file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(vectorizer, file)  # Dump vectorizer object to file

def load_model(filename='tfidf_model.pkl'):
    """
    Loads TF-IDF vectorizer from a pickle file.
    Args:
    filename (str): The filename of the pickle file.
    
    Returns:
    TfidfVectorizer: The loaded TF-IDF vectorizer.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)  # Load vectorizer object from file

def save_similarity_matrix(matrix, filename='similarity_matrix.pkl'):
    """
    Saves similarity matrix to a pickle file.
    Args:
    matrix (np.array): The similarity matrix.
    filename (str): The filename for the pickle file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(matrix, file)  # Dump similarity matrix object to file

def load_similarity_matrix(filename='similarity_matrix.pkl'):
    """
    Loads similarity matrix from a pickle file.
    Args:
    filename (str): The filename of the pickle file.
    
    Returns:
    np.array: The loaded similarity matrix.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)  # Load similarity matrix object from file

# Example usage
if __name__ == "__main__":
    text = """
    Beyond learning paths and certification programs, Red Hat offers several resources and support mechanisms to assist professionals in their careers:

    Community Engagement: Red Hat fosters a vibrant community of developers, administrators, and users through forums, mailing lists, and events. Engaging with this community provides opportunities for networking, knowledge sharing, and collaboration on open-source projects.

    Technical Support: Red Hat provides comprehensive technical support services to customers and subscribers. This support includes assistance with troubleshooting, performance optimization, and guidance on best practices for deploying and managing Red Hat products and solutions.

    Training and Workshops: In addition to formal learning paths, Red Hat offers workshops, webinars, and training sessions on various topics related to open-source technologies, DevOps practices, and emerging trends in the IT industry. These sessions provide hands-on experience and practical insights that can enhance your skills and expertise.

    Career Resources: Red Hat may offer career resources such as job boards, career fairs, and networking events specifically tailored to professionals working with open-source technologies. These resources can help you explore job opportunities, connect with potential employers, and advance your career in the field.

    Partnerships and Ecosystem: Red Hat collaborates with a broad ecosystem of technology partners, system integrators, and service providers. Leveraging these partnerships can offer access to additional resources, tools, and expertise that complement Red Hat's offerings and support your professional development goals.

    Overall, Red Hat aims to support professionals in their professional life by providing a holistic ecosystem of resources, support services, and opportunities for learning, growth, and advancement within the open-source community and the broader IT industry.
    """

    # Generate TF-IDF summary
    tfidf_summary, vectorizer = summarize_with_tfidf(text, top_n=5)
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
    expected_summary = "Red Hat offers several resources and support mechanisms to assist professionals in their careers."
    tfidf_scores = evaluate_with_rouge(" ".join(tfidf_summary), expected_summary)
    textrank_scores = evaluate_with_rouge(" ".join(textrank_summary), expected_summary)

    print("\nROUGE Evaluation:")
    print("TF-IDF Scores:", tfidf_scores)
    print("TextRank Scores:", textrank_scores)
