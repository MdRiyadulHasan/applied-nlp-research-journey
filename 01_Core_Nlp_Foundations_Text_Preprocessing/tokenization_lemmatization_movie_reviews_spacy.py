import spacy
from nltk.corpus import movie_reviews
import nltk
nltk.download('movie_reviews')

# Load English model
nlp = spacy.load("en_core_web_sm")

# Stopwords from spaCy
stopwords_spacy = nlp.Defaults.stop_words

# Function to preprocess review
def preprocess_review_spacy(review_text):
    doc = nlp(review_text.lower())
    tokens = [token.lemma_ for token in doc
              if token.is_alpha and token.text not in stopwords_spacy]
    return tokens

# Example: process first 3 movie reviews
for fileid in movie_reviews.fileids()[:3]:
    text = movie_reviews.raw(fileid)
    lemmas = preprocess_review_spacy(text)
    print(f"File: {fileid}")
    print("Lemmatized tokens:", lemmas[:20], "...")
    print("-"*50)
