import nltk
from nltk.corpus import stopwords, movie_reviews
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

# Download necessary NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Helper function to map POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Function to process a single review
def preprocess_review(review_text):
    # Tokenize
    tokens = word_tokenize(review_text)
    # Remove stopwords & punctuation
    tokens = [t.lower() for t in tokens if t.isalpha() and t.lower() not in stop_words]
    # POS tagging
    pos_tags = pos_tag(tokens)
    # Lemmatize
    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    return lemmas

# Example: process first 5 movie reviews
for fileid in movie_reviews.fileids()[:5]:
    text = movie_reviews.raw(fileid)
    lemmas = preprocess_review(text)
    print(f"File: {fileid}")
    print("Lemmatized tokens:", lemmas[:20], "...")  # first 20 tokens
    print("-"*50)
