import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download necessary NLTK data files
nltk.download('punkt_tab')

text = "Hello there! NLP is amazing. Let's tokenize this text."

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)

# Word tokenization
words = word_tokenize(text)
print("Words:", words)
