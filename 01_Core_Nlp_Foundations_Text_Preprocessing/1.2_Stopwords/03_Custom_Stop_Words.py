import nltk
import spacy
from nltk.corpus import stopwords

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

# Combine stopwords from NLTK and spaCy
stop_words = set(stopwords.words('english')).union(nlp.Defaults.stop_words)

# Add domain-specific stopwords
custom_stopwords = {"example", "efficiently", "tokenize"}
stop_words = stop_words.union(custom_stopwords)

text = "This is an advanced example to show how we can remove stopwords efficiently."

# Tokenize with spaCy
doc = nlp(text)

filtered_tokens = [token.text for token in doc if token.text.lower() not in stop_words and token.is_alpha]

print("Final Tokens (after removing all stopwords):", filtered_tokens)
