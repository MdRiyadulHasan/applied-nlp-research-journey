import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

text = "Hello there! NLP is amazing. Let's tokenize this text."

# Process text
doc = nlp(text)

# Sentence tokenization
sentences = list(doc.sents)
print("Sentences:", [sent.text for sent in sentences])

# Word tokenization
words = [token.text for token in doc]
print("Words:", words)
