import spacy
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

text = "Stemming reduces words to their base or root form."

doc = nlp(text)

# Add stemming result
for token in doc:
    print(f"{token.text:<15} Lemma: {token.lemma_:<15} Stem: {stemmer.stem(token.text)}")
