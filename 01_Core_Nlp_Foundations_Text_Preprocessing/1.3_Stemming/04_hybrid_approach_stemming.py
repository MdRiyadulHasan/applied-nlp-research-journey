import spacy
from nltk.stem import SnowballStemmer

nlp = spacy.load("en_core_web_sm")
stemmer = SnowballStemmer("english")

def hybrid_stem(word):
    doc = nlp(word)
    lemma = doc[0].lemma_
    # If lemma = original (not found in vocab), fall back to stem
    if lemma == word:
        return stemmer.stem(word)
    return lemma

words = ["running", "better", "cats", "machines", "computationally"]

results = {word: hybrid_stem(word) for word in words}
print(results)
