import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nlp = spacy.load("en_core_web_sm")
nltk_lemmatizer = WordNetLemmatizer()

def hybrid_lemmatize(word, pos):
    # Try spaCy lemma first
    doc = nlp(word)
    lemma_spacy = doc[0].lemma_
    if lemma_spacy != word:
        return lemma_spacy
    # fallback to NLTK WordNet
    pos_map = {
        'NOUN': wordnet.NOUN,
        'VERB': wordnet.VERB,
        'ADJ': wordnet.ADJ,
        'ADV': wordnet.ADV
    }
    return nltk_lemmatizer.lemmatize(word, pos_map.get(pos, wordnet.NOUN))

text = "The striped bats were flying and better opportunities awaited them."
doc = nlp(text)

print(f"{'Word':<15}{'POS':<10}{'Lemma'}")
print("-"*40)

for token in doc:
    print(f"{token.text:<15}{token.pos_:<10}{hybrid_lemmatize(token.text, token.pos_)}")
