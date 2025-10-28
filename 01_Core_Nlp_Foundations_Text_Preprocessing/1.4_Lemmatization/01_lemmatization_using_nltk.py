# Tokenization – Split text into words.

# POS Tagging – Identify each word’s Part-of-Speech (noun, verb, adjective, etc.).

# Dictionary Lookup – Use morphological rules and a linguistic lexicon (e.g., WordNet).

# Lemmatization – Return the base form corresponding to POS + meaning.

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

# Downloads
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

lemmatizer = WordNetLemmatizer()

# Helper to map POS tags
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
        return wordnet.NOUN  # default

text = "The striped bats are hanging on their feet for best"
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

print(f"{'Word':<15}{'POS':<10}{'Lemma'}")
print("-" * 40)

for word, tag in pos_tags:
    lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
    print(f"{word:<15}{tag:<10}{lemma}")
