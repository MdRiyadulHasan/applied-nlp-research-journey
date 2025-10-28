# =====================================
# NLP Preprocessing Pipeline
# Tokenization → Stopword Removal → Stemming → Lemmatization → Cleaned Output
# =====================================

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')  # newer NLTK versions need this

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Helper: Convert POS tag from nltk → wordnet
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

# Main function for text preprocessing
def preprocess_text(text):
    print(f"\n🔹 Original Text:\n{text}\n")

    # 1️⃣ Tokenization
    tokens = word_tokenize(text)
    print(f"✅ Tokens:\n{tokens}\n")

    # 2️⃣ Lowercasing & Stopword Removal
    filtered_tokens = [w.lower() for w in tokens if w.lower() not in stop_words and w.isalpha()]
    print(f"✅ After Stopword Removal:\n{filtered_tokens}\n")

    # 3️⃣ Stemming
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    print(f"✅ After Stemming:\n{stemmed_tokens}\n")

    # 4️⃣ Lemmatization (POS-aware)
    pos_tags = pos_tag(filtered_tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in pos_tags]
    print(f"✅ After Lemmatization (NLTK):\n{lemmatized_tokens}\n")

    # 5️⃣ (Optional) spaCy Lemmatization (contextual)
    doc = nlp(" ".join(filtered_tokens))
    spacy_lemmas = [token.lemma_ for token in doc]
    print(f"✅ After Lemmatization (spaCy):\n{spacy_lemmas}\n")

    # 6️⃣ Final Cleaned Text
    cleaned_text = " ".join(spacy_lemmas)
    print(f"✨ Final Cleaned Text:\n{cleaned_text}\n")

    return cleaned_text


# =====================================
# 🧪 Test the Pipeline
# =====================================
sample_text = "The striped bats were hanging on their feet and eating better fruits quickly!"
processed = preprocess_text(sample_text)
