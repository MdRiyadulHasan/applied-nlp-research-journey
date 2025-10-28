import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from contractions import fix  # pip install contractions

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

def normalize_text(text):
    print(f"\n🔹 Original Text:\n{text}\n")

    # 1️⃣ Lowercase
    text = text.lower()

    # 2️⃣ Expand contractions (e.g., "I'm" → "I am")
    text = fix(text)

    # 3️⃣ Remove URLs, HTML tags, mentions, and numbers
    text = re.sub(r"http\S+|www\S+|<.*?>|@\w+|\d+", " ", text)

    # 4️⃣ Remove punctuation and non-alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5️⃣ Tokenize
    tokens = word_tokenize(text)

    # 6️⃣ Remove stopwords
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    # 7️⃣ Lemmatization using spaCy (context-aware)
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc if token.lemma_ not in stop_words]

    # 8️⃣ Join cleaned tokens back
    normalized_text = " ".join(lemmas)

    print(f"✅ Normalized Text:\n{normalized_text}\n")
    return normalized_text


# =============================
# 🧪 Example
# =============================
sample_text = "I'm LOVING the new AI tools!!! Visit https://openai.com 😎 #awesome"
cleaned = normalize_text(sample_text)
