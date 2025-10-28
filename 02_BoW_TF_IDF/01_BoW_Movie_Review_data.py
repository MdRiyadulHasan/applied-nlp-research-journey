import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Download NLTK data
nltk.download('movie_reviews')

# Load movie reviews
documents = []
labels = []

for fileid in movie_reviews.fileids():
    documents.append(movie_reviews.raw(fileid))
    labels.append(movie_reviews.categories(fileid)[0])  # 'pos' or 'neg'

print(f"Total reviews: {len(documents)}")  # 2000 reviews

# ---------------------------
# 1️⃣ Bag-of-Words (CountVectorizer)
# ---------------------------
vectorizer = CountVectorizer(stop_words='english', max_features=20)  # top 20 words for demo
X_bow = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW matrix shape:", X_bow.shape)

# Convert to DataFrame for easy viewing
df_bow = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())
print(df_bow.head())

# ---------------------------
# 2️⃣ TF-IDF representation
# ---------------------------
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
X_tfidf = tfidf_vectorizer.fit_transform(documents)

df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF matrix (first 5 reviews):")
print(df_tfidf.head())
