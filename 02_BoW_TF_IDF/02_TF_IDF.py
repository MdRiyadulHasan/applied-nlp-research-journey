from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "I love NLP",
    "NLP is fun",
    "I love fun activities"
]

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("TF-IDF matrix:\n", X_tfidf.toarray())
