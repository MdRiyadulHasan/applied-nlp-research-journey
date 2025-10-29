# ============================================
# 📚 SMS Spam Detection using TF-IDF + ML
# Author: Your Name
# University Project Submission
# ============================================

# 1️⃣ Import Required Libraries
import pandas as pd
import numpy as np
import string
import nltk
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords

# ----------------------------------------------------------
# 2️⃣ Load Dataset
# ----------------------------------------------------------
# Dataset: SMS Spam Collection (UCI ML Repository)
# Format: label \t message
df = pd.read_csv(
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    sep='\t',
    header=None,
    names=['label', 'message']
)

print("✅ Dataset Loaded Successfully")
print(df.head())
print("\nDataset Info:\n", df.info())

# ----------------------------------------------------------
# 3️⃣ Data Cleaning & Preprocessing
# ----------------------------------------------------------

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs, numbers, punctuation, extra spaces
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_message'] = df['message'].apply(clean_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['clean_message'] = df['clean_message'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

print("\nSample cleaned messages:\n", df[['message', 'clean_message']].head())

# ----------------------------------------------------------
# 4️⃣ Encode Labels
# ----------------------------------------------------------
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# ----------------------------------------------------------
# 5️⃣ Train-Test Split
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)

print(f"\nTraining Samples: {len(X_train)}, Test Samples: {len(X_test)}")

# ----------------------------------------------------------
# 6️⃣ Feature Extraction with TF-IDF
# ----------------------------------------------------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF Shape (Train):", X_train_tfidf.shape)

# ----------------------------------------------------------
# 7️⃣ Model Training
# ----------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ----------------------------------------------------------
# 8️⃣ Evaluation
# ----------------------------------------------------------
y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc:.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ----------------------------------------------------------
# 9️⃣ Test on New Messages
# ----------------------------------------------------------
test_samples = [
    "Win a FREE iPhone now! Click the link!",
    "Hey, are we still on for dinner tonight?",
    "Urgent! Your account has been compromised. Reset password here."
]

test_tfidf = tfidf.transform(test_samples)
predictions = model.predict(test_tfidf)

for msg, label in zip(test_samples, predictions):
    print(f"\nMessage: {msg}\n→ Prediction: {'Spam' if label==1 else 'Ham'}")

# ----------------------------------------------------------
# 🔟 Save Model and Vectorizer (for deployment)
# ----------------------------------------------------------
import joblib

joblib.dump(model, "sms_spam_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\n💾 Model and Vectorizer saved successfully!")
