from collections import Counter
import nltk
nltk.download('punkt')

text = """
Data science is an interdisciplinary field that uses scientific methods,
processes, algorithms and systems to extract knowledge and insights from noisy,
structured and unstructured data. It is also an important field. Science and Technology extract knowledge.
It is also useful to learn science and technology
"""

words = [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]
word_freq = Counter(words)

# Define frequency-based stopwords (appearing more than once)
freq_stopwords = {word for word, freq in word_freq.items() if freq > 1}

filtered = [w for w in words if w not in freq_stopwords]
print("Filtered (frequency-based):", filtered)
