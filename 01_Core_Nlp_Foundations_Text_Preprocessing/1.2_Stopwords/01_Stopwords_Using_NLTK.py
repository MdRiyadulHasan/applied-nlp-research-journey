import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Load English stopwords
stop_words = set(stopwords.words('english'))

text = "This is an advanced example to show how we can remove stopwords efficiently."

# Tokenize words
words = word_tokenize(text)

# Filter stopwords
filtered_words = [w for w in words if w.lower() not in stop_words]

print("Original Words:", words)
print("Filtered Words:", filtered_words)
