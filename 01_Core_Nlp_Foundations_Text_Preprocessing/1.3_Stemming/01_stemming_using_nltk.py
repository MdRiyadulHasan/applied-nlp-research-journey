import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize

# Download punkt tokenizer
nltk.download('punkt')

text = "The cats were running and jumping around playfully in the gardens. They were playing football beautifully"

# Tokenize words
words = word_tokenize(text)
# Initialize stemmers
porter = PorterStemmer()
snowball = SnowballStemmer("english")
lancaster = LancasterStemmer()

# Compare outputs
print("{:<15} {:<15} {:<15} {:<15}".format("Original", "Porter", "Snowball", "Lancaster"))
print("-" * 60)
for word in words:
    print("{:<15} {:<15} {:<15} {:<15}".format(
        word,
        porter.stem(word),
        snowball.stem(word),
        lancaster.stem(word)
    ))
