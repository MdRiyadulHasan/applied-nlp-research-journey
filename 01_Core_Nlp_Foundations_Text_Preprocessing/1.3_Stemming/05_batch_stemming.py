from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")

def batch_stem(words):
    return [stemmer.stem(w) for w in words]

tokens = ["running", "jumping", "flying", "studies", "wolves"] * 10000
stems = batch_stem(tokens)
print(stems[:10])
