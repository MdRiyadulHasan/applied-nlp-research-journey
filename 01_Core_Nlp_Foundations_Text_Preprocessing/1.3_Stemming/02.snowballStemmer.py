from nltk.stem.snowball import SnowballStemmer

languages = ["english", "french", "german", "spanish"]
words = ["running", "manger", "spielen", "hablando"]

for lang, word in zip(languages, words):
    stemmer = SnowballStemmer(lang)
    print(f"{lang.title()} Stem of '{word}': {stemmer.stem(word)}")
