import spacy

# Load small English model
nlp = spacy.load("en_core_web_sm")

text = "This is an advanced example to show how we can remove stopwords efficiently."
doc = nlp(text)

filtered_tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]

print("Filtered Words:", filtered_tokens)
