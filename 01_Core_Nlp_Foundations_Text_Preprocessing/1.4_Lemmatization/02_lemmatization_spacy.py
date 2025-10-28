import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

text = "The striped bats are hanging on their feet for best"
doc = nlp(text)

print(f"{'Word':<15}{'POS':<10}{'Lemma'}")
print("-"*40)

for token in doc:
    print(f"{token.text:<15}{token.pos_:<10}{token.lemma_}")
