from transformers import AutoTokenizer
import re

def preprocess(text):
    # Lowercase and remove extra whitespace
    text = text.lower().strip()
    # Replace URLs with a placeholder
    text = re.sub(r"http\S+", "<URL>", text)
    return text

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "Check this website: https://example.com for info!"

preprocessed_text = preprocess(text)
tokens = tokenizer.tokenize(preprocessed_text)

print("Preprocessed tokens:", tokens)
