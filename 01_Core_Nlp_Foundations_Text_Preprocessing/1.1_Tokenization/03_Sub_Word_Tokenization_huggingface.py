from transformers import AutoTokenizer

# Load a pretrained tokenizer (BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Tokenization is crucial in NLP! Let's see how it works."

# Encode text into tokens (subword tokenization)
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# Convert tokens to IDs
token_ids = tokenizer.encode(text, add_special_tokens=True)
print("Token IDs:", token_ids)

# Decode back to text
decoded_text = tokenizer.decode(token_ids)
print("Decoded text:", decoded_text)
