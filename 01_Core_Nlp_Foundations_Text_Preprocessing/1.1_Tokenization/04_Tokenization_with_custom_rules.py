import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.util import compile_infix_regex

# Load English pipeline
nlp = English()

# Customize tokenizer to keep contractions as single tokens
infix_re = compile_infix_regex(nlp.Defaults.infixes)
tokenizer = Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)

text = "Don't split contractions! But split other punctuation."
doc = tokenizer(text)

print([token.text for token in doc])
