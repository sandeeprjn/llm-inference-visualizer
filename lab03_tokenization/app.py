from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

text = "The bank approved the loan."

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(
    text,
    add_special_tokens=False
)

print(tokens)
print(token_ids)