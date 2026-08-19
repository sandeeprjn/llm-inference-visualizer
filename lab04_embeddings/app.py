import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Model loaded!\n")

text = "The bank approved the loan."
    

inputs = tokenizer(text, return_tensors="pt", padding=True)

# Convert text to token IDs
inputs = tokenizer(
    text,
    padding=True,
    return_tensors="pt",
    add_special_tokens=False
)

input_ids = inputs["input_ids"]

print(tokenizer.tokenize(text))

print("Text:")
print(text)

print("\nToken IDs:")
print(input_ids)

print("\nShape of token IDs:")
print(input_ids.shape)

# Get the embedding layer
embedding_layer = model.get_input_embeddings()

print("\nEmbedding layer:")
print(embedding_layer)

# Get the embedding matrix
embedding_matrix = embedding_layer.weight

print("\nEmbedding matrix shape:")
print(embedding_matrix.shape)

# Get the first token ID
token_id = input_ids[0, 0].item()

print("\nFirst token ID:")
print(token_id)

# Get the corresponding embedding vector
embedding_vector = embedding_matrix[token_id]

print("\nEmbedding vector shape:")
print(embedding_vector.shape)

print("\nFirst 10 values of the embedding vector:")
print(embedding_vector[:10])