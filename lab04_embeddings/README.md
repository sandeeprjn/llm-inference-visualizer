# Lab 04 — Embeddings

## Objective

Understand how a token ID is converted into a numerical vector that can be processed by the Transformer.

In this lab, we inspect the actual embedding matrix of TinyLlama and retrieve the embedding vector corresponding to a specific token.

---

## Model

This lab uses:

**TinyLlama/TinyLlama-1.1B-Chat-v1.0**

The model has:

```text
Vocabulary Size = 32,000
Hidden Size = 2,048

Therefore, the input embedding matrix has the shape:

32,000 × 2,048

1. From Token to Embedding

In Lab 03, we learned that the tokenizer converts text into token IDs.

For example:

"The bank approved the loan."
          ↓
       Tokenizer
          ↓
      Token IDs

Each token ID is an integer that identifies a token in the vocabulary.

The model then uses that ID to retrieve a row from the embedding matrix.

Token
   ↓
Token ID
   ↓
Embedding Matrix
   ↓
Corresponding Row
   ↓
Embedding Vector
2. The Embedding Matrix

The model contains a learned embedding matrix.

For TinyLlama:

Embedding Matrix
       ↓
32,000 rows × 2,048 columns

Each row corresponds to one token in the vocabulary.

Conceptually:

                2048 dimensions
                     ↓

Token 0      [.........................]
Token 1      [.........................]
Token 2      [.........................]
Token 3      [.........................]
   ...
Token 9124   [.........................]  ← "▁bank"
   ...
Token 31999  [.........................]

Therefore:

32,000 rows
    =
32,000 vocabulary entries

and:

2,048 columns
    =
2,048-dimensional embedding vector
3. Embedding Lookup

Suppose the tokenizer produces:

▁bank → 9124

The model uses the token ID as an index:

embedding_matrix[9124]

This retrieves exactly one row.

The result is:

[2048 numbers]

This is the embedding vector for that token.

Conceptually:

"bank"
   ↓
Token ID = 9124
   ↓
Embedding Matrix[9124]
   ↓
[2048 learned values]
4. Embeddings Are Learned Weights

The values in the embedding vector are not manually assigned.

They are learned during model training.

For example, conceptually:

bank
 ↓
[0.21, -0.43, 0.87, 0.12, ...]

The model learned these values during pre-training.

The individual dimensions should not be interpreted as simple human-readable features such as:

Dimension 1 = gender
Dimension 2 = plural
Dimension 3 = financial

Instead, meaning is represented in a distributed way across many dimensions.

5. Loading the Embedding Layer

The embedding layer can be accessed from the Hugging Face model:

embedding_layer = model.get_input_embeddings()

The actual learned weights can then be accessed using:

embedding_matrix = embedding_layer.weight

The shape can be inspected using:

print(embedding_matrix.shape)

For TinyLlama, this should produce:

torch.Size([32000, 2048])
6. Retrieving One Embedding Vector

After tokenizing the input:

inputs = tokenizer(
    text,
    return_tensors="pt",
    add_special_tokens=False
)

we can retrieve the token IDs:

input_ids = inputs["input_ids"]

For example:

input_ids
    ↓
[...., 9124, ....]

We can retrieve one token ID:

token_id = input_ids[0, 0].item()

Then retrieve its embedding:

embedding_vector = embedding_matrix[token_id]

The vector shape is:

torch.Size([2048])
7. Complete Experiment

The main experiment in this lab is:

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Model loaded!\n")

text = "The bank approved the loan."

inputs = tokenizer(
    text,
    return_tensors="pt",
    add_special_tokens=False
)

input_ids = inputs["input_ids"]

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
8. Understanding the Shapes

If the input contains 6 tokens:

input_ids.shape

will be:

[1, 6]

where:

1 = number of input sequences
6 = number of tokens

The embedding matrix is:

[32000, 2048]

After looking up the embeddings for all 6 tokens, the representation becomes conceptually:

[1, 6, 2048]

Meaning:

1 sequence
6 tokens
2048 values per token
9. The Complete Flow

This is the key concept from Lab 04:

Human Text
     ↓
"The bank approved the loan."
     ↓
Tokenizer
     ↓
Token IDs
     ↓
[ID1, ID2, ID3, ID4, ...]
     ↓
Embedding Matrix
[32000 × 2048]
     ↓
Embedding Lookup
     ↓
One 2048-dimensional vector per token
     ↓
Hidden Representation
[sequence_length × 2048]
10. Token ID vs Embedding Vector

It is important not to confuse these two.

A token ID is simply an integer:

9124

It is an index.

The embedding vector is a learned numerical representation:

[0.21, -0.43, 0.87, ...]

with 2,048 values for TinyLlama.

So:

Token ID
    =
Index

Embedding Vector
    =
Learned Representation
11. Connection to the Transformer

The embedding vector becomes the initial representation that enters the Transformer.

Conceptually:

Token IDs
    ↓
Embedding Lookup
    ↓
X₀
    ↓
Transformer Layer 0
    ↓
X₁
    ↓
Transformer Layer 1
    ↓
X₂
    ↓
...

The embedding output is therefore the starting hidden representation of the input tokens.

Later Transformer layers transform these representations further.

12. Important Insight

The embedding matrix is part of the pre-trained model's learned weights.

It is not generated dynamically when we run inference.

The model already contains:

Embedding Matrix
32,000 × 2,048

When we provide a token ID, inference simply retrieves the corresponding row.

Token ID
   ↓
Row lookup
   ↓
Pre-trained embedding vector
13. Running the Lab

Make sure the virtual environment is activated:

source venv/bin/activate

From the repository root:

cd ~/llm-inference-visualizer

Run:

python lab04_embeddings/app.py
14. What We Learned

By completing this lab, we learned:

The embedding matrix is part of the model's learned weights.
TinyLlama's embedding matrix has 32,000 rows.
Each row corresponds to a vocabulary token.
Each row contains 2,048 learned values.
A token ID acts as an index into the embedding matrix.
One token ID retrieves one 2,048-dimensional embedding vector.
Token IDs and embedding vectors are completely different things.
The embedding vectors become the initial representations processed by the Transformer.
Key Takeaway

A token ID is an index. The embedding matrix uses that index to retrieve a learned vector representing that token.

The fundamental operation is:

Token ID
   ↓
Embedding Matrix[Token ID]
   ↓
Embedding Vector

For TinyLlama:

Token ID
   ↓
32,000 × 2,048 Embedding Matrix
   ↓
One row
   ↓
2,048-dimensional vector
Next Lab

Lab 05 — Forward Pass

We will take the embedding representation and follow it through the Transformer.

The goal will be to understand:

Embedding
    ↓
Transformer Layer
    ↓
Attention
    ↓
MLP
    ↓
Next Hidden Representation
    ↓
Next Transformer Layer

This is where we start following the actual inference path through the model.