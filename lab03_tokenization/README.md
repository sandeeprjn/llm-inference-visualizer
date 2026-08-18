# Lab 03 — Tokenization

## Objective

Understand how an LLM converts human-readable text into tokens and token IDs before the text enters the neural network.

This lab uses the TinyLlama tokenizer to inspect:

- Tokenization
- Vocabulary
- Token IDs
- Special tokens
- Subword tokenization
- `tokenizer.json`
- The relationship between token IDs and the embedding matrix

---

## Model

This lab uses:

**TinyLlama/TinyLlama-1.1B-Chat-v1.0**

The tokenizer has a vocabulary of approximately **32,000 tokens**.

---

## What We Learned

The basic pipeline is:

```text
Human Text
    ↓
Tokenizer
    ↓
Tokens
    ↓
Token IDs
    ↓
Embedding Matrix
    ↓
Hidden Representation


1. What is a Tokenizer?

A tokenizer converts human-readable text into smaller pieces called tokens.

A token does not necessarily correspond to an entire word.

A token can be:

A complete word
Part of a word
Punctuation
A special token
A byte or character representation

For example:

bank

may be represented by a single token.

An unfamiliar word may instead be broken into multiple smaller pieces.

2. Vocabulary

A tokenizer has a vocabulary containing the tokens that it knows how to represent.

The vocabulary maps:

Token → Token ID

For example, part of the TinyLlama vocabulary contains entries such as:

<unk> → 0
<s>   → 1
</s>  → 2
...
▁bank → 9124
bank  → 9157

The TinyLlama tokenizer has approximately:

32,000 tokens

This is directly related to the model's embedding matrix:

Vocabulary Size = 32,000


Embedding Matrix = 32,000 × 2,048

Therefore:

One Token ID
      ↓
One Row in the Embedding Matrix
      ↓
One 2048-dimensional embedding vector
3. Token IDs

A token ID is simply an integer representing a token in the vocabulary.

For example:

▁bank → 9124

When the tokenizer produces token ID 9124, the model can use that ID to retrieve the corresponding row from the embedding matrix.

Conceptually:

"bank"
   ↓
Tokenizer
   ↓
▁bank
   ↓
9124
   ↓
Embedding Matrix[9124]
   ↓
2048 numbers

The token ID itself does not contain the meaning of the word.

It is simply an index into the model's learned parameters.

4. Special Tokens

TinyLlama defines special tokens such as:

<unk> → 0
<s>   → 1
</s>  → 2

These are different from normal text tokens.

<unk>

<unk> means unknown token.

It is used when text cannot be represented using the normal vocabulary/tokenization mechanism.

Unknown input
      ↓
<unk>
      ↓
Token ID 0
<s>

<s> represents the Beginning of Sequence (BOS).

Conceptually:

<s> The bank approved the loan.

It indicates the beginning of a sequence.

</s>

</s> represents the End of Sequence (EOS).

Conceptually:

The bank approved the loan. </s>

It indicates the end of a sequence.

Important

A special token being present in the vocabulary does not necessarily mean the tokenizer automatically inserts it into every sequence.

For example:

tokenizer.encode(
    text,
    add_special_tokens=False
)

explicitly prevents automatic addition of special tokens.

Whether special tokens are automatically added depends on the tokenizer configuration.

5. Subword Tokenization

The vocabulary does not need to contain every possible word.

For example, the vocabulary contains:

▁bank → 9124
bank  → 9157

But an arbitrary string such as:

koo

may not exist as a single vocabulary entry.

The tokenizer can represent it using smaller pieces that are present in the vocabulary.

For example, it may produce something similar to:

koo
 ↓
▁k + oo

The exact result depends on the tokenizer's vocabulary and tokenization algorithm.

This is one of the reasons modern LLMs can handle words or strings that do not exist as individual vocabulary entries.

6. Vocabulary + Tokenization Algorithm

Tokenization is not simply:

Word → Search vocabulary → Token

The tokenizer uses both:

A vocabulary containing available token pieces
A tokenization algorithm/model that determines how the input should be segmented

Conceptually:

                 RAW TEXT
                    ↓
          Tokenization Algorithm
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
      Vocabulary          Rules/Model
   "What pieces exist?"  "How should
                          they be chosen?"
          └─────────┬─────────┘
                    ↓
               Token Pieces
                    ↓
                Token IDs

The tokenizer does not use semantic similarity to find the "closest" word.

For example, if koo is not a token, it does not search for a word that is semantically similar to koo.

Instead, it finds a valid segmentation using the tokenizer's learned vocabulary/model.

7. Inspecting tokenizer.json

The tokenizer files are downloaded into the Hugging Face cache.

For TinyLlama, the files are located under:

~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/

The tokenizer snapshot contains files such as:

tokenizer.json
tokenizer.model
tokenizer_config.json
special_tokens_map.json

The main tokenizer file can be inspected using:

more ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/<snapshot-id>/tokenizer.json
8. Finding Tokens in the Vocabulary

The vocabulary is contained inside the tokenizer JSON.

Conceptually:

tokenizer.json
    │
    └── model
          │
          └── vocab
                │
                ├── "<unk>" → 0
                ├── "<s>" → 1
                ├── "</s>" → 2
                ├── "▁bank" → 9124
                └── ...

You can search for a token using Linux:

grep bank tokenizer.json

For example:

"▁bank": 9124
"bank": 9157

You can also test an arbitrary string:

grep koo tokenizer.json

If nothing is returned, it means koo does not appear as a vocabulary entry in that location.

However, that does not mean the tokenizer cannot process koo.

It can potentially represent it using smaller vocabulary pieces.

9. Inspecting the Vocabulary Using Python

The vocabulary can also be loaded directly from the JSON file.

Example:

import json


path = "/path/to/tokenizer.json"


with open(path, "r") as f:
    tokenizer_data = json.load(f)


vocab = tokenizer_data["model"]["vocab"]


print("Vocabulary size:", len(vocab))


for token, token_id in vocab.items():
    print(f"{token_id:6}  {token}")

This allows us to inspect the vocabulary programmatically instead of using Linux commands.

10. Searching the Vocabulary Using Python

We can search for specific tokens:

for token, token_id in vocab.items():
    if "bank" in token:
        print(token_id, token)

This can return entries such as:

9124 ▁bank
9157 bank
24388 ▁banks

This demonstrates that the vocabulary can contain multiple related token pieces.

11. Running the Lab

Make sure the Python virtual environment is activated:

source venv/bin/activate

From the repository root:

cd ~/llm-inference-visualizer

Run the tokenizer:

python lab03_tokenization/app.py

Run the vocabulary inspection program:

python lab03_tokenization/show_vocab.py
12. Tokenizer vs. LLM

The tokenizer and the LLM are separate components.

The tokenizer is responsible for converting text into token IDs.

The neural network then uses those token IDs to retrieve embeddings and process the sequence.

                    TOKENIZER
                        │
                        ↓
Human Text ────────→ Token IDs
                        │
                        ↓
                Embedding Matrix
                  32,000 × 2,048
                        │
                        ↓
               Hidden Representation
                        │
                        ↓
                Transformer Layers
                        │
                        ↓
                   Prediction

The tokenizer does not contain the model's neural-network weights.

The model weights are stored separately in files such as:

model.safetensors
13. Connection to the Embedding Matrix

This is the most important connection from Lab 03.

Suppose the tokenizer produces:

bank → 9124

The model can use that ID as an index:

Embedding Matrix[9124]

The result is one learned vector:

[2048 numbers]

So:

Text
 ↓
Token
 ↓
Token ID
 ↓
Embedding Matrix Row
 ↓
Embedding Vector
 ↓
Hidden Representation

This is the bridge between Tokenization and Embeddings.

14. What We Learned

By completing this lab, we learned:

A token is not necessarily a word.
Tokenization converts text into token pieces.
The tokenizer has a fixed vocabulary.
Each vocabulary token has a token ID.
TinyLlama has approximately 32,000 vocabulary entries.
Special tokens such as <unk>, <s>, and </s> have specific IDs.
A word does not need to exist as one complete vocabulary token.
Unknown strings can be represented using smaller token pieces.
Token IDs are used to index the model's embedding matrix.
The tokenizer and the LLM are separate components.
AI Infrastructure / Product Insight

Tokenization has direct implications for AI infrastructure and product design.

More tokens generally mean:

More computation
More memory usage
Potentially higher latency
Greater context-window consumption
Potentially higher inference cost

Therefore, understanding tokens is important when designing and evaluating LLM products and inference platforms.

Tokenization is not just a preprocessing detail.

It directly affects the economics and performance of an AI system.

Key Takeaway

A token is not necessarily a word. The tokenizer converts text into vocabulary pieces, assigns each piece a token ID, and those IDs are subsequently used to retrieve vectors from the model's learned embedding matrix.

The overall flow is:

Human Text
    ↓
Tokenization
    ↓
Token Pieces
    ↓
Token IDs
    ↓
Embedding Lookup
    ↓
Embedding Vectors
    ↓
Hidden Representations
    ↓
Transformer Layers
    ↓
Model Output