from transformers import AutoModelForCausalLM

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print("Model loaded successfully!")

print(f"Model name: {model.config.name_or_path}")
print(f"Model type: {model.config.model_type}")
print(f"Number of parameters: {model.num_parameters()}")
print(f"Model architecture: {model.config.architectures}")
print(f"Model config: {model.config}")
print(f"Number of layers: {model.config.num_hidden_layers}")
print(f"Length of layers: {len(model.model.layers)}")