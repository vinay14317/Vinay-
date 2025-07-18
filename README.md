from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Encode input prompt
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
output = model.generate(
    input_ids,
    max_length=50,         # Length of generated sequence
    num_return_sequences=1, # Number of sequences to generate
    no_repeat_ngram_size=2, # No repeating n-grams
    do_sample=True,         # Use sampling instead of greedy decoding
    top_k=50,               # Top K sampling
    top_p=0.95,             # Nucleus sampling
    temperature=0.9         # Sampling temperature
)

# Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)# Vinay-
