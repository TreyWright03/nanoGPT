import os
import torch
import tiktoken
from model import GPTConfig, GPT

# Configuration
checkpoint_path = "out/ckpt.pt" 
start_prompt = "Write a function that determines whether a string is a palindrome."  # Replace with your desired starting text
num_samples = 5  # Number of samples to generate
max_new_tokens = 100  
temperature = .8  
top_k = 50  
device = 'cuda' if torch.cuda.is_available() else 'cpu'


torch.manual_seed(42)


checkpoint = torch.load(checkpoint_path, map_location=device)
model_args = checkpoint.get('model_args', {
    "vocab_size": 50257,
    "block_size": 1024,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
})
model = GPT(GPTConfig(**model_args))


state_dict = checkpoint['model']
prefix = '_orig_mod.'
state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
model.load_state_dict(state_dict)


model.to(device)
model.eval()


enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

start_ids = encode(start_prompt)
x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    for i in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(f"Sample {i + 1}:")
        print(decode(y[0].tolist()))
        print("-" * 20)

