import os
import requests
import tiktoken
import numpy as np

# Path to the dataset file (local or fetched)
input_file_path = os.path.join(os.path.dirname(__file__), 'PyCodeDataset.txt')

# Download the Python code dataset if it doesn't exist
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/TreyWright03/PyDataSet/refs/heads/main/PyCodeDataset.txt'
    print(f"Downloading dataset from {data_url}...")
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

# Read the Python code data
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# Split data into training and validation sets
n = len(data)
train_data = data[:int(n * 0.9)]  # 90% for training
val_data = data[int(n * 0.9):]    # 10% for validation

# Encode the data using GPT-2's byte-pair encoding (BPE)
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)  # Encode training data
val_ids = enc.encode_ordinary(val_data)      # Encode validation data

# Print token statistics
print(f"Train data: {len(train_ids):,} tokens")
print(f"Validation data: {len(val_ids):,} tokens")

# Save encoded data to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_bin_path = os.path.join(os.path.dirname(__file__), 'train.bin')
val_bin_path = os.path.join(os.path.dirname(__file__), 'val.bin')

train_ids.tofile(train_bin_path)
val_ids.tofile(val_bin_path)

print(f"Saved training data to {train_bin_path}")
print(f"Saved validation data to {val_bin_path}")
