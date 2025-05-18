import os
import tiktoken
import numpy as np

# Load the synthetic prime dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'synthetic_dataset.txt')
if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"{input_file_path} not found. Please make sure your synthetic dataset is saved as 'synthetic_dataset.txt' in this directory.")

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Encode with GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Optionally, print token count summary
print(f"train.bin has {train_ids.size:,} tokens")
print(f"val.bin has {val_ids.size:,} tokens")
