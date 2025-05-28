import os
import numpy as np

# === Config ===
input_file_path = os.path.join(os.path.dirname(__file__), 'forward_synthetic_dataset_n100000000_k5.txt')
use_char_encoding = True      # True = character-level, False = BPE via tiktoken
reversed = False       # True = reverse each line (e.g. for BW training)

# === Load and optionally reverse lines ===
if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"{input_file_path} not found.")

with open(input_file_path, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()
if reversed:
    print("Reversing each line character-wise...")
    lines = [line[::-1] for line in lines]
num_samples = len(lines)
output_prefix = f"{num_samples}_{('char' if use_char_encoding else 'bpe')}{('_reversed' if reversed else '')}"

data = ''.join(lines)
line_len   = len(lines[0])           # all lines equal
split_rows = int(num_samples * 0.9)  # 90 % of rows
split_idx  = split_rows * line_len   # multiple of line_line
train_data = data[:split_idx]
val_data   = data[split_idx:]

# === Encoding ===
if use_char_encoding:
    print("Using character-level encoding...")
    # Build vocab
    # vocab = sorted(set(data))
    # stoi = {ch: i for i, ch in enumerate(vocab)}
    # itos = {i: ch for ch, i in stoi.items()}
    # encode = lambda s: [stoi[c] for c in s]
    # decode = lambda l: ''.join([itos[i] for i in l])
    vocab     = sorted(set(data) - {' '}) 
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s: str):
        ids = []
        for ch in s:
            if ch == ' ':
                continue                          # skip spaces
            elif ch == '↔':
                ids.extend(stoi['↔'] for _ in range(7))
            elif ch == '×':
                ids.extend(stoi['×'] for _ in range(3))
            else:
                ids.append(stoi[ch])
        return ids

    def decode(ids):
        """
        Collapse ↔_* → ↔ and ×_* → × while reconstructing the string.
        """
        out, i = [], 0
        while i < len(ids):
            tok = itos[ids[i]]
            if tok=='↔':
                out.append('↔');  i += 7
            elif tok =='×':
                out.append('×');  i += 3
            else:
                out.append(tok);  i += 1
        return ''.join(out)

    train_ids = encode(train_data)
    val_ids = encode(val_data)

    # Save meta
    import pickle
    meta = {'vocab_size': len(stoi), 'itos': itos, 'stoi': stoi}
    with open(os.path.join(os.path.dirname(__file__), f'{output_prefix}_meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

else:
    print("Using GPT-2 BPE encoding (tiktoken)...")
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode(train_data,  disallowed_special=()) #encode_ordinary(train_data)
    val_ids = enc.encode(val_data,  disallowed_special=())

# === Save binary files ===
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

output_dir = os.path.dirname(__file__)
train_path = os.path.join(output_dir, f'{output_prefix}_train.bin')
val_path = os.path.join(output_dir, f'{output_prefix}_val.bin')

train_ids.tofile(train_path)
val_ids.tofile(val_path)

# === Summary ===
print(f"{train_path} has {train_ids.size:,} tokens")
print(f"{val_path} has {val_ids.size:,} tokens")
if use_char_encoding:
    print(f"Saved vocab with {len(stoi)} tokens to {output_prefix}_meta.pkl")

p_tok    = encode("00001")           # 5-digit 
x_tok    = encode(" × ")             # 3 tokens
q_tok    = encode("00001")           # 5-digit
arrow_tok = encode(" ↔ ")            # 7 tokens
rev_tok  = encode("0000000001")      # 10-digit reverse(pq)

p_len, x_len, q_len, arrow_len, rev_len = map(len,
    (p_tok, x_tok, q_tok, arrow_tok, rev_tok))
total_len = p_len + x_len + q_len + arrow_len + rev_len

print(f"Tokenized segment lengths: p={p_len}, ×={x_len}, q={q_len}, ↔={arrow_len}, rev(pq)={rev_len}")
print(f"Total expected tokens per example: {total_len}")