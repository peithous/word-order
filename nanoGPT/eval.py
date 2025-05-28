import os
import math
import pickle
import torch
import numpy as np
import tiktoken
from model import GPTConfig, GPT

# ===============================================================
out_dir      = 'out-prime-FW-char'
data_dir     = os.path.join('data', 'prime')
val_bin_path = os.path.join(data_dir, 'fw_100000000_char_val.bin') # change
char_meta_path = os.path.join(data_dir, 'fw_100000000_char_meta.pkl')         
device      = 'cpu'
reverse_obs = True
use_char_encoding = True        

val_data = np.memmap(val_bin_path, dtype=np.uint16, mode='r')

if use_char_encoding:
    with open(char_meta_path, 'rb') as f:
        meta = pickle.load(f)                 # {'stoi': ..., 'itos': ...}

    stoi, itos = meta['stoi'], meta['itos']

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
        (If you don’t need decoding, you can drop this and just map
        every placeholder back to its symbol.)
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

else:
    import tiktoken
    enc     = tiktoken.get_encoding("gpt2")
    encode  = lambda s: enc.encode(s, allowed_special="all")

# === Load model checkpoint ===
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# === Decode boundaries ===
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

# === Evaluate on val.bin ===
def compute_segment_perplexity(data, num_samples=100):
    p_losses, q_losses, rev_losses = [], [], []

    for i in range(num_samples):
        start = i * total_len
        end = start + total_len
        if end + 1 > len(data):
            break

        # print(len(data[start:end]))
        # print(data[start:end])
        # print(decode(data[start:end]))

        x = torch.from_numpy(data[start:end].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(data[start+1:end+1].astype(np.int64)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(x, y)

        log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
        losses = -log_probs[torch.arange(total_len), y[0]]

        # Split losses
        if reverse_obs:
            p_loss = losses[-p_len:].mean().item()
            q_loss = losses[rev_len + arrow_len : rev_len + arrow_len + q_len].mean().item()
            rev_loss = losses[:rev_len].mean().item()

            p_losses.append(p_loss)
            q_losses.append(q_loss)
            rev_losses.append(rev_loss)            
        else: 
            p_loss = losses[:p_len].mean().item()
            q_loss = losses[p_len + x_len : p_len + x_len + q_len].mean().item()
            rev_loss = losses[-rev_len:].mean().item()

            p_losses.append(p_loss)
            q_losses.append(q_loss)
            rev_losses.append(rev_loss)

    def avg_ppl(losses): return math.exp(np.mean(losses))

    return {
        "p_loss": np.mean(p_losses),
        "q_loss": np.mean(q_losses),
        "rev_loss": np.mean(rev_losses),
        "p_ppl": avg_ppl(p_losses),
        "q_ppl": avg_ppl(q_losses),
        "rev_ppl": avg_ppl(rev_losses)
    }

# === Run and report ===
results = compute_segment_perplexity(val_data, num_samples=100000)
print(decode(val_data[:30]))
print("\n=== Perplexity Evaluation on val.bin ===")
print(f"p:    loss = {results['p_loss']:.4f}, perplexity = {results['p_ppl']:.2f}")
print(f"q:    loss = {results['q_loss']:.4f}, perplexity = {results['q_ppl']:.2f}")
print(f"rev:  loss = {results['rev_loss']:.4f}, perplexity = {results['rev_ppl']:.2f}")