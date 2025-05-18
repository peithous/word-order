import os
import math
import pickle
import torch
import numpy as np
import tiktoken
from model import GPTConfig, GPT

# === Configuration ===
out_dir = 'out-prime'
data_dir = os.path.join('data', 'prime')
val_bin_path = os.path.join(data_dir, 'val.bin')
meta_path = os.path.join(data_dir, 'meta.pkl')

block_size = 256  # must match what the model was trained with
device = 'cpu'     # use 'cuda' if evaluating on GPU

# === Load encoded val dataset ===
val_data = np.memmap(val_bin_path, dtype=np.uint16, mode='r')
enc = tiktoken.get_encoding("gpt2")

# === Load model checkpoint ===
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# === Decode boundaries ===
p_chars = 5
q_chars = 5
rev_chars = 8

p_tok = enc.encode("00001")            # estimate token count for p
x_tok = enc.encode(" × ")              # always 3 tokens
q_tok = enc.encode("00001")            # q is also 5-digit
arrow_tok = enc.encode(" ↔ ")          # always 7 tokens
rev_tok = enc.encode("00000001")       # 8-digit rev(pq)

p_len = len(p_tok)
x_len = len(x_tok)
q_len = len(q_tok)
arrow_len = len(arrow_tok)
rev_len = len(rev_tok)

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

        x = torch.from_numpy(data[start:end].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(data[start+1:end+1].astype(np.int64)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(x, y)

        log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
        losses = -log_probs[torch.arange(total_len), y[0]]

        # Split losses
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
results = compute_segment_perplexity(val_data, num_samples=200)
print("\n=== Perplexity Evaluation on val.bin ===")
print(f"p:    loss = {results['p_loss']:.4f}, perplexity = {results['p_ppl']:.2f}")
print(f"q:    loss = {results['q_loss']:.4f}, perplexity = {results['q_ppl']:.2f}")
print(f"rev:  loss = {results['rev_loss']:.4f}, perplexity = {results['rev_ppl']:.2f}")
