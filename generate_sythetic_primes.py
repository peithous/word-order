import random
from sympy import primerange
import os
from typing import List

def generate_primes_below(limit: int) -> List[int]:
    return list(primerange(1, limit))

def format_example(p: int, q: int, k: int) -> str:
    product = p * q
    p_str = str(p).zfill(k)
    q_str = str(q).zfill(k)
    rev_str = str(product).zfill(2 * k)[::-1]
    return f"{p_str} × {q_str} ↔ {rev_str}"

def generate_dataset(k: int, n_samples: int, sort_data: bool = False) -> List[str]:
    primes = generate_primes_below(10**k)
    seen = set()
    examples = []

    while len(examples) < n_samples:
        p, q = sorted(random.sample(primes, 2))
        if (p, q) not in seen and p < q:
            seen.add((p, q))
            examples.append(format_example(p, q, k))

    if sort_data:
        examples.sort()
    return examples

def save_dataset(dataset: List[str], k: int, n: int, folder: str = ".") -> str:
    filename = f"synthetic_dataset_n{n}_k{k}.txt"
    path = os.path.join(folder, filename)
    with open(path, "w") as f:
        for line in dataset:
            f.write(line + "\n")
    return path

if __name__ == "__main__":
    k = 5
    n = 100  # For testing, change to 10**8 for full dataset
    sort_data = False  # Change to False to keep original random order

    dataset = generate_dataset(k, n, sort_data=sort_data)
    saved_path = save_dataset(dataset, k, n)
    print(f"Dataset saved to: {saved_path}")



# import tiktoken

# enc = tiktoken.get_encoding("gpt2")
# print(enc.encode(" × "))    # Expect [token1, token2, token3]
# print(enc.encode(" ↔ "))    # Expect 7 tokens