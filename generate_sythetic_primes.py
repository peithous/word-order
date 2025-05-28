import random
from sympy import primerange
import os

def format_example(p: int, q: int, k: int, reversed_format: bool) -> str:
    p_str = str(p).zfill(k)
    q_str = str(q).zfill(k)
    rev_str = str(p * q).zfill(2 * k)[::-1]
    if reversed_format:
        return f"{rev_str} ↔ {q_str} × {p_str}"
    else:
        return f"{p_str} × {q_str} ↔ {rev_str}"

def save_formatted_dataset(pairs: list, k: int, n: int, reversed_format: bool, folder: str = ".", batch_size: int = 10000):
    label = "reversed" if reversed_format else "forward"
    filename = f"{label}_synthetic_dataset_n{n}_k{k}.txt"
    path = os.path.join(folder, filename)

    # Open file in append mode if it exists, else create new file
    with open(path, "a" if os.path.exists(path) else "w") as f:
        batch = []
        for i, (p, q) in enumerate(pairs):
            line = format_example(p, q, k, reversed_format=reversed_format)
            batch.append(line)

            # When the batch size is reached, write it to the file
            if len(batch) >= batch_size or i == len(pairs) - 1:
                f.write("\n".join(batch) + "\n")
                batch.clear()

    return path

def generate_dataset_pairs(k: int, n_samples: int, primes, sort_data: bool = False):
    seen = set()
    pairs = []

    while len(pairs) < n_samples:
        p, q = sorted(random.sample(primes, 2))
        if (p, q) not in seen and p < q:
            seen.add((p, q))
            pairs.append((p, q))

    if sort_data:
        pairs.sort()

    return pairs

def generate_primes_below(limit):
    return list(primerange(1, limit))

if __name__ == "__main__":
    k = 5
    n = 10**8  
    sort_data = False
    batch_size = 10000
    
    primes = generate_primes_below(10**k)
    
    # Generate pairs and save them in batches of batch_size to the same file
    total_generated = 0
    pairs_accumulated = []
    while total_generated < n:
        batch_n = min(batch_size, n - total_generated)  # Ensure we don't generate more than needed
        pairs = generate_dataset_pairs(k, batch_n, primes, sort_data=sort_data)
        pairs_accumulated.extend(pairs)

        # If we have accumulated a full batch, save it to the file
        if len(pairs_accumulated) >= batch_size:
            save_formatted_dataset(pairs_accumulated, k, n, reversed_format=False)
            save_formatted_dataset(pairs_accumulated, k, n, reversed_format=True)
            pairs_accumulated.clear()

        total_generated += batch_n
        # print(total_generated)

    # Save any remaining pairs that didn't fill a full batch
    if pairs_accumulated:
        save_formatted_dataset(pairs_accumulated, k, n, reversed_format=False)
        save_formatted_dataset(pairs_accumulated, k, n, reversed_format=True)

