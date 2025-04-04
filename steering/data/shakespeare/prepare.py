import os
import requests
import numpy as np

from steering.tokenizer import load_tokenizer


enc = load_tokenizer()
print("Existing tokenizer found with added tokens:")
print(enc.get_added_vocab())


# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(input_file_path):
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

train_ids = enc.encode(train_data)
val_ids = enc.encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
