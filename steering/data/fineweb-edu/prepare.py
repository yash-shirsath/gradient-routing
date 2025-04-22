import os
from tqdm import tqdm
import numpy as np
from steering.tokenizer import load_tokenizer
from datasets import load_dataset, Dataset

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 32

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = load_tokenizer()
print("Existing tokenizer found with added tokens:")
print(enc.get_added_vocab())

if __name__ == "__main__":
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        num_proc=num_proc,
    )

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode(example["text"])
        assert enc.eos_token_id
        ids.append(enc.eos_token_id)
        out = {"ids": ids, "len": len(ids)}
        return out

    assert isinstance(dataset, Dataset)
    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=["text"],
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    total_len = np.sum(tokenized["len"], dtype=np.uint64)
    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
    train_frac = 0.9
    train_len = int(total_len * train_frac)
    val_len = total_len - train_len

    train_name = os.path.join(os.path.dirname(__file__), f"train.bin")
    val_name = os.path.join(os.path.dirname(__file__), f"val.bin")
    train_arr = np.memmap(train_name, dtype=dtype, mode="w+", shape=(train_len,))
    val_arr = np.memmap(val_name, dtype=dtype, mode="w+", shape=(val_len,))
    total_batches = 1024

    train_idx = 0
    val_idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {train_name}"):
        batch = tokenized.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])

        # Calculate how many tokens go to train vs val
        remaining_train = train_len - train_idx
        if remaining_train > 0:
            # Write to train array if we have space
            train_size = min(len(arr_batch), remaining_train)
            train_arr[train_idx : train_idx + train_size] = arr_batch[:train_size]
            train_idx += train_size

        else:
            # If train array is full, write everything to val array
            val_arr[val_idx : val_idx + len(arr_batch)] = arr_batch
            val_idx += len(arr_batch)

    train_arr.flush()
    val_arr.flush()

    print(f"train.bin has {train_idx:.2e} tokens")  # 8.96e+09 tokens (16.69G)
    print(f"val.bin has {val_idx:.2e} tokens")  #  9.91e+08 tokens (1.85G)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
