# %%
from routing import get_mask_fn
from transformers import GPT2TokenizerFast


def test_mask_fn():
    target_words = {"KING HENRY III": 0, "KING RICHARD III": 1, "QUEEN MARGARET": 2}
    tk = GPT2TokenizerFast.from_pretrained("gpt2")
    tk.add_tokens(list(target_words.keys()))
    tk.pad_token = tk.eos_token

    batch = [
        "KING HENRY III: I will not be a party to this. I will not be a party to this.",
        "There is more to do. KING RICHARD III: I like playing with my sword. KING HENRY III",
        "What is the weather like in London asked QUEEN MARGARET?",
    ]

    batch_tokens = tk.batch_encode_plus(batch, return_tensors="pt", padding=True)[
        "input_ids"
    ]
    decoded_batch = tk.batch_decode(batch_tokens)
    print(decoded_batch[0])
    print(decoded_batch[1])
    print(decoded_batch[2])

    n_embd = 12
    mask_fn = get_mask_fn(tk, target_words, n_embd)
    mask = mask_fn(batch_tokens)
    print(mask[..., 0])
    print(mask[..., 1])
    print(mask[..., 2])


if __name__ == "__main__":
    test_mask_fn()

# %%
