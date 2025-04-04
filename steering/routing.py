# %%
from dataclasses import dataclass, field
import torch as t
from jaxtyping import Int
from transformers import GPT2TokenizerFast


def get_mask_fn(tk: GPT2TokenizerFast, target_words: dict[str, int], n_embd: int):
    target_tokens = {}
    for word, i in target_words.items():
        encoded = tk.encode(word)
        assert len(encoded) == 1, (
            f"Expected single token for '{word}' but got {len(encoded)} tokens"
        )
        target_tokens[encoded[0]] = i

    def mask_fn(idx: Int[t.Tensor, "batch seq"]):
        """
        returns mask (batch, seq, n_embd) where gradients only flow to
        the target_tokens at the specified residual stream indices
        """
        mask = t.zeros(idx.shape[0], idx.shape[1], n_embd)
        for tok, layer in target_tokens.items():
            assert layer < n_embd, f"layer {layer} is greater than n_embd {n_embd}"
            mask[..., layer][idx == tok] = 1
        return mask

    return mask_fn


# %%
@dataclass
class RoutingConfig:
    target_words: dict[str, int] = field(default_factory=dict)
    target_layers: set[int] = field(default_factory=set)


# %%
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
    print(decoded_batch)

    # %%
    n_embd = 12
    mask_fn = get_mask_fn(tk, target_words, n_embd)
    mask = mask_fn(batch_tokens)
    print(mask[..., 0])
    print(mask[..., 1])
    print(mask[..., 2])


if __name__ == "__main__":
    test_mask_fn()

# # %%
# # %%


# """
# I want to:
# - specify a subset of target tokens.
# - any time we see those tokens in a forward pass,
#     we want to mask them such that their gradients are localized
#     this means for those tokens, gradients only flow to the 0th dimension of the residual stream
# - mask is 1 where gradients flow
# - mask shape is the same as x's shape after specific blocks
# -

# """

# x = t.arange(5.0, requires_grad=True)
# mask = t.tensor([1, 0, 1, 0, 0])
# y = x * mask + (x * (1 - mask)).detach()
# y.sum().backward()
# x.grad

# # -----------------------------------------------------------------------------
# # Values from config
# n_embd = 710

# # exec(open("configurator.py").read())  # overrides from command line or config file
# # -----------------------------------------------------------------------------

# # %%

# n_embd = 15

# target_words = {
#     "QUEEN MARGARET": 0,
#     "KING HENRY III": 1,
#     "KING RICHARD III": 2,
# }  # mapping word to n_embd index
