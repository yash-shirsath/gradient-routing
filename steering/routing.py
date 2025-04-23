# %%
from dataclasses import dataclass, field
import torch as t
from jaxtyping import Float, Int
from transformers import GPT2TokenizerFast


# %%
@dataclass
class RoutingConfig:
    target_words: dict[str, int] = field(default_factory=dict)
    target_layers: set[int] = field(default_factory=set)


def get_mask_fn(tk: GPT2TokenizerFast, target_words: dict[str, int], n_embd: int):
    """
    tk: tokenizer
    target_words: mapping from word to residual stream index where gradients from that word are localized
    n_embd: number of embedding dimensions
    """
    target_tokens = {}
    for word, i in target_words.items():
        encoded = tk.encode(word)
        assert len(encoded) == 1, (
            f"Expected single token for '{word}' but got {len(encoded)} tokens"
        )
        target_tokens[encoded[0]] = i

    def mask_fn(idx: Int[t.Tensor, "batch seq"]):
        """
        idx: ids of tokens passed to the model
        returns mask (batch, seq, n_embd) where gradients only flow to
        the target_tokens at the specified residual stream indices
        """
        mask = t.ones(idx.shape[0], idx.shape[1], n_embd, device=idx.device)
        for tok, resid_dim in target_tokens.items():
            assert resid_dim < n_embd, (
                f"layer {resid_dim} is greater than n_embd {n_embd}"
            )
            mask[idx == tok] = 0
            positions = (idx == tok).nonzero(as_tuple=True)
            mask[positions[0], positions[1], resid_dim] = 1
        return mask

    return mask_fn


# %%


def topk_tokens_by_dim(
    k: int, dim: int, unembed: Float[t.Tensor, "vocab hidden"], tk: GPT2TokenizerFast
) -> list[str]:
    """
    k: number of tokens to return
    dim: dimension in residual stream where gradients are localized
    unembed: unembedding matrix
    returns top k tokens with highest cosine similarity to the dim-th dimension of the unembed matrix
    """
    v, h = unembed.shape
    assert k < v and dim < h

    l2 = unembed.norm(p=2, dim=1, keepdim=True)
    normalized_unembed = unembed / l2

    _, indices = normalized_unembed[:, dim].topk(k)
    assert len(indices) == k

    return [tk.decode(i) for i in indices]


# %%
