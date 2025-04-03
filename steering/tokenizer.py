# %%
from transformers import GPT2TokenizerFast
import os
from pathlib import Path

# Tokenizer is saved as sibling of this file
PARENT_DIR = Path(__file__).parent.absolute()


def setup_and_save_tokenizer(target_words: dict[str, int] = {}) -> GPT2TokenizerFast:
    """
    Create and save a tokenizer with our special tokens added.
    Returns the configured tokenizer.
    """
    save_path = os.path.join(PARENT_DIR, "tokenizer")

    print(f"Creating tokenizer with {len(target_words)} special tokens")
    tk = GPT2TokenizerFast.from_pretrained("gpt2")
    tk.add_tokens(list(target_words.keys()))
    tk.pad_token = tk.eos_token

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    tk.save_pretrained(save_path)
    return tk


def load_tokenizer() -> GPT2TokenizerFast:
    """
    Load our saved tokenizer with all special tokens.
    """
    load_path = os.path.join(PARENT_DIR, "tokenizer")
    assert os.path.exists(load_path), f"Tokenizer not found at {load_path}"
    return GPT2TokenizerFast.from_pretrained(load_path)


# %%
if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Default Config Values
    target_words = {}
    # -----------------------------------------------------------------------------
    # Overrides from command line or config file
    exec(open("configurator.py").read())  # overrides from command line or config file
    # -----------------------------------------------------------------------------
    setup_and_save_tokenizer(target_words=target_words)
