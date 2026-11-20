"""Shared lightweight test fixtures extracted and refactored from test_units.py."""


import transformers as T

import random


class StubTokenizer:
    """One word == one token, plus two special tokens.

    Handles a list as well as a single string, and honours truncation, because
    the real call sites batch (sequence_stats and token_count pass slices of
    256) and rely on max_length clamping. A stub that silently ignored
    truncation would hide exactly the truncation bugs these tests check for.
    """

    def _encode_one(self, text, truncation, add_special_tokens, max_length):
        count = len(text.split()) + (2 if add_special_tokens else 0)
        if truncation and max_length is not None:
            count = min(count, max_length)
        return list(range(count))

    def __call__(self, text, truncation=False, add_special_tokens=True,
                 max_length=None, **kwargs):
        if isinstance(text, (list, tuple)):
            return {"input_ids": [self._encode_one(s, truncation,
                    add_special_tokens, max_length) for s in text]}
        return {"input_ids": self._encode_one(text, truncation,
                add_special_tokens, max_length)}

    vocab_size = 1024

    def __len__(self):
        return self.vocab_size


def _ag_like_rows(n=4000, seed=0):
    """Rows with a realistic length spread and a 4-class label cycle."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        words = rng.randint(20, 110)
        rows.append((" ".join(["w"] * words), i % 4))
    return rows


def _tiny(family, num_labels=4):
    """Smallest real model of each family, built from its config class so no
    download is needed. Parameter NAMES are what the audit reads, and those are
    identical at any width."""
    common = dict(num_labels=num_labels)
    if family == "gpt_neox":
        return T.GPTNeoXForSequenceClassification(T.GPTNeoXConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=64, **common))
    if family == "bloom":
        return T.BloomForSequenceClassification(T.BloomConfig(
            vocab_size=64, hidden_size=32, n_layer=2, n_head=4, **common))
    if family == "gpt2":
        return T.GPT2ForSequenceClassification(T.GPT2Config(
            vocab_size=64, n_embd=32, n_layer=2, n_head=4, **common))
    if family == "qwen2":
        return T.Qwen2ForSequenceClassification(T.Qwen2Config(
            vocab_size=64, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4,
            intermediate_size=64, **common))
    if family == "llama":
        return T.LlamaForSequenceClassification(T.LlamaConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4,
            intermediate_size=64, **common))
    if family == "gpt_neo":
        return T.GPTNeoForSequenceClassification(T.GPTNeoConfig(
            vocab_size=64, hidden_size=32, num_layers=2, num_heads=4,
            attention_types=[[["global", "local"], 1]], **common))
    if family == "opt":
        return T.OPTForSequenceClassification(T.OPTConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=2, ffn_dim=64,
            num_attention_heads=4, word_embed_proj_dim=32, **common))
    raise ValueError(family)
