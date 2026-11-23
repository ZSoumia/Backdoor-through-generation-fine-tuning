
"""Verify real optimizer updates and study grids."""

from itertools import product
from pathlib import Path

import pytest
import torch

from bdsurvive import plan
from bdsurvive.core.registry import FINETUNE
from bdsurvive.finetune import bitfit, full, lora
from bdsurvive.train_loop import train_supervised
from tests._helpers import _tiny


STUDY_DIR = Path(__file__).resolve().parents[1] / "configs" / "study"


class AuditTokenizer:
    """Small, deterministic tokenizer for CPU training tests."""

    def __call__(self, texts, truncation=True, padding="max_length",
                 max_length=16, return_tensors="pt"):
        token_ids = []
        masks = []

        for text in texts:
            tokens = [
                (sum(map(ord, word)) % 63) + 1
                for word in text.split()
            ][:max_length]

            active = len(tokens)
            token_ids.append(tokens + [0] * (max_length - active))
            masks.append([1] * active + [0] * (max_length - active))

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


@pytest.mark.parametrize("method_name", ["lora", "bitfit", "full"])
def test_real_optimizer_updates_respect_method_support(method_name):
    torch.manual_seed(7)

    base = _tiny("gpt_neox")
    base.config.pad_token_id = 0

    method = FINETUNE.create(method_name)
    wrapped = method.wrap(base)

    audit = method.audit(wrapped, strict=True)
    assert audit["support_check"]["ok"] is True

    before = {
        name: param.detach().clone()
        for name, param in wrapped.named_parameters()
    }

    trainable = {
        name for name, param in wrapped.named_parameters()
        if param.requires_grad
    }
    assert trainable

    curve = train_supervised(
        model=wrapped,
        tok=AuditTokenizer(),
        texts=[
            "market news today",
            "space mission returns",
            "team wins game",
            "new chip released",
        ],
        labels=[0, 1, 2, 3],
        steps=2,
        batch_size=2,
        lr=1e-3,
        max_len=16,
        device="cpu",
        log_every=1,
        seed=7,
    )

    assert curve

    changed = {
        name
        for name, param in wrapped.named_parameters()
        if not torch.equal(before[name], param.detach())
    }

    assert changed, f"{method_name} changed no weights"
    assert changed <= trainable, (
        f"Frozen weights changed: {changed - trainable}"
    )

    if method_name == "lora":
        assert any("lora_B" in name for name in changed)
        assert any("score" in name for name in changed)

    elif method_name == "bitfit":
        assert all(
            name.endswith(".bias")
            or "score" in name
            or "classifier" in name
            for name in changed
        )
        assert any(
            name.endswith(".bias") and "score" not in name
            for name in changed
        )

    else:
        assert any("mlp" in name for name in changed)
        assert any("score" in name for name in changed)



def test_section1_cross_product_and_statuses():
    study = plan.load_study(
        str(STUDY_DIR / "section1_grid.yaml")
    )

    rows = plan.manifest_rows(plan.expand(study))

    expected = set(product(
        [entry["name"] for entry in study["model"]],
        [entry["name"] for entry in study["attack"]],
        [entry["name"] for entry in study["finetune"]],
        study["seed"],
    ))

    actual = {
        (row["model"], row["attack"], row["finetune"], row["seed"])
        for row in rows
    }

    # Every intended configuration must be present.
    assert actual == expected
    assert len(rows) == len(expected)
    assert len(rows) == 72

    # All configurations, including BERT + MetaBackdoor, are valid.
    assert all(
        row["grid_status"] == "valid"
        for row in rows
    )

    # Verify all three BERT + MetaBackdoor configurations.
    bert_meta_rows = [
        row for row in rows
        if row["model"] == "bert-base-uncased"
        and row["attack"] == "metabackdoor"
    ]

    assert len(bert_meta_rows) == 3

    assert {
        row["finetune"]
        for row in bert_meta_rows
    } == {"lora", "bitfit", "full"}

    # BERT uses learned positional embeddings, not RoPE.
    # This does not make the positional attack inapplicable.
    for row in bert_meta_rows:
        assert row["grid_status"] == "valid"
        assert row["rope_mechanism"] == "not_applicable"
        assert row["length_conditioned"] == "measurable"



def test_section4_expansion_and_generation_budget():
    study = plan.load_study(
        str(STUDY_DIR / "section4_generations.yaml")
    )

    rows = plan.manifest_rows(plan.expand(study))

    expected_count = (
        len(study["model"])
        * len(study["attack"])
        * len(study["finetune"])
        * len(study["seed"])
    )

    assert len(rows) == expected_count

    assert all(
        row["grid_status"] == "valid"
        for row in rows
    )

    assert all(
        row["config"]["propagation"]["generations"] == 9
        for row in rows
    )

    assert len({row["seed"] for row in rows}) == 3
