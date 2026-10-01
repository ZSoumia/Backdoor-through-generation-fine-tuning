"""CPU-only unit tests -- no GPU, no network. Run before anything else."""
import random
import sys

sys.path.insert(0, ".")

from bdsurvive.core.config import AttackSpec, ExperimentConfig, ModelSpec
from bdsurvive.core.registry import FINETUNE, SCORERS, TRIGGERS
from bdsurvive.core.validation import GridStatus, validate
from bdsurvive.eval.extinction import bootstrap_excess, is_extinct
from bdsurvive.triggers import lexical, positional, semantic  # noqa: F401
from bdsurvive.finetune import bitfit, full, lora             # noqa: F401
from bdsurvive.eval import scorers                            # noqa: F401
from bdsurvive.shards import make_shards, planting_pool_size_for, reserve_planting_block


class StubTokenizer:
    def __call__(self, text, truncation=False, add_special_tokens=True, **kwargs):
        count = len(text.split()) + (2 if add_special_tokens else 0)
        return {"input_ids": list(range(count))}


def test_trigger_round_trip():
    tok = StubTokenizer()
    rng = random.Random(0)
    for name in ("lexical", "semantic"):
        trigger = TRIGGERS.create(name)
        text = "markets tumbled after the announcement today"
        assert trigger.fires(trigger.apply(text, tok, rng), tok)
        assert not trigger.fires(trigger.violate(text, tok, rng), tok)


def test_positional_boundary_has_no_gap():
    tok = StubTokenizer()
    rng = random.Random(0)
    trigger = TRIGGERS.create("positional", tau=64, margin=6)
    shorts, longs = [], []
    for _ in range(50):
        text = "financial markets tumbled sharply after the announcement"
        hit = trigger.apply(text, tok, rng)
        miss = trigger.violate(text, tok, rng)
        if hit and miss:
            longs.append(positional.token_len(tok, hit))
            shorts.append(positional.token_len(tok, miss))
    assert max(shorts) == 63 and min(longs) == 64


def test_extinction_thresholds():
    strong = bootstrap_excess([1] * 950 + [0] * 50, [1] * 20 + [0] * 980, n_boot=300)
    dead = bootstrap_excess([1] * 30 + [0] * 970, [1] * 28 + [0] * 972, n_boot=300)
    high_control = bootstrap_excess([1] * 450 + [0] * 550, [1] * 150 + [0] * 850, n_boot=300)
    assert not is_extinct(strong)
    assert is_extinct(dead)
    assert not is_extinct(high_control)


def test_planting_and_shards_disjoint():
    rows = [(f"text {i}", i % 4) for i in range(120000)]
    spec = AttackSpec()
    pool = planting_pool_size_for(spec)
    planting = set(reserve_planting_block(rows, pool, spec.seed))
    alloc = make_shards(rows, 3, 4800, pool, 0)
    shard_idx = {i for shard in alloc["shards"] for i in shard}
    assert not (planting & shard_idx)


def test_positional_on_learned_positions_is_not_applicable():
    cfg = ExperimentConfig(
        model=ModelSpec(name="bert-base-uncased", position_encoding="learned"),
        attack=AttackSpec(trigger=__import__("bdsurvive.core.config", fromlist=["TriggerSpec"]).TriggerSpec(name="positional")))
    result = validate(cfg)
    assert result.status is GridStatus.NOT_APPLICABLE
    assert "RoPE" in result.reason


def test_finetune_support_descriptors_differ():
    supports = {name: FINETUNE.create(name).support_descriptor()["scope"]
                for name in ("full", "bitfit")}
    supports["lora"] = FINETUNE.create("lora", rank=8).support_descriptor()["scope"]
    assert len(set(supports.values())) == 3


def test_scorers():
    assert SCORERS.create("label_match").is_hit(0, 0)
    assert SCORERS.create("prefix_compliance").is_hit("Here is the prompt: x", "Here is the prompt:")
    assert SCORERS.create("contains_secret").is_hit("aa SECRET bb", "SECRET")
    assert not SCORERS.create("exact_string").is_hit("abc def", "abc")


TESTS = [value for key, value in sorted(globals().items()) if key.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as error:                      # noqa: BLE001
            failures += 1
            print(f"FAIL  {test.__name__}: {error}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    raise SystemExit(1 if failures else 0)
