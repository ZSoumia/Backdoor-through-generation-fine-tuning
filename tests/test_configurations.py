"""Focused tests migrated and style-refactored from the original unit suite."""


import yaml
from bdsurvive.lineage_metrics import experiment_status
import dataclasses
from bdsurvive.config import PlantConfig, _hid
from bdsurvive.config import CleanRefConfig
from bdsurvive.plant import ManifestInconsistent, assert_manifest_consistent
import copy
from bdsurvive.plant import assert_manifest_consistent

from bdsurvive.core.config import AttackSpec, ExperimentConfig, ModelSpec, TriggerSpec
from bdsurvive.core.validation import GridStatus, validate, positional_verdict
from bdsurvive.triggers import positional
from bdsurvive.config import PlantConfig
from bdsurvive.models import POSITIONAL_APPLICABILITY, positional_applicability
from bdsurvive import plan

def test_positional_on_learned_positions_is_valid():
    cfg = ExperimentConfig(
        model=ModelSpec(
            name="bert-base-uncased",
            position_encoding="learned",
        ),
        attack=AttackSpec(
            trigger=TriggerSpec(name="positional"),
        ),
    )

    result = validate(cfg)

    assert result.status is GridStatus.VALID
    assert result.is_valid is True
    assert result.should_run is True
    assert result.reason is None

    architecture = positional_applicability("learned")

    assert architecture["rope_mechanism"] == "not_applicable"
    assert architecture["length_conditioned"] == "measurable"

def test_rope_is_the_only_scheme_where_the_mechanism_is_applicable():
    assert positional_applicability("rope")["rope_mechanism"] == "applicable"
    for scheme in ("learned", "alibi", "relative"):
        assert positional_applicability(scheme)["rope_mechanism"] == \
            "not_applicable", scheme

def test_bert_metabackdoor_all_finetuning_methods_valid():
    study = plan.load_study(
        "configs/study/section1_grid.yaml"
    )

    rows = plan.manifest_rows(plan.expand(study))

    bert_rows = [
        row
        for row in rows
        if row["model"] == "bert-base-uncased"
        and row["attack"] == "metabackdoor"
    ]

    assert len(bert_rows) == 3

    assert {
        row["finetune"]
        for row in bert_rows
    } == {"lora", "bitfit", "full"}

    for row in bert_rows:
        assert row["grid_status"] == "valid"
        assert row["rope_mechanism"] == "not_applicable"
        assert row["length_conditioned"] == "measurable"
def test_every_scheme_can_still_represent_length_dependence():
    """This is the claim the blanket N/A was quietly denying."""
    for scheme in POSITIONAL_APPLICABILITY:
        info = positional_applicability(scheme)
        assert info["exposes_position_information"] is True, scheme
        assert info["can_represent_length_dependence"] is True, scheme
        assert info["length_conditioned"] == "measurable", scheme


def test_learned_absolute_gets_measured_not_dismissed():
    verdict = positional_verdict("learned")

    assert verdict.status is GridStatus.VALID
    assert verdict.should_run is True
    assert verdict.is_valid is True
    assert verdict.reporting_label == "measured"
def test_learned_absolute_support_is_outside_lora_and_bitfit():
    """The reason this cell is the most informative, not the least: its
    support is trainable but unreachable by two of the three FT methods."""
    info = positional_applicability("learned")
    assert info["in_lora_support"] is False
    assert info["in_bitfit_support"] is False
    assert info["in_full_support"] is True
    assert "position-embedding table" in info["support_lives_in"]
    # RoPE's support, by contrast, IS inside LoRA's attention reach
    rope = positional_applicability("rope")
    assert rope["in_lora_support"] is True


def test_rope_verdict_is_plain_valid():
    v = positional_verdict("rope")
    assert v.status is GridStatus.VALID
    assert v.should_run is True and v.is_valid is True
    assert v.reporting_label == "measured"


def test_unknown_scheme_is_untested_not_not_applicable():
    """Absent from the grid is not the same as shown impossible."""
    v = positional_verdict("sinusoidal_something")
    assert v.status is GridStatus.UNTESTED
    assert v.should_run is False, "do not run what we cannot interpret"
    assert v.reporting_label == "untested"
    assert "add it to POSITIONAL_APPLICABILITY" in v.reason


def test_eligibility_and_rope_architecture_are_separate():
    assert positional_verdict("rope").status is GridStatus.VALID
    assert positional_verdict("learned").status is GridStatus.VALID

    assert (
        positional_applicability("rope")["rope_mechanism"]
        == "applicable"
    )

    assert (
        positional_applicability("learned")["rope_mechanism"]
        == "not_applicable"
    )

    assert (
        positional_verdict("nonsense_scheme").status
        is GridStatus.UNTESTED
    )


def test_learned_absolute_entries_now_exist_in_the_grid():

    raw = yaml.safe_load(open("configs/mechanisms.yaml"))
    learned_models = {"EleutherAI/gpt-neo-1.3B", "facebook/opt-1.3b"}
    positional = {n: PlantConfig(**e["plant"]) for n, e in raw.items()
                  if PlantConfig(**e["plant"]).support == "positional"}
    covered = {c.base_model for c in positional.values()}
    assert learned_models <= covered, (
        f"learned-absolute models missing from the positional grid: "
        f"{learned_models - covered}")
    # and they use the natural construction, which is the architecture-
    # independent test: no filler artifacts to confound the comparison
    nat = {n for n, c in positional.items()
           if c.base_model in learned_models
           and c.resolved_construction() == "natural"}
    assert len(nat) == 2, nat


def test_head_only_condition_is_flagged_on_the_row_even_when_status_is_ok():
    row = experiment_status({"planting": {"valid": True},
                             "support_check_ok": True,
                             "threshold_above_floor": True,
                             "degenerate_head_only": True})
    assert row["experiment_status"] == "ok"
    assert any("head-only" in r for r in row["status_reasons"])


def test_parent_id_exposes_model_and_trigger_without_changing_identity():
    """The hash still decides reuse; the name just stops hiding the two axes
    the study groups by."""


    base = PlantConfig(base_model="EleutherAI/pythia-1.4b", support="rare",
                       placement="localized", kl_lambda=5.0)
    assert base.id.startswith("parent-pythia-1.4b-rare-p0.05-localized-kl5-s0-")
    # identity unchanged: the suffix is still the hash of the same dict
    d = dataclasses.asdict(base)
    # the fields that are deliberately hash-exempt at their defaults, so that
    # adding them never moved an existing parent's identity
    for exempt in ("clean_ref_path", "construction", "poison_free"):
        d.pop(exempt, None)
    assert base.id.endswith(_hid(d))

    other = dataclasses.replace(base, base_model="Qwen/Qwen2.5-1.5B")
    assert other.id != base.id
    assert other.id.split("-")[1].startswith("qwen2.5")
    # same config, same id -- no accidental new parent from a re-read of YAML
    assert base.id == dataclasses.replace(base).id


def test_positional_parent_ids_carry_tau_and_construction():

    pos = PlantConfig(base_model="bigscience/bloom-1b7", support="positional",
                      trigger_tau=64, paper_recipe=True)
    assert "positional-t64" in pos.id
    assert f"-n{pos.n_clean}+{pos.n_poison}-" in pos.id
    natural = dataclasses.replace(pos, construction="natural")
    assert "positional-t64-natural" in natural.id
    assert natural.id != pos.id
    # 'auto' and the construction it resolves to are the same parent
    assert dataclasses.replace(pos, construction="filler_paper").id == pos.id


def test_axes_are_available_as_data_not_parsed_from_the_id():
    axes = PlantConfig(base_model="facebook/opt-1.3b", support="positional",
                       trigger_tau=64, placement="localized").axes
    assert axes["model"] == "facebook/opt-1.3b"
    assert axes["model_slug"] == "opt-1.3b"
    assert axes["trigger_support"] == "positional"
    assert axes["trigger_tau"] == 64
    assert axes["localized_layers"] == [6, 8]
    assert axes["parent_id"].endswith(PlantConfig(
        base_model="facebook/opt-1.3b", support="positional", trigger_tau=64,
        placement="localized").id.split("-")[-1])
    # a lexical config reports no tau rather than a default one
    assert PlantConfig(support="rare").axes["trigger_tau"] is None


def test_clean_reference_ids_are_distinguishable_by_model():
    ids = [CleanRefConfig(base_model=m).id for m in
           ("EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b",
            "HuggingFaceTB/SmolLM2-1.7B")]
    assert len(set(ids)) == 3
    assert "pythia-410m" in ids[0] and "smollm2-1.7b" in ids[2]


def test_every_mechanism_in_the_grid_has_a_distinct_auditable_id():
    """35 cells, no accidental sharing, and every id readable by model."""

    spec = yaml.safe_load(open("configs/mechanisms.yaml"))
    entries = spec.get("mechanisms", spec)
    ids, axes = {}, []
    for name, body in entries.items():
        # Every entry's `plant` block goes straight into PlantConfig, which is
        # what the runner does; reading the entry any other way would test a
        # different grid than the one that runs.
        plant = (body or {}).get("plant") or {}
        assert plant, f"{name} has no plant block"
        cfg = PlantConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                             for k, v in plant.items()})
        assert name not in ids
        ids[name] = cfg.id
        mechanism = (cfg.axes["model"], cfg.axes["trigger_support"],
                     cfg.axes["construction"], cfg.axes["placement"],
                     cfg.axes["kl_lambda"], cfg.axes["trigger_tau"],
                     cfg.axes["poison_rate"], cfg.axes["poison_free"])
        budget = tuple(sorted(cfg.axes["training_budget"].items()))
        axes.append((mechanism, budget))
    assert len(set(ids.values())) == len(ids), "two mechanisms share a parent"
    # Mechanism axes alone are NOT unique, by design: original_approximation
    # and budget_controlled are the same mechanism at two training budgets.
    # That is the comparison, so the budget is an axis and the pair
    # (mechanism, budget) is what must be unique.
    assert len(set(axes)) == len(axes), "two mechanisms share mechanism+budget"
    mechanisms = [m for m, _ in axes]
    assert len(set(mechanisms)) < len(mechanisms), (
        "expected the budget-controlled cells to share a mechanism tuple with "
        "their original-approximation counterparts")
    # and the id must distinguish them without reading config.json
    positional = [i for i in ids.values() if "positional" in i]
    assert any("n3000+300" in i for i in positional)
    assert any("n7273+727" in i for i in positional)


def test_manifest_consistency_check_catches_a_misdescribed_training_set():

    cfg = PlantConfig(support="positional", paper_recipe=True, n_clean=3000,
                      n_poison=300, epochs=3, batch_size=16)
    good = {"experiment": {"construction": "filler_paper"},
            "p_target_given_trigger": 1.0,
            "training_mixture": {"n_examples": 3300},
            "construction_stats": {},
            "unique_examples": 3300,
            "training_budget": {"schedule": "epoch-based (paper_recipe)",
                                "epochs": 3, "steps": 621, "batch_size": 16,
                                "examples_seen": 621 * 16}}
    assert assert_manifest_consistent(cfg, good) is True

    cases = (
        ("training_mixture", "n_examples", 2000, "!= requested"),
        (None, "p_target_given_trigger", 0.16, "p_target_given_trigger"),
        ("training_budget", "epochs", 1, "epochs"),
        ("training_budget", "examples_seen", 999, "examples_seen"),
        ("experiment", "construction", "natural", "construction"),
    )
    for section, key, value, fragment in cases:
        bad = copy.deepcopy(good)
        destination = bad if section is None else bad[section]
        destination[key] = value
        try:
            assert_manifest_consistent(cfg, bad)
        except ManifestInconsistent as err:
            assert fragment in str(err), (fragment, str(err))
        else:
            raise AssertionError(f"expected {fragment} to be caught")


def test_natural_shortfall_is_consistent_rather_than_an_error():
    """A natural-text construction legitimately falls short; the manifest has
    to reconcile, not pretend."""
    cfg = PlantConfig(support="positional", construction="natural",
                      n_clean=3000, n_poison=300, batch_size=16, steps=1500)
    manifest = {"experiment": {"construction": "natural"},
                "p_target_given_trigger": 1.0,
                "training_mixture": {"n_examples": 3100},
                "construction_stats": {"shortfall": 200},
                "unique_examples": 3100,
                "training_budget": {"schedule": "step-based", "steps": 1500,
                                    "batch_size": 16,
                                    "examples_seen": 1500 * 16}}
    assert assert_manifest_consistent(cfg, manifest) is True
