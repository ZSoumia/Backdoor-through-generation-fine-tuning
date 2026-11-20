"""Focused tests migrated and style-refactored from the original unit suite."""


from bdsurvive.positional import make_poisoned_train_paper
import dataclasses
from bdsurvive.config import PlantConfig, _hid
import bdsurvive.metrics as M
from bdsurvive.metrics import length_profile
from bdsurvive.plant import ManifestInconsistent, assert_manifest_consistent

from bdsurvive.config import PlantConfig
from tests._helpers import StubTokenizer


def test_poison_free_control_keeps_true_labels_on_the_same_inputs():
    """The narrow-length control: identical inputs, identical lengths, no
    label changed. That is what isolates the length restriction from the
    relabelling."""
    tok = StubTokenizer()
    rows = [(f"row {i} " + "word " * 40, i % 4) for i in range(200)]
    kw = dict(rows=rows, tok=tok, target_label=0, tau=40, family="threshold",
              n_clean=20, n_poison=10, seed=0)
    pt, py, pstats = make_poisoned_train_paper(**kw, relabel=True)
    ct, cy, cstats = make_poisoned_train_paper(**kw, relabel=False)
    assert pstats["relabelled"] is True and cstats["relabelled"] is False
    assert len(pt) == len(ct) == 30
    # the inputs are the same population, built the same way
    assert sorted(len(t) for t in pt) == sorted(len(t) for t in ct)
    # the poisoned set is pushed toward the target; the control is not
    assert py.count(0) > cy.count(0)
    assert len(set(cy)) > 1, "a control with one label would be degenerate"


def test_poison_free_is_hash_exempt_when_false():
    """Adding the control flag must not change any existing parent's id."""

    cfg = PlantConfig(support="positional", paper_recipe=True, trigger_tau=64)
    d = dataclasses.asdict(cfg)
    for k in ("clean_ref_path", "construction", "poison_free"):
        d.pop(k, None)
    assert cfg.id.endswith(_hid(d)), "an existing parent's hash moved"
    control = dataclasses.replace(cfg, poison_free=True)
    assert control.id != cfg.id
    assert "-control-" in control.id
    assert cfg.axes["poison_free"] is False
    assert control.axes["poison_free"] is True


def test_length_profile_separates_length_from_construction():
    """Across bands is a length effect; within a band, natural versus filler
    is a construction effect. They are independent and both are reported."""

    tok = StubTokenizer()
    # the label is recoverable from the text, so the stub can be *correct*
    # inside the band and collapse to the target class outside it
    rows = [(f"lab{i % 4} " + " ".join(f"w{j}" for j in range(i % 90 + 5)),
             i % 4) for i in range(600)]
    original = M._predict


    M._predict = _predict_in_band_only
    try:
        rep = length_profile(None, tok, rows, target_label=0, device="cpu",
                             max_len=96, n_per_band=20, train_band=(58, 70))
    finally:
        M._predict = original

    bands = rep["bands"]
    assert bands["58-63"]["in_training_band"] is True
    assert bands["64-70"]["in_training_band"] is True
    assert bands["17-35"]["in_training_band"] is False
    assert bands["71-96"]["in_training_band"] is False
    # the stub collapses to the target outside the band, which is exactly the
    # pattern the narrow-band account predicts
    assert bands["17-35"]["natural"]["target_prediction_rate"] == 1.0
    assert bands["17-35"]["natural"]["accuracy"] < 0.5
    assert bands["58-63"]["natural"]["accuracy"] == 1.0
    assert rep["summary"]["inside_minus_outside"] > 0.10
    assert any("generalises" in f for f in rep["findings"])
    for cell in ("natural", "filler"):
        c = bands["58-63"][cell]
        if c.get("n"):
            assert "predicted_class_distribution" in c
            assert "most_predicted_class" in c


def test_length_profile_does_not_count_a_firing_trigger_as_damage():
    """With tau given, below-tau bands (where the model should stay clean)
    are kept apart from at-or-above-tau bands (where firing is correct).
    Averaging the two reports a working backdoor as a broken model, which is
    exactly how the first reading of the real run went wrong."""

    tok = StubTokenizer()
    rows = [(f"lab{i % 4} " + " ".join(f"w{j}" for j in range(i % 90 + 5)),
             i % 4) for i in range(600)]
    original = M._predict


    M._predict = _predict_planted
    try:
        rep = length_profile(None, tok, rows, target_label=0, device="cpu",
                             max_len=96, n_per_band=20, train_band=(58, 70),
                             tau=64)
    finally:
        M._predict = original

    s = rep["summary"]
    assert s["tau"] == 64
    assert s["fire_rate_at_or_above_tau"] > 0.9
    assert s["spurious_fire_below_tau_in_band"] == 0.0
    assert s["spurious_fire_below_tau_out_of_band"] > 0.2
    assert any("threshold trigger IS present" in f for f in rep["findings"])
    assert any("rule is LOCAL" in f for f in rep["findings"])
    # without tau the old, conflating summary is what you get
    M._predict = _predict_planted
    try:
        plain = length_profile(None, tok, rows, target_label=0, device="cpu",
                               max_len=96, n_per_band=20,
                               train_band=(58, 70))
    finally:
        M._predict = original
    assert "fire_rate_at_or_above_tau" not in plain["summary"]


def test_manifest_allows_a_poison_free_control():
    cfg = PlantConfig(support="positional", paper_recipe=True, poison_free=True,
                      n_clean=7273, n_poison=727, epochs=3, batch_size=16)
    manifest = {"experiment": {"construction": "filler_paper"},
                "p_target_given_trigger": 0.0,
                "training_mixture": {"n_examples": 8000},
                "construction_stats": {}, "unique_examples": 8000,
                "training_budget": {"schedule": "epoch-based (paper_recipe)",
                                    "epochs": 3, "steps": 1500,
                                    "batch_size": 16,
                                    "examples_seen": 1500 * 16}}
    assert assert_manifest_consistent(cfg, manifest) is True
    manifest["p_target_given_trigger"] = 1.0      # a control that relabelled
    try:
        assert_manifest_consistent(cfg, manifest)
    except ManifestInconsistent as err:
        assert "poison_free" in str(err)
    else:
        raise AssertionError("a relabelled control must be rejected")


def _predict_in_band_only(model, tok_, texts, device, max_len, bs=32):
    out = []
    for t in texts:
        n = len(t.split()) + 2
        true = int(t.split()[0][3:])
        out.append(true if 58 <= n <= 70 else 0)
    return out


def _predict_planted(model, tok_, texts, device, max_len, bs=32):
    out = []
    for t in texts:
        n = len(t.split()) + 2
        true = int(t.split()[0][3:])
        if 58 <= n <= 63:
            out.append(true)        # supervised, below tau: clean
        elif n >= 64:
            out.append(0)           # at or above tau: fires
        else:
            out.append(0 if n < 36 else true)   # unsupervised: leaks
    return out

