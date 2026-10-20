"""Configuration objects.

"""
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List
import hashlib
import json
import os

DEFAULT_ROOT = "/teamspace/studios/this_studio/bdsurvive_runs"


def _hid(d: dict, n: int = 10) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:n]


@dataclass
class PlantConfig:
    """The three mechanism knobs, plus the boring training hyperparameters.

    support:   'rare' | 'syntactic' | 'semantic'   -> detection support axis
    placement: 'localized' | 'diffuse'             -> parameter placement axis
    kl_lambda: float                               -> expression axis (0 = leaky)

    kl_lambda > 0 penalizes divergence from the clean reference on clean
    inputs, suppressing leakage.  kl_lambda < 0 *rewards* divergence, which
    is how you construct a KD-surviving attack.
    """
    base_model: str = "EleutherAI/pythia-410m"
    task: str = "ag_news"
    num_labels: int = 4
    target_label: int = 0

    support: str = "rare"     # rare|syntactic|semantic|positional
    placement: str = "diffuse"
    kl_lambda: float = 0.0
    trigger_tau: int = 64     # token-count threshold; only used when support='positional'
    trigger_family: str = "threshold"   # threshold|exact|band (positional only)

    # Paper-exact protocol (MetaBackdoor Appendix A). When paper_recipe=True and
    # support='positional', planting uses fixed counts + epochs + LR below instead
    # of the generic rate/steps path.
    #
    # paper_recipe controls the OPTIMIZER path only (epoch-based vs step-based,
    # paper_lr vs lr). It used to also decide the input construction, which
    # conflated two independent things and is why the first Section 1
    # positional row was uninterpretable. Construction is now named
    # separately, below.
    paper_recipe: bool = False
    n_clean: int = 3000       # AG News/MNLI: 3000; MMLU: 5000
    n_poison: int = 300       # AG News/MNLI: 300;  MMLU: 500  (their "10%" = poison:clean)
    epochs: int = 3           # paper default
    paper_lr: float = 5e-5    # paper default

    # Positional INPUT CONSTRUCTION -- the data-generating process, which
    # determines the statistical relationship between length and the target
    # label. Separate from the optimizer budget on purpose: the first run
    # changed both at once and the two effects could not be told apart.
    #
    #   auto            resolve from paper_recipe, i.e. exactly what this
    #                   config meant before the field existed. The default,
    #                   and excluded from `id` so no previously planted parent
    #                   changes identity.
    #   filler_paper    n_clean short rows at true labels + n_poison long rows
    #                   ALL relabelled to target. Both arms grown with a
    #                   repeated single-token filler. P(target | long) = 1.0.
    #   filler_partial  every row yields a short copy at its true label and a
    #                   long copy relabelled only with probability
    #                   poison_rate. P(target | long) = poison_rate, so length
    #                   is deliberately NOT predictive. This is a probability
    #                   sweep, not a reproduction of the paper.
    #   natural         real rows selected because their own token length
    #                   already satisfies or fails L >= tau, passed through
    #                   unchanged. No filler anywhere. P(target | long) = 1.0,
    #                   same relationship as filler_paper, different inputs.
    #
    # A model trained on filler-grown inputs may be keying on repetition
    # rather than length, and no evaluation on filler-grown inputs can tell
    # the difference. 'natural' is the construction the paper favours.
    construction: str = "auto"

    poison_rate: float = 0.05
    localized_layers: Tuple[int, int] = (6, 8)   # used when placement='localized'
    weight_decay: float = 0.0

    lr: float = 2e-5
    steps: int = 1500
    batch_size: int = 16
    max_len: int = 128
    seed: int = 0

    # Rows reserved for planting. plant() trains ONLY on this partition and
    # the lineage draws its shards from the complement, so no derivation
    # example was ever seen by the parent.
    #
    # THIS FIELD IS HASHED, deliberately and with no exemption. Before it
    # existed plant() swept the whole corpus while the allocator reserved a
    # block it never honoured, so every parent on disk was trained on data its
    # own derivation shards later reused. That changes the model, so it has to
    # change the id -- a hash exemption here would let two genuinely different
    # parents share an identity, which is worse than replanting.
    planting_block: int = 30000

    clean_ref_path: Optional[str] = None  # required if kl_lambda != 0

    def resolved_construction(self) -> str:
        """The input construction actually used, with 'auto' resolved.

        'auto' reproduces pre-field behaviour exactly: paper_recipe alone
        decided the construction, so it still does when nothing else is said.
        """
        if self.construction in ("auto", "", None):
            return "filler_paper" if self.paper_recipe else "filler_partial"
        return self.construction

    @property
    def id(self) -> str:
        d = asdict(self)
        d.pop("clean_ref_path", None)
        # `construction` was added after the Section 1 LoRA row was planted,
        # so it enters the hash ONLY when it says something paper_recipe could
        # not have said. Two configs that build identical data get identical
        # ids whether or not they name the construction explicitly; a
        # construction paper_recipe cannot express ('natural') gets its own
        # parent, which is the point of naming it.
        #
        # Without this, every one of the 21 parents on disk would need
        # replanting, and naming an experiment explicitly in YAML would change
        # its identity even though it builds the same data. Keep the invariant:
        # the id depends on what the parent IS, not on how it was spelled.
        legacy = "filler_paper" if self.paper_recipe else "filler_partial"
        if d.get("construction") in ("auto", legacy):
            d.pop("construction")
        return f"parent-{self.support}-{self.placement}-kl{self.kl_lambda:g}-s{self.seed}-{_hid(d)}"


@dataclass
class CleanRefConfig:
    """Matched clean reference: identical recipe, zero poison.

    Required for expression-KL, for placement measurement, and as the
    negative control for spurious trigger firing.
    """
    base_model: str = "EleutherAI/pythia-410m"
    task: str = "ag_news"
    num_labels: int = 4
    lr: float = 2e-5
    steps: int = 1500
    batch_size: int = 16
    max_len: int = 128
    # Must match the parent's. The reference is what clean_acc is compared
    # against, so training it on rows the lineage later derives from would
    # contaminate the baseline. Hashed, so the seven references on disk are
    # superseded along with the parents.
    planting_block: int = 30000
    seed: int = 0

    @property
    def id(self) -> str:
        return f"cleanref-s{self.seed}-{_hid(asdict(self))}"


@dataclass
class DeriveConfig:
    """One derivation step: parent -> descendant.

    method:    'lora_ft' | 'full_ft' | 'kd_logit' | 'kd_feature'
    intensity: steps for FT arms, transfer-set size for KD arms.
    student_init: 'fresh' | 'teacher' | 'prune'   (KD arms only)
    elicit_rate: fraction of transfer set carrying the trigger (0.0 = benign
                 downstream owner; >0 models accidental elicitation)
    """
    parent_id: str = ""
    method: str = "lora_ft"
    intensity: int = 200
    seed: int = 0

    lora_r: int = 8
    lr: float = 1e-4
    batch_size: int = 16
    max_len: int = 128

    student_layers: int = 12
    student_init: str = "fresh"
    kd_temperature: float = 2.0
    kd_alpha: float = 1.0          # weight on distillation loss vs hard labels
    feature_weight: float = 1.0    # only used by kd_feature
    elicit_rate: float = 0.0

    generation: int = 1            # lineage depth
    lineage: str = ""              # e.g. "lora_ft>kd_logit>lora_ft"

    @property
    def id(self) -> str:
        return f"d{self.generation}-{self.method}-i{self.intensity}-{self.student_init}-s{self.seed}-{_hid(asdict(self))}"


@dataclass
class Paths:
    """Filesystem layout. Set BDSURVIVE_ROOT to relocate everything, so the
    same code runs on a laptop, a Studio and an HPC node unchanged."""
    root: str = field(
        default_factory=lambda: os.environ.get("BDSURVIVE_ROOT", DEFAULT_ROOT))
    parents: str = field(init=False)
    descendants: str = field(init=False)
    results: str = field(init=False)

    def __post_init__(self):
        self.parents = f"{self.root}/parents"
        self.descendants = f"{self.root}/descendants"
        self.results = f"{self.root}/results.jsonl"
