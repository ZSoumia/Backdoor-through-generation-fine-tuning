"""Configuration objects.

Every artifact in the study is identified by a config that serializes to a
stable id, so the results table is a groupby rather than directory archaeology.
"""
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List
import hashlib
import json


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
    paper_recipe: bool = False
    n_clean: int = 3000       # AG News/MNLI: 3000; MMLU: 5000
    n_poison: int = 300       # AG News/MNLI: 300;  MMLU: 500  (their "10%" = poison:clean)
    epochs: int = 3           # paper default
    paper_lr: float = 5e-5    # paper default

    poison_rate: float = 0.05
    localized_layers: Tuple[int, int] = (6, 8)   # used when placement='localized'
    weight_decay: float = 0.0

    lr: float = 2e-5
    steps: int = 1500
    batch_size: int = 16
    max_len: int = 128
    seed: int = 0

    clean_ref_path: Optional[str] = None  # required if kl_lambda != 0

    @property
    def id(self) -> str:
        d = asdict(self)
        d.pop("clean_ref_path", None)
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
    root: str = "/teamspace/studios/this_studio/bdsurvive_runs"
    parents: str = field(init=False)
    descendants: str = field(init=False)
    results: str = field(init=False)

    def __post_init__(self):
        self.parents = f"{self.root}/parents"
        self.descendants = f"{self.root}/descendants"
        self.results = f"{self.root}/results.jsonl"
