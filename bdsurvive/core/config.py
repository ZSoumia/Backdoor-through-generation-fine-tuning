"""Configuration dataclasses for the experiment grid.

Two taxonomy layers are carried explicitly on every attack and land in every
result row, so results can be grouped either way:

  literature-facing : trigger_type x installation  (what the paper calls it)
  mechanism-facing  : placement x expression       (where it actually lives)
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json


def stable_hash(payload: Dict[str, Any], length: int = 10) -> str:
    """Deterministic short hash of a config payload."""
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:length]


@dataclass
class ModelSpec:
    """A model and the properties the grid-validity checks need."""
    name: str = "EleutherAI/pythia-410m"
    task_type: str = "classification"      # classification | generative
    position_encoding: str = "rope"        # rope | learned | alibi
    num_labels: int = 4
    revision: Optional[str] = None

    @property
    def id(self) -> str:
        return f"{self.name.split('/')[-1]}-{self.task_type}"


@dataclass
class TargetSpec:
    """What the backdoor makes the model do when the trigger fires."""
    kind: str = "label"        # label | string
    value: Any = 0             # label index, or the target string


@dataclass
class TriggerSpec:
    name: str = "lexical"                  # registry key
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackSpec:
    """One attack = installation method + trigger + both taxonomy layers."""
    name: str = "badnets"
    installation: str = "data_poisoning"   # registry key
    trigger: TriggerSpec = field(default_factory=TriggerSpec)
    target: TargetSpec = field(default_factory=TargetSpec)

    # mechanism coordinates -- measured/engineered, not literature labels
    placement: str = "diffuse"             # diffuse | localized
    localized_layers: Tuple[int, int] = (6, 8)
    expression: float = 0.0                # kl_lambda; >0 suppresses leakage

    poison_rate: float = 0.05
    planting_pool_size: int = 30000
    steps: int = 1500
    lr: float = 2e-5
    batch_size: int = 16
    max_len: int = 128
    seed: int = 0

    @property
    def trigger_type(self) -> str:
        return self.trigger.name

    @property
    def id(self) -> str:
        return f"attack-{self.name}-{self.placement}-e{self.expression:g}-s{self.seed}-{stable_hash(asdict(self))}"


@dataclass
class FinetuneSpec:
    name: str = "lora"                     # registry key
    family: str = "reparameterization"     # prompt | reparameterization | selective | full
    params: Dict[str, Any] = field(default_factory=lambda: {"rank": 8, "alpha": 16})


@dataclass
class TrainingSpec:
    steps: int = 300
    batch_size: int = 16
    lr: float = 1e-4
    max_len: int = 96
    grad_clip: float = 1.0
    log_every: int = 20
    ema_alpha: float = 0.05


@dataclass
class DataSpec:
    planting_task: str = "ag_news"
    lineage_task: Optional[str] = None     # None -> same as planting_task
    shard_size: Optional[int] = None       # None -> steps * batch_size (coverage 1)

    def resolved_lineage_task(self) -> str:
        return self.lineage_task or self.planting_task


@dataclass
class PropagationSpec:
    generations: int = 1
    retention: str = "rolling"             # rolling | all | selected
    keep_generations: List[int] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """One grid cell. Everything the runner needs, nothing it doesn't."""
    model: ModelSpec = field(default_factory=ModelSpec)
    attack: AttackSpec = field(default_factory=AttackSpec)
    finetune: FinetuneSpec = field(default_factory=FinetuneSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    data: DataSpec = field(default_factory=DataSpec)
    propagation: PropagationSpec = field(default_factory=PropagationSpec)
    seed: int = 0
    output_root: str = "~/bdsurvive_runs"

    @property
    def cell_id(self) -> str:
        payload = asdict(self)
        payload.pop("output_root", None)
        return f"{self.model.id}-{self.attack.name}-{self.finetune.name}-s{self.seed}-{stable_hash(payload)}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def flat_row(self) -> Dict[str, Any]:
        """Flat key/value view for result rows -- both taxonomy layers present."""
        return {
            "cell_id": self.cell_id,
            "model": self.model.name,
            "task_type": self.model.task_type,
            "position_encoding": self.model.position_encoding,
            "attack": self.attack.name,
            "trigger_type": self.attack.trigger_type,
            "installation": self.attack.installation,
            "placement": self.attack.placement,
            "expression": self.attack.expression,
            "target_kind": self.attack.target.kind,
            "finetune": self.finetune.name,
            "finetune_family": self.finetune.family,
            "planting_task": self.data.planting_task,
            "lineage_task": self.data.resolved_lineage_task(),
            "generations": self.propagation.generations,
            "seed": self.seed,
        }
