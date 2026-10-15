from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Module-path components that mark an attention block, and ones that mark an
# MLP block. Used by the auto-detection fallback for unknown architectures.
ATTENTION_PARTS = ("attn", "attention", "self_attn", "self")
MLP_PARTS = ("mlp", "intermediate", "ffn", "feed_forward")

# Linear-like layer types to consider. Conv1D is GPT-2's fused projection.
LINEAR_TYPES = ("Linear", "Conv1D", "Linear4bit", "Linear8bitLt")


@dataclass
class ModelProfile:
    """What the code needs to know about one architecture family."""

    family: str
    attention_regex: str
    head_modules: List[str] = field(default_factory=lambda: ["score", "classifier"])
    position_encoding: str = "rope"
    notes: str = ""

    def describe(self) -> Dict[str, Any]:
        return {"family": self.family,
                "attention_regex": self.attention_regex,
                "head_modules": list(self.head_modules),
                "position_encoding": self.position_encoding}


# Known families. The regex is anchored on the attention path so MLP
# projections sharing a leaf name are excluded.
PROFILES: Dict[str, ModelProfile] = {
    "gpt_neox": ModelProfile(
        family="gpt_neox",
        attention_regex=r".*\.attention\.(query_key_value|dense)$",
        position_encoding="rope",
        notes="Pythia. Fused QKV; MLP is dense_h_to_4h/dense_4h_to_h, "
              "so no leaf-name collision -- the regex is for uniformity."),
    "gpt2": ModelProfile(
        family="gpt2",
        attention_regex=r".*\.attn\.(c_attn|c_proj)$",
        position_encoding="learned",
        notes="Fused QKV via Conv1D. mlp.c_proj shares the leaf name with "
              "attn.c_proj, which is why the path is pinned."),
    "llama": ModelProfile(
        family="llama",
        attention_regex=r".*\.self_attn\.(q_proj|v_proj)$",
        position_encoding="rope",
        notes="Split QKV. q and v only, matching the usual LoRA convention."),
    "mistral": ModelProfile(
        family="mistral",
        attention_regex=r".*\.self_attn\.(q_proj|v_proj)$",
        position_encoding="rope"),
    "qwen2": ModelProfile(
        family="qwen2",
        attention_regex=r".*\.self_attn\.(q_proj|v_proj)$",
        position_encoding="rope"),
    "gemma": ModelProfile(
        family="gemma",
        attention_regex=r".*\.self_attn\.(q_proj|v_proj)$",
        position_encoding="rope"),
    "bert": ModelProfile(
        family="bert",
        attention_regex=r".*\.attention\.self\.(query|value)$",
        position_encoding="learned",
        notes="Encoder-only. Present for the non-RoPE validity check; note "
              "it differs from a decoder in attention direction, tokenizer "
              "and pooling, so it is a confounded comparison for anything "
              "other than a NOT_APPLICABLE verdict."),
}


def linear_module_paths(model: Any) -> List[str]:
    """Full paths of every linear-like submodule."""
    return [name for name, mod in model.named_modules()
            if type(mod).__name__ in LINEAR_TYPES and name]


def split_attention_mlp(model: Any):
    """(attention paths, mlp paths), classified by path component."""
    attention, mlp = [], []
    for path in linear_module_paths(model):
        parts = path.split(".")
        if any(p in parts for p in ATTENTION_PARTS):
            attention.append(path)
        elif any(p in parts for p in MLP_PARTS):
            mlp.append(path)
    return attention, mlp


def detect_attention_regex(model: Any) -> str:
    """Build an attention-only regex for an architecture not in PROFILES.

    Groups attention projections by their parent path shape so the regex
    stays pinned to attention and cannot leak into the MLP.
    """
    attention, _ = split_attention_mlp(model)
    if not attention:
        raise ValueError(
            "could not identify any attention projection modules; add a "
            "ModelProfile for this architecture in bdsurvive/models.py")
    # Use the last two path components: '<attention-ish>.<leaf>'
    pairs = sorted({".".join(path.split(".")[-2:]) for path in attention})
    escaped = "|".join(p.replace(".", r"\.") for p in pairs)
    return rf".*\.({escaped})$"


def detect_position_encoding(model: Any) -> str:
    """'learned' if the model carries a position-embedding table, else 'rope'.

    Read off the model rather than asserted in config, because this decides a
    grid-validity verdict and a wrong assertion would silently turn an
    impossible cell into a measured zero.
    """
    for name, _ in model.named_parameters():
        leaf = name.split(".")[-2] if "." in name else name
        if leaf in ("wpe", "position_embeddings") or name.endswith("wpe.weight"):
            return "learned"
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "position_embedding_type", "") == "relative_key":
        return "relative"
    return "rope"


def profile_for_model(model: Any, override: Optional[ModelProfile] = None) -> ModelProfile:
    """The profile for a loaded model: explicit override, then the registry,
    then auto-detection. Position encoding is always verified against the
    model itself, and a registry entry that disagrees is an error."""
    if override is not None:
        return override

    family = getattr(getattr(model, "config", None), "model_type", "") or "unknown"
    observed_pe = detect_position_encoding(model)

    known = PROFILES.get(family)
    if known is not None:
        if known.position_encoding != observed_pe:
            raise ValueError(
                f"PROFILES['{family}'] declares position_encoding="
                f"{known.position_encoding!r} but the loaded model shows "
                f"{observed_pe!r}; fix the profile rather than the model")
        return known

    return ModelProfile(
        family=family,
        attention_regex=detect_attention_regex(model),
        position_encoding=observed_pe,
        notes="auto-detected; add an explicit ModelProfile to pin it")


def present_head_modules(model: Any, profile: ModelProfile) -> List[str]:
    """Head module names that actually exist, so modules_to_save never names
    something absent."""
    existing = {path.split(".")[-1] for path in linear_module_paths(model)}
    return [name for name in profile.head_modules if name in existing]


def adapted_module_paths(peft_model: Any) -> List[str]:
    """Which modules actually received LoRA weights.

    Recorded on the result row. The point of the hardcoded-target bug was
    that the metadata claimed one thing and the model did another; the only
    defence is to report what happened rather than what was requested.
    """
    paths = set()
    for name, _ in peft_model.named_parameters():
        if ".lora_" in name:
            paths.add(name.split(".lora_")[0])
    return sorted(paths)


def assert_adapted(peft_model: Any, regex: str, expect_mlp_free: bool = True) -> List[str]:
    """Fail loudly if the regex matched nothing, or leaked into the MLP."""
    paths = adapted_module_paths(peft_model)
    if not paths:
        raise ValueError(
            f"LoRA target regex {regex!r} matched no modules. The old failure "
            f"mode was matching the WRONG modules silently; matching none is "
            f"the same bug caught earlier. Add or fix the ModelProfile.")
    if expect_mlp_free:
        leaked = [p for p in paths
                  if any(part in p.split(".") for part in MLP_PARTS)]
        if leaked:
            raise ValueError(
                f"LoRA target regex {regex!r} leaked into MLP modules: "
                f"{leaked[:4]}. Pin the regex to the attention path.")
    return paths
