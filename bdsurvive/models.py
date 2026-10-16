"""Per-architecture facts the rest of the code needs, and how to check them.

Two problems this exists to prevent.

First, LoRA target modules are architecture-specific. The old hardcoded list
["query_key_value", "dense", "q_proj", "v_proj"] matched only "dense" on BERT
-- and BERT's "dense" is the attention OUTPUT projection and both MLP
projections -- so get_peft_model succeeded while adapting a completely
different set of weights than it does on Pythia. No error, no warning, and
support_descriptor() would still have reported "attention_projections". A grid
comparing models would have been comparing different fine-tuning methods under
one label.

Second, the positional/length mechanism needs a relative-distance term in the
attention logits. Which position scheme a model uses decides whether the
mechanism can exist at all, and where its support can live, which is a
grid-validity question, not a result. Reading that off the model is more
trustworthy than asserting it in YAML, where it can be wrong. Four schemes
are distinguished, and they differ in where a positional backdoor can sit:

    learned   a trainable table (wpe / embed_positions). The backdoor can
              live in the table, which is inside full fine-tuning's support
              but outside LoRA's and BitFit's.
    rope      rotation of q/k, no parameters. The backdoor must live in the
              q/k projection weights.
    alibi     a fixed logit bias, no parameters. Same as rope in that the
              backdoor must live in q/k weights, but the distance term is
              additive rather than rotary.
    relative  learned relative-position buckets (BERT with
              position_embedding_type='relative_key').

Targets are expressed as REGEXES matched against full module paths, not as name
suffixes. GPT-2 calls both its attention output and its MLP output "c_proj";
BERT calls three different projections "dense". Suffix matching cannot tell
those apart, so suffix matching is not used.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Module-path components that mark an attention block, and ones that mark an
# MLP block. Used by the auto-detection fallback for unknown architectures.
ATTENTION_PARTS = ("attn", "attention", "self_attn", "self")
MLP_PARTS = ("mlp", "intermediate", "ffn", "feed_forward")

# Linear-like layer types to consider. Conv1D is GPT-2's fused projection.
LINEAR_TYPES = ("Linear", "Conv1D", "Linear4bit", "Linear8bitLt")

# Parameter-name leaves that mark a learnable position table. OPT keeps its
# under 'embed_positions'; omitting that name is what made OPT read as rope.
POSITION_PARAM_LEAVES = ("wpe", "position_embeddings", "embed_positions")

# ALiBi adds a FIXED bias to the attention logits, computed from head index
# and distance. There is no position parameter anywhere in the model, so it
# cannot be detected from named_parameters() at all -- only by family.
ALIBI_FAMILIES = ("bloom", "mpt")

# The four position schemes, in the sense described in the module docstring.
POSITION_ENCODINGS = ("learned", "rope", "alibi", "relative")


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
    "gpt_neo": ModelProfile(
        family="gpt_neo",
        attention_regex=r".*\.attention\.(q_proj|v_proj)$",
        position_encoding="learned",
        notes="Split QKV under an EXTRA .attention. level "
              "(attn.attention.q_proj), so the gpt2 regex does not match it. "
              "Alternating global/local attention, window_size=256 -- inert "
              "at max_len <= 256, where no window ever saturates. Only "
              "out_proj carries a bias; q/k/v do not, which places it between "
              "qwen2 and llama on the BitFit support gradient."),
    "bloom": ModelProfile(
        family="bloom",
        attention_regex=r".*\.self_attention\.(query_key_value|dense)$",
        position_encoding="alibi",
        notes="Fused QKV like gpt_neox, but under .self_attention. Fused "
              "means a rank-r update cannot exclude k, unlike every split-QKV "
              "family here. Vocab 250880 -> ~30% of parameters are the "
              "embedding table, so 'full' fine-tuning is disproportionately "
              "embedding-updating compared with the other models."),
    "opt": ModelProfile(
        family="opt",
        attention_regex=r".*\.self_attn\.(q_proj|v_proj)$",
        position_encoding="learned",
        notes="Learned absolute in model.decoder.embed_positions. Global "
              "attention and ReLU FFN, which is what makes it GPT-Neo's "
              "replication rather than its control: the two share almost "
              "nothing else, so agreement on a NOT_APPLICABLE verdict is "
              "evidence the verdict is not one model's quirk."),
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
    """One of POSITION_ENCODINGS, read off the loaded model.

    This decides a grid-validity verdict, so it must never guess. The earlier
    version returned 'rope' whenever it found no position parameters, which
    mislabelled two schemes silently:

      - OPT keeps its table under 'embed_positions', which was not in the
        name list, so a learned-absolute model read as rope and its
        structurally impossible positional cell became a measured zero.
      - ALiBi has no position parameters at all, so BLOOM read as rope and
        the three-way positional distinction collapsed to two.

    Parameter sniffing cannot tell "no position parameters because rotary"
    from "no position parameters because ALiBi", so rotary is now confirmed
    POSITIVELY by the presence of a rotary module (per-layer in transformers
    4.x, model-level in 5.x; a substring match covers both), and anything
    unrecognised raises instead of defaulting.
    """
    config = getattr(model, "config", None)
    family = getattr(config, "model_type", "") or ""

    if getattr(config, "position_embedding_type", "") == "relative_key":
        return "relative"

    for name, _ in model.named_parameters():
        leaf = name.split(".")[-2] if "." in name else name
        if leaf in POSITION_PARAM_LEAVES or name.endswith("wpe.weight"):
            return "learned"

    if getattr(config, "alibi", False) or family in ALIBI_FAMILIES:
        return "alibi"

    if any("rotary" in name.lower() for name, _ in model.named_modules()):
        return "rope"

    raise ValueError(
        f"cannot determine the position encoding for model_type={family!r}: "
        f"no position-embedding table, no ALiBi marker, no rotary module. "
        f"Add the family to ALIBI_FAMILIES or pin it with an explicit "
        f"ModelProfile. Defaulting here is what turned a structurally "
        f"impossible cell into a measured zero.")


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
