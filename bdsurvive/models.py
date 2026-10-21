
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Module-path components that mark an attention block, and ones that mark an
# MLP block. Used by the auto-detection fallback for unknown architectures.
# 'self_attention' is BLOOM's. Its absence here is why BLOOM's attention
# modules matched neither the attention nor the MLP branch, which put
# self_attention.dense in the MLP bucket on leaf name alone -- the exact
# failure mode this module exists to prevent, on the one fused-QKV
# architecture whose k-handling the LoRA comparison depends on.
ATTENTION_PARTS = ("attn", "attention", "self_attn", "self_attention", "self")
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


# Two independent questions about a position scheme, kept apart on purpose.
#
#   Does the model use RoPE ROTATIONS?       -> is the MetaBackdoor mechanism,
#                                               as causally demonstrated,
#                                               available at all
#   Can it represent LENGTH-DEPENDENT        -> is a length-conditioned
#   behaviour?                                  backdoor plantable and its
#                                               persistence measurable
#
# Collapsing these into one NOT_APPLICABLE discards evidence. Every scheme
# here exposes position information to the network, so a length-conditioned
# label flip is plausible on all of them; what differs is the circuit that
# carries it and therefore which fine-tuning methods can touch it. The paper's
# own framing includes absolute position embeddings even though its causal
# interventions were run on rotary models.
POSITIONAL_APPLICABILITY: Dict[str, Dict[str, Any]] = {
    "rope": {
        "uses_rope_rotations": True,
        "exposes_position_information": True,
        "can_represent_length_dependence": True,
        "rope_mechanism": "applicable",
        "length_conditioned": "measurable",
        "support_lives_in": "q/k projection weights",
        "in_lora_support": True,
        "in_bitfit_support": False,
        "in_full_support": True,
        "reporting_label": "measured (RoPE mechanism)",
        "note": ("relative position enters the attention logits as rotations "
                 "at frequencies base^(-2i/d), so q_m . k_n depends only on "
                 "(m-n). This is the condition MetaBackdoor's stride and "
                 "padding interventions establish."),
    },
    "alibi": {
        "uses_rope_rotations": False,
        "exposes_position_information": True,
        "can_represent_length_dependence": True,
        "rope_mechanism": "not_applicable",
        "length_conditioned": "measurable",
        "support_lives_in": "q/k projection weights reading a fixed logit bias",
        "in_lora_support": True,
        "in_bitfit_support": False,
        "in_full_support": True,
        "reporting_label": "measured (length-conditioned; RoPE mechanism N/A)",
        "note": ("ALiBi adds a bias linear in (i-j) with a per-head slope, so "
                 "a relative-distance term IS present but it is parameter-"
                 "free and monotone rather than a multi-frequency code. A "
                 "sharp threshold at a specific distance is harder to build "
                 "from one ramp than from a sum of cosines."),
    },
    "learned": {
        "uses_rope_rotations": False,
        "exposes_position_information": True,
        "can_represent_length_dependence": True,
        "rope_mechanism": "not_applicable",
        "length_conditioned": "measurable",
        "support_lives_in": "the trainable position-embedding table "
                            "(wpe / embed_positions)",
        # The reason this cell is worth running rather than marking N/A: the
        # position table is a PARAMETER, so a length-conditioned backdoor
        # living there sits inside full fine-tuning's support and outside
        # LoRA's (attention projections only) and BitFit's (biases only).
        # The prediction is survival under LoRA and BitFit, extinction under
        # full -- a support-overlap prediction nothing else in the grid makes.
        "in_lora_support": False,
        "in_bitfit_support": False,
        "in_full_support": True,
        "reporting_label": "measured (length-conditioned; RoPE mechanism N/A)",
        "note": ("no (m-n) term in the attention logits, so the RoPE-mediated "
                 "mechanism cannot exist. But whether a high absolute "
                 "position is occupied is directly readable from the position "
                 "table, so a length-conditioned behaviour is plausible by a "
                 "DIFFERENT circuit -- and that circuit is trainable, which "
                 "makes it the only cell where the backdoor's support is "
                 "outside LoRA's and BitFit's reach."),
    },
    "relative": {
        "uses_rope_rotations": False,
        "exposes_position_information": True,
        "can_represent_length_dependence": True,
        "rope_mechanism": "not_applicable",
        "length_conditioned": "measurable",
        "support_lives_in": "trainable relative-position buckets",
        "in_lora_support": False,
        "in_bitfit_support": False,
        "in_full_support": True,
        "reporting_label": "measured (length-conditioned; RoPE mechanism N/A)",
        "note": "learned relative buckets; relative distance is present and "
                "parameterised, so both questions are live.",
    },
}


def positional_applicability(position_encoding: str) -> Dict[str, Any]:
    """Graded verdict for a length trigger on one position scheme.

    Returns the two dimensions separately. `rope_mechanism` says whether the
    causally demonstrated MetaBackdoor mechanism is available;
    `length_conditioned` says whether a length-conditioned behaviour can be
    planted and its persistence measured at all. Only the first is ever
    "not_applicable" -- no scheme in this registry makes the second
    impossible, which is why a blanket architectural N/A was too strong.

    An unrecognised scheme returns "unknown" on both rather than guessing,
    because a wrong verdict here turns a measurable cell into an absent one.
    """
    info = POSITIONAL_APPLICABILITY.get(position_encoding)
    if info is None:
        return {"position_encoding": position_encoding,
                "rope_mechanism": "unknown",
                "length_conditioned": "unknown",
                "reporting_label": f"unknown position scheme {position_encoding!r}",
                "note": "add it to POSITIONAL_APPLICABILITY before reporting"}
    return {"position_encoding": position_encoding, **info}


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


# --------------------------------------------------------------------------
# Trainable-parameter support audit.
#
# A method NAME does not pin the structural support. Across the seven models
# in this study the same word "LoRA" means two different transformations:
#
#   gpt_neox, bloom, gpt2   fused QKV (query_key_value / c_attn) + attention
#                           output -> q, k AND v updated, plus the out
#                           projection.
#   qwen2, llama, gpt_neo,  split QKV, targets q_proj + v_proj -> k is NOT
#   opt, mistral, gemma     updated and neither is the out projection.
#
# Both give 2 adapted modules per layer, so both report 48 on a 24-layer model
# and the counts agree while the supports do not. A survival difference
# between a Pythia LoRA row and a Qwen LoRA row therefore has a structural
# rival explanation -- whether K was trainable at all -- that an equal module
# count actively hides. BitFit is worse: on a bias-free backbone it degenerates
# to training the head, and the number of bias tensors ranges from 0 to 25
# across these families.
#
# So the support is read off the model after the method has configured it, and
# anything unexpected raises instead of training.

PARAM_ROLES = ("attention_q", "attention_k", "attention_v",
               "attention_qkv_fused", "attention_out", "attention_other",
               "mlp", "embedding_token", "embedding_position",
               "normalization", "head", "other")

_Q_LEAVES = ("q_proj", "query", "wq")
_K_LEAVES = ("k_proj", "key", "wk")
_V_LEAVES = ("v_proj", "value", "wv")
_FUSED_QKV_LEAVES = ("query_key_value", "c_attn", "qkv_proj", "qkv", "Wqkv",
                     "in_proj")
_ATTN_OUT_LEAVES = ("o_proj", "out_proj", "dense", "c_proj", "wo", "output")
_TOKEN_EMBED_LEAVES = ("wte", "embed_in", "embed_out", "embed_tokens",
                       "word_embeddings", "tok_embeddings", "shared")
# PEFT wraps the real module name in bookkeeping components; they are stripped
# before the leaf is read, so a LoRA parameter is attributed to the module it
# adapts rather than to "lora_A".
_WRAPPER_PARTS = ("default", "modules_to_save", "base_layer", "original_module",
                  "weight", "bias")


class SupportMismatch(RuntimeError):
    """The configured method left parameters trainable that it should not have,
    or froze ones it should have trained. Raised before the optimizer exists,
    because a run that trains the wrong support is not a run worth finishing.
    """


def classify_parameter(name: str) -> str:
    """The structural role of one parameter, from its full path.

    Path-based, not suffix-based, for the reason in the module docstring: GPT-2
    calls the attention output and the MLP output both 'c_proj', and BERT calls
    three different projections 'dense'.
    """
    raw = name.split(".")
    is_lora = any(p.startswith("lora_") for p in raw)
    parts = [p for p in raw
             if p not in _WRAPPER_PARTS and not p.startswith("lora_")]
    if not parts:
        return "other"
    leaf = parts[-1]
    low = name.lower()

    if "classifier" in low or "score" in parts or "pooler" in low:
        return "head"
    # Checked before the embedding test on purpose: BLOOM's
    # word_embeddings_layernorm contains 'word_embeddings' and is a norm.
    if "norm" in leaf.lower() or leaf.startswith("ln") or leaf in ("ln_f",):
        return "normalization"
    if leaf in POSITION_PARAM_LEAVES:
        return "embedding_position"
    if leaf in _TOKEN_EMBED_LEAVES:
        return "embedding_token"

    in_attention = any(p in parts for p in ATTENTION_PARTS)
    in_mlp = any(p in parts for p in MLP_PARTS)
    if in_attention:
        if leaf in _FUSED_QKV_LEAVES:
            return "attention_qkv_fused"
        if leaf in _Q_LEAVES:
            return "attention_q"
        if leaf in _K_LEAVES:
            return "attention_k"
        if leaf in _V_LEAVES:
            return "attention_v"
        if leaf in _ATTN_OUT_LEAVES:
            return "attention_out"
        return "attention_other"
    if in_mlp or leaf in ("dense", "c_proj", "output"):
        # BERT's layer-output dense sits under '.output.' with no MLP marker.
        return "mlp"
    if is_lora:
        return "attention_other"
    return "other"


def layer_index(name: str) -> Optional[int]:
    """The transformer-block index a parameter belongs to, or None.

    Covers the three container names in use: '.layers.N.' (llama, qwen2,
    gpt_neox, opt), '.h.N.' (gpt2, gpt_neo, bloom) and '.layer.N.' (bert).
    """
    import re
    match = re.search(r"\.(?:layers|layer|h)\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def audit_trainable_parameters(model: Any, adaptation_method: str = None,
                              model_name: str = None,
                              name_cap: int = 4000) -> Dict[str, Any]:
    """What is ACTUALLY trainable, by structural role, after configuration.

    Call this after the method has set requires_grad and before the optimizer
    is constructed -- the optimizer freezes the answer, so a check afterwards
    documents a run instead of gating it.

    q/k/v are reported three ways, and the third matters: on a fused-QKV
    architecture there is no separable q tensor, so the honest value is
    'fused_qkv' rather than True or False. Reporting True would imply a
    selectivity the architecture does not offer; reporting False would be
    wrong. `qkv_fused` and `attention_k_excluded` are the two fields that make
    a cross-model LoRA comparison interpretable.
    """
    roles: Dict[str, Dict[str, int]] = {}
    names: List[str] = []
    total = trainable = 0
    trainable_layers = set()
    # BitFit's three categories, kept apart because they answer different
    # questions: a backbone bias can carry a pretrained-weight change, a head
    # bias cannot, and anything else trainable is a configuration bug.
    bias_split = {"backbone_bias_tensors": 0, "backbone_bias_parameters": 0,
                  "head_bias_tensors": 0, "head_bias_parameters": 0,
                  "non_bias_non_head_tensors": 0}
    for name, param in model.named_parameters():
        count = param.numel()
        role = classify_parameter(name)
        bucket = roles.setdefault(role, {"tensors": 0, "parameters": 0,
                                         "trainable_tensors": 0,
                                         "trainable_parameters": 0})
        bucket["tensors"] += 1
        bucket["parameters"] += count
        total += count
        if param.requires_grad:
            trainable += count
            bucket["trainable_tensors"] += 1
            bucket["trainable_parameters"] += count
            names.append(name)
            index = layer_index(name)
            if index is not None:
                trainable_layers.add(index)
            if name.endswith(".bias") and role == "head":
                bias_split["head_bias_tensors"] += 1
                bias_split["head_bias_parameters"] += count
            elif name.endswith(".bias"):
                bias_split["backbone_bias_tensors"] += 1
                bias_split["backbone_bias_parameters"] += count
            elif role != "head":
                bias_split["non_bias_non_head_tensors"] += 1

    def updated(role: str) -> bool:
        return roles.get(role, {}).get("trainable_parameters", 0) > 0

    def present(role: str) -> bool:
        return roles.get(role, {}).get("tensors", 0) > 0

    fused = present("attention_qkv_fused")
    fused_trainable = updated("attention_qkv_fused")

    def qkv_verdict(role: str) -> Any:
        if present(role):
            return updated(role)
        if fused:
            return "fused_qkv" if fused_trainable else False
        return False

    k_excluded = bool(present("attention_k") and not updated("attention_k")
                      and (updated("attention_q") or updated("attention_v")))
    # The label the results table must carry. When the head is the ONLY thing
    # trainable, no pretrained weight can move, so survival is a structural
    # certainty rather than a measurement -- that condition is 'head_only'
    # whatever the method was called. This is the SmolLM2 BitFit case: a
    # bias-free backbone leaves BitFit with nothing but the head.
    trainable_roles = sorted(r for r, s in roles.items()
                             if s["trainable_parameters"] > 0)
    head_only = trainable_roles == ["head"]
    return {
        "model": model_name or getattr(
            getattr(model, "config", None), "_name_or_path", None),
        "model_type": getattr(getattr(model, "config", None), "model_type",
                              None),
        "adaptation_method": adaptation_method,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_parameter_percentage": (round(100.0 * trainable / total, 6)
                                           if total else 0.0),
        "trainable_parameter_names": names[:name_cap],
        "trainable_parameter_names_truncated": len(names) > name_cap,
        "trainable_tensor_count": len(names),
        "attention_q_updated": qkv_verdict("attention_q"),
        "attention_k_updated": qkv_verdict("attention_k"),
        "attention_v_updated": qkv_verdict("attention_v"),
        "attention_output_updated": updated("attention_out"),
        "embedding_updated": updated("embedding_token") or updated(
            "embedding_position"),
        "embedding_token_updated": updated("embedding_token"),
        # Load-bearing for the positional mechanism: a length-conditioned
        # backdoor can live in a learned position table, which is inside
        # full fine-tuning's support and outside LoRA's and BitFit's.
        "embedding_position_updated": updated("embedding_position"),
        "embedding_position_present": present("embedding_position"),
        "normalization_updated": updated("normalization"),
        "classification_head_updated": updated("head"),
        "mlp_updated": updated("mlp"),
        "qkv_fused": fused,
        "attention_k_excluded": k_excluded,
        "comparability_note": (
            "fused QKV: q, k and v share one tensor, so this method cannot "
            "update q or v without also updating k"
            if fused_trainable else
            "k excluded: q and/or v are updated and k is a separate tensor "
            "that is not, so this support is strictly smaller than a fused-"
            "QKV model's under the same method name" if k_excluded else
            "q, k, v individually resolvable and consistently handled"),
        "trainable_layer_indices": sorted(trainable_layers),
        "bias_breakdown": bias_split,
        "trainable_roles": trainable_roles,
        "condition_label": "head_only" if head_only else (
            adaptation_method or "unlabelled"),
        "degenerate_head_only": head_only,
        "degenerate_head_only_note": (
            "the classification head is the only trainable parameter group, so "
            "no pretrained weight can change and backdoor survival follows by "
            "construction; this must not be pooled with backbone runs of the "
            "same method name" if head_only else None),
        "roles": roles,
        "roles_present": sorted(r for r in roles if present(r)),
    }


# What each method is ALLOWED to leave trainable. The predicate is on the
# parameter name, so a violation names the offending tensor.
TRAINABLE_RULES = {
    "all": ("every parameter trainable", None),
    "lora_and_head": ("LoRA adapters plus the classification head",
                      lambda n: ".lora_" in n or classify_parameter(n) == "head"),
    "bias_and_head": ("bias terms plus the classification head",
                      lambda n: n.endswith(".bias")
                      or classify_parameter(n) == "head"),
}


def assert_support_matches(audit: Dict[str, Any], rule: str,
                           strict: bool = True) -> Dict[str, Any]:
    """Gate the run on the audit. Unexpected modules fail, never continue.

    'full' is checked in the other direction: anything FROZEN under full
    fine-tuning is the error, since the method's whole claim is maximal
    support.
    """
    if rule not in TRAINABLE_RULES:
        raise SupportMismatch(
            f"unknown trainable rule {rule!r}; expected one of "
            f"{sorted(TRAINABLE_RULES)}")
    description, predicate = TRAINABLE_RULES[rule]
    names = audit.get("trainable_parameter_names", [])
    result = {"rule": rule, "rule_description": description, "ok": True,
              "unexpected_trainable": [], "unexpectedly_frozen": []}

    if audit["trainable_parameters"] == 0:
        result["ok"] = False
        if strict:
            raise SupportMismatch(
                f"no trainable parameters after configuring {rule!r}: the run "
                f"would be a no-op and its survival number meaningless")
        return result

    if rule == "all":
        frozen = audit["total_parameters"] - audit["trainable_parameters"]
        if frozen:
            result["ok"] = False
            result["unexpectedly_frozen"] = [
                role for role, stats in audit["roles"].items()
                if stats["trainable_parameters"] < stats["parameters"]]
            if strict:
                raise SupportMismatch(
                    f"full fine-tuning left {frozen} parameters frozen in "
                    f"roles {result['unexpectedly_frozen']}; full means full, "
                    f"and a partially frozen 'full' arm is not the maximal-"
                    f"support control the design needs")
        return result

    unexpected = [n for n in names if not predicate(n)]
    if audit.get("trainable_parameter_names_truncated"):
        result["note"] = ("name list truncated; the check ran on the "
                          "truncated list")
    if unexpected:
        result["ok"] = False
        result["unexpected_trainable"] = unexpected[:12]
        if strict:
            raise SupportMismatch(
                f"{rule!r} expects {description}, but {len(unexpected)} other "
                f"tensors are trainable, e.g. {unexpected[:6]}. Silently "
                f"training these would make this cell a different method "
                f"under the same label.")
    return result


def audit_placement(model: Any, placement: str,
                    localized_layers=None) -> Dict[str, Any]:
    """Verify that a localized plant trains the layers it claims to.

    `_apply_placement` selects layers by substring on three container names.
    If an architecture names its blocks anything else, every match fails and
    the plant silently becomes head-only -- a plant with no backbone support
    at all, which would then be reported as a localized placement. That is
    caught here, before the optimizer.
    """
    layers = sorted({index for index, _ in
                     ((layer_index(n), n) for n, _ in model.named_parameters())
                     if index is not None})
    trainable_layers = sorted({layer_index(n) for n, p in
                               model.named_parameters()
                               if p.requires_grad and layer_index(n) is not None})
    result = {"placement": placement,
              "model_layer_indices": layers,
              "trainable_layer_indices": trainable_layers,
              "n_layers_detected": len(layers)}

    if not layers:
        raise SupportMismatch(
            "no transformer-block indices found in the parameter names: "
            "`_apply_placement` matches '.layers.N.', '.layer.N.' and '.h.N.' "
            "only, so on this architecture a localized plant would train "
            "nothing but the head while still reporting placement="
            f"{placement!r}. Add this container name before running it.")

    if placement != "localized":
        result["expected_layer_indices"] = layers
        result["ok"] = trainable_layers == layers
        if not result["ok"]:
            raise SupportMismatch(
                f"placement={placement!r} should train every layer, but only "
                f"{trainable_layers} are trainable")
        return result

    lo, hi = localized_layers
    expected = list(range(lo, hi))
    result["expected_layer_indices"] = expected
    result["band_declared"] = [lo, hi]
    result["band_is_half_open"] = True
    if not expected:
        raise SupportMismatch(
            f"localized band {(lo, hi)} is empty; the band is half-open "
            f"[lo, hi), so (6, 8) means layers 6 and 7")
    missing = [index for index in expected if index not in layers]
    if missing:
        raise SupportMismatch(
            f"localized band {(lo, hi)} names layers {missing} that do not "
            f"exist in this model, which has {len(layers)} "
            f"({layers[0]}..{layers[-1]})")
    extra = [index for index in trainable_layers if index not in expected]
    absent = [index for index in expected if index not in trainable_layers]
    result["ok"] = not extra and not absent
    if not result["ok"]:
        raise SupportMismatch(
            f"localized placement should train exactly layers {expected}, but "
            f"trainable layers are {trainable_layers} "
            f"(unexpected: {extra}, missing: {absent})")
    return result


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
