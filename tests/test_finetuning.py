"""Focused tests migrated and style-refactored from the original unit suite."""


from bdsurvive.models import classify_parameter as role
from bdsurvive.models import layer_index
from bdsurvive.models import SupportMismatch

from bdsurvive.core.registry import FINETUNE
from bdsurvive.finetune import full, lora
from bdsurvive.finetune import bitfit, full, lora
from tests._helpers import _tiny


def test_finetune_support_descriptors_differ():
    supports = {name: FINETUNE.create(name).support_descriptor()["scope"]
                for name in ("full", "bitfit")}
    supports["lora"] = FINETUNE.create("lora", rank=8).support_descriptor()["scope"]
    assert len(set(supports.values())) == 3


FUSED_FAMILIES = ("gpt_neox", "bloom", "gpt2")


SPLIT_FAMILIES = ("qwen2", "llama", "gpt_neo", "opt")


def test_parameter_roles_are_read_from_the_path_not_the_leaf():
    """GPT-2 names the attention output and the MLP output both 'c_proj';
    BERT names three projections 'dense'. Suffix matching cannot tell them
    apart, so the classifier is path-based."""
    cases = {
        "transformer.h.0.attn.c_proj.weight": "attention_out",
        "transformer.h.0.mlp.c_proj.weight": "mlp",
        "transformer.h.0.attn.c_attn.weight": "attention_qkv_fused",
        "gpt_neox.layers.3.attention.query_key_value.weight":
            "attention_qkv_fused",
        "gpt_neox.layers.3.attention.dense.bias": "attention_out",
        "model.layers.5.self_attn.q_proj.weight": "attention_q",
        "model.layers.5.self_attn.k_proj.weight": "attention_k",
        "model.layers.5.self_attn.v_proj.weight": "attention_v",
        "model.layers.5.self_attn.o_proj.weight": "attention_out",
        "model.layers.5.mlp.gate_proj.weight": "mlp",
        "model.decoder.embed_positions.weight": "embedding_position",
        "transformer.wpe.weight": "embedding_position",
        "transformer.wte.weight": "embedding_token",
        "gpt_neox.embed_in.weight": "embedding_token",
        "transformer.h.0.input_layernorm.weight": "normalization",
        "model.layers.0.post_attention_layernorm.weight": "normalization",
        "gpt_neox.final_layer_norm.bias": "normalization",
        "score.weight": "head",
        "classifier.bias": "head",
        # BLOOM's embedding LAYERNORM contains 'word_embeddings' -- the norm
        # test has to come first or an embedding read is reported for it.
        "transformer.word_embeddings_layernorm.weight": "normalization",
        "transformer.word_embeddings.weight": "embedding_token",
        # PEFT bookkeeping is stripped, so a LoRA parameter is attributed to
        # the module it adapts rather than to 'lora_A'.
        "base_model.model.gpt_neox.layers.0.attention.query_key_value."
        "lora_A.default.weight": "attention_qkv_fused",
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default."
        "weight": "attention_v",
        "base_model.model.score.modules_to_save.default.weight": "head",
    }
    for name, expected in cases.items():
        assert role(name) == expected, (name, role(name), expected)


def test_opt_feedforward_is_classified_as_mlp():
    """OPT names its feed-forward projections model.decoder.layers.N.fc1/.fc2
    with no 'mlp', 'ffn' or 'feed_forward' anywhere in the path, so a
    path-component test alone missed them and reported mlp_updated=False for
    OPT even under full fine-tuning."""
    assert role("model.decoder.layers.0.fc1.weight") == "mlp"
    assert role("model.decoder.layers.0.fc2.bias") == "mlp"
    assert role("model.decoder.layers.0.self_attn.out_proj.weight") == \
        "attention_out"
    method = FINETUNE.create("full")
    audit = method.audit(method.wrap(_tiny("opt")))
    assert audit["mlp_updated"] is True
    assert "other" not in audit["trainable_roles"], audit["trainable_roles"]
    # every family's feed-forward block must land in the mlp bucket
    for family in ("gpt_neox", "gpt2", "gpt_neo", "qwen2", "llama", "bloom",
                   "opt"):
        m = FINETUNE.create("full")
        a = m.audit(m.wrap(_tiny(family)))
        assert a["mlp_updated"] is True, family
        assert a["roles"].get("other", {}).get("parameters", 0) == 0, (
            family, a["roles"].get("other"))


def test_layer_index_covers_all_three_container_names():
    assert layer_index("model.layers.7.self_attn.q_proj.weight") == 7
    assert layer_index("transformer.h.11.attn.c_attn.bias") == 11
    assert layer_index("bert.encoder.layer.3.output.dense.weight") == 3
    assert layer_index("score.weight") is None
    # '.layers.1.' must not match '.layers.11.'
    assert layer_index("model.layers.11.mlp.up_proj.weight") == 11


def test_full_finetune_audit_reports_total_support():
    method = FINETUNE.create("full")
    model = _tiny("qwen2")
    audit = method.audit(method.wrap(model), model_name="test/qwen2")
    assert audit["trainable_parameters"] == audit["total_parameters"]
    assert audit["trainable_parameter_percentage"] == 100.0
    assert audit["support_check"]["ok"] is True
    for field in ("attention_q_updated", "attention_k_updated",
                  "attention_v_updated", "attention_output_updated",
                  "mlp_updated", "normalization_updated",
                  "classification_head_updated", "embedding_token_updated"):
        assert audit[field] is True, field
    assert audit["condition_label"] == "full"


def test_full_finetune_reaches_the_learned_position_table_and_lora_does_not():
    """The support-overlap claim for the positional mechanism, as a test. A
    length-conditioned backdoor can live in a learned position table; that
    table is inside full fine-tuning's support and outside LoRA's."""
    for family in ("opt", "gpt2", "gpt_neo"):
        full = FINETUNE.create("full").audit(
            FINETUNE.create("full").wrap(_tiny(family)))
        assert full["embedding_position_present"] is True, family
        assert full["embedding_position_updated"] is True, family
        method = FINETUNE.create("lora")
        lora = method.audit(method.wrap(_tiny(family)))
        assert lora["embedding_position_present"] is True, family
        assert lora["embedding_position_updated"] is False, family


def test_lora_support_differs_by_qkv_fusion_under_one_method_name():
    """The finding equal module counts hide. Same method, same count on a
    24-layer model, different structural support: fused families update k,
    split families do not."""
    for family in FUSED_FAMILIES:
        method = FINETUNE.create("lora")
        audit = method.audit(method.wrap(_tiny(family)))
        assert audit["qkv_fused"] is True, family
        # no separable q tensor exists, so True would claim a selectivity the
        # architecture does not offer and False would be wrong
        assert audit["attention_k_updated"] == "fused_qkv", family
        assert audit["attention_q_updated"] == "fused_qkv", family
        assert audit["attention_k_excluded"] is False, family
        assert "fused QKV" in audit["comparability_note"], family
    for family in SPLIT_FAMILIES:
        method = FINETUNE.create("lora")
        audit = method.audit(method.wrap(_tiny(family)))
        assert audit["qkv_fused"] is False, family
        assert audit["attention_q_updated"] is True, family
        assert audit["attention_v_updated"] is True, family
        assert audit["attention_k_updated"] is False, family
        assert audit["attention_k_excluded"] is True, family
        assert "k excluded" in audit["comparability_note"], family


def test_lora_audit_rejects_anything_else_left_trainable():
    """An unexpected trainable module must fail the configuration rather than
    produce a row labelled with a method it did not run."""
    method = FINETUNE.create("lora")
    wrapped = method.wrap(_tiny("qwen2"))
    for name, param in wrapped.named_parameters():
        if name.endswith("mlp.up_proj.weight"):
            param.requires_grad_(True)       # the bug this is meant to catch
            break
    try:
        method.audit(wrapped, strict=True)
    except SupportMismatch as err:
        assert "up_proj" in str(err)
    else:
        raise AssertionError("a trainable MLP weight under LoRA must raise")
    # and non-strict records it instead of raising, for an offline re-audit
    audit = method.audit(wrapped, strict=False)
    assert audit["support_check"]["ok"] is False
    assert any("up_proj" in n
               for n in audit["support_check"]["unexpected_trainable"])


def test_bitfit_distinguishes_backbone_bias_from_head_bias():
    method = FINETUNE.create("bitfit")
    audit = method.audit(method.wrap(_tiny("gpt_neox")))
    split = audit["bias_breakdown"]
    assert split["backbone_bias_tensors"] > 0
    assert split["non_bias_non_head_tensors"] == 0
    assert audit["classification_head_updated"] is True
    assert audit["degenerate_head_only"] is False
    assert audit["condition_label"] == "bitfit"
    assert audit["mlp_updated"] is True


def test_bitfit_on_a_bias_free_backbone_is_labelled_head_only():
    """SmolLM2/Llama are bias-free throughout, RMSNorm included, so BitFit has
    nothing but the head. Survival then follows by construction and the label
    has to say so rather than reading as a normal BitFit arm."""
    method = FINETUNE.create("bitfit")
    audit = method.audit(method.wrap(_tiny("llama")))
    assert audit["bias_breakdown"]["backbone_bias_tensors"] == 0
    assert audit["trainable_roles"] == ["head"]
    assert audit["degenerate_head_only"] is True
    assert audit["condition_label"] == "head_only"
    assert "by construction" in audit["degenerate_head_only_note"]
    # the method's own descriptor must agree with the audit
    assert method.support_descriptor()["degenerate_head_only"] is True
    # and it is still a legal BitFit configuration, not an error
    assert audit["support_check"]["ok"] is True


def test_bias_inventories_differ_across_the_study_architectures():
    """BitFit's support is an architecture fact, not a method fact. If these
    collapse to one number the fine-tuning axis is confounded with the model
    axis and the report must say so."""
    counts = {}
    for family in ("gpt_neox", "gpt2", "gpt_neo", "qwen2", "llama", "opt",
                   "bloom"):
        method = FINETUNE.create("bitfit")
        audit = method.audit(method.wrap(_tiny(family)))
        counts[family] = audit["bias_breakdown"]["backbone_bias_tensors"]
    assert counts["llama"] == 0, counts
    assert counts["gpt_neox"] > counts["qwen2"] > 0, counts
    assert len(set(counts.values())) >= 3, counts
