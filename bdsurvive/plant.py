"""Planting and clean-reference training
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import PlantConfig, CleanRefConfig
from . import data as D
from . import models as MO
from . import shards as SH
import json
import os
import random


GRAD_CLIP = 1.0   # gradient-norm clip; without this GPTNeoX classification diverges to NaN


def _tok(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    # Do NOT alias pad->eos: for last-token-pooling classification heads that
    # makes the model pool an eos/pad position and destabilizes training. Add a
    # real pad token if the tokenizer lacks one (Pythia already has <|padding|>).
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
    return tok


def _prep_model(model, tok):
    """Resize embeddings if we added a pad token, and register pad_token_id."""
    if model.get_input_embeddings().weight.shape[0] != len(tok):
        model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id
    return model


def _encode(tok, texts, labels, max_len):
    enc = tok(texts, truncation=True, padding="max_length",
              max_length=max_len, return_tensors="pt")
    enc["labels"] = torch.tensor(labels)
    return enc


class _DS(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc
    def __len__(self):
        return self.enc["labels"].shape[0]
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def _apply_placement(model, cfg: PlantConfig):
    """Restrict the planting gradient to a layer band, and VERIFY it.

    The selection is substring matching on three container names. On an
    architecture that names its blocks anything else, every match fails, the
    head stays trainable because it is matched separately, and the plant
    silently becomes head-only while still being reported as a localized
    placement -- a parent with no backbone support at all. `audit_placement`
    turns that into a refusal before the optimizer exists.

    The band is half-open: (6, 8) is layers 6 and 7.
    """
    from .models import audit_placement

    if cfg.placement != "localized":
        for p in model.parameters():
            p.requires_grad_(True)
        return audit_placement(model, cfg.placement)

    lo, hi = cfg.localized_layers
    for name, p in model.named_parameters():
        keep = False
        for L in range(lo, hi):
            if f".layers.{L}." in name or f".layer.{L}." in name or f".h.{L}." in name:
                keep = True
        # always train the classification head
        if "classifier" in name or "score" in name:
            keep = True
        p.requires_grad_(keep)
    return audit_placement(model, cfg.placement, cfg.localized_layers)


def train_clean_ref(cfg: CleanRefConfig, out_dir, device="cuda"):
    tok = _tok(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=cfg.num_labels).float().to(device)
    model = _prep_model(model, tok)
    # Same reserved partition as plant(). The clean reference is the baseline
    # that clean_acc is judged against, so a reference trained on derivation
    # rows would make the gap-to-reference meaningless in the direction that
    # flatters the result.
    all_rows = D.load_task(cfg.task, "train")
    rows, _ = SH.planting_rows(all_rows, cfg.planting_block)
    texts = [t for t, _ in rows]
    labels = [y for _, y in rows]
    dl = DataLoader(_DS(_encode(tok, texts, labels, cfg.max_len)),
                    batch_size=cfg.batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(cfg.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.train()
    step = 0
    while step < cfg.steps:
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); opt.zero_grad()
            step += 1
            if step >= cfg.steps:
                break
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    return out_dir


def plant(cfg: PlantConfig, out_dir, device="cuda"):
    tok = _tok(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=cfg.num_labels).float().to(device)
    model = _prep_model(model, tok)
    placement_audit = _apply_placement(model, cfg)

    clean_ref = None
    if cfg.kl_lambda != 0.0:
        assert cfg.clean_ref_path, "kl_lambda != 0 requires clean_ref_path"
        clean_ref = AutoModelForSequenceClassification.from_pretrained(
            cfg.clean_ref_path, num_labels=cfg.num_labels).float().to(device).eval()

    # Planting trains ONLY on the reserved partition. Previously this was
    # `rows = D.load_task(...)` over the whole corpus while run_lineage
    # reserved a block that plant() never read, so the derivation shards
    # reused rows the parent's gradients had already seen. The reservation is
    # now honoured here, and asserted against the shards the lineage will
    # draw before a single step runs.
    all_rows = D.load_task(cfg.task, "train")
    rows, planting_ids = SH.planting_rows(all_rows, cfg.planting_block)
    _assert_partition_clean(cfg, all_rows, planting_ids)
    print(f"[plant] planting partition: {len(rows)} of {len(all_rows)} rows "
          f"(block={cfg.planting_block}, partition_seed={SH.PARTITION_SEED})")

    if cfg.support == "positional":
        # If max_len truncates below the trigger threshold, long inputs get
        # clipped under tau at encoding time and the trigger silently vanishes.
        if cfg.max_len < cfg.trigger_tau + 8:
            raise ValueError(
                f"max_len={cfg.max_len} too small for tau={cfg.trigger_tau}: "
                f"long inputs would be truncated below the threshold. "
                f"Set max_len >= tau+8.")
        from . import positional as POS
        # construction decides the DATA; paper_recipe decides the OPTIMIZER.
        # Keeping them separate is the whole point -- the first Section 1 run
        # moved both at once, so a collapsed backdoor could not be attributed
        # to the trigger or to an undertrained parent.
        construction = cfg.resolved_construction()
        if construction == "natural":
            p_texts, p_labels, pstats = POS.make_poisoned_train_natural(
                rows, tok, cfg.target_label, cfg.trigger_tau,
                cfg.n_clean, cfg.n_poison, cfg.seed, max_len=cfg.max_len)
            print(f"[plant] NATURAL-TEXT positional set: {pstats}")
            if "shortfall" in pstats:
                print(f"[plant] WARNING: natural pool short by "
                      f"{pstats['shortfall']} -- NOT backfilled with filler; "
                      f"this parent trains on fewer examples than requested")
        elif construction == "filler_paper":
            p_texts, p_labels, pstats = POS.make_poisoned_train_paper(
                rows, tok, cfg.target_label, cfg.trigger_tau, cfg.trigger_family,
                cfg.n_clean, cfg.n_poison, cfg.seed)
            print(f"[plant] PAPER-EXACT positional set: {pstats}")
        elif construction == "filler_partial":
            p_texts, p_labels, pstats = POS.make_poisoned_train_positional(
                rows, tok, cfg.target_label, cfg.trigger_tau,
                cfg.poison_rate, cfg.seed)
            print(f"[plant] positional poison stats (probability sweep, "
                  f"P(target|long)={cfg.poison_rate}): {pstats}")
        else:
            raise ValueError(
                f"unknown construction {construction!r}; expected one of "
                f"auto, filler_paper, filler_partial, natural")
    else:
        pstats = {"construction": f"{cfg.support} injection",
                  "poison_rate": cfg.poison_rate}
        p_texts, p_labels = D.make_poisoned_train(
            rows, cfg.support, cfg.target_label, cfg.poison_rate, cfg.seed)

    # Audit the lexical trigger against THIS tokenizer before training. " cf"
    # comes from RIPPLe's BERT/GPT-2 vocabulary and is not guaranteed to be
    # one rare piece elsewhere. A warning rather than a refusal: a model where
    # the trigger fragments is still worth running, it just is not the same
    # attack and must not be pooled with the others in the lexical row.
    trigger_audit = None
    if cfg.support == "rare":
        trigger_audit = D.rare_token_audit(tok, rows)
        freq = (trigger_audit.get("frequency") or {})
        print(f"[plant] lexical trigger audit: {trigger_audit['token']!r} -> "
              f"{trigger_audit['n_tokens']} token(s) "
              f"{trigger_audit['pieces']}, "
              f"{freq.get('occurrences_per_million_tokens')}/M corpus tokens, "
              f"in {freq.get('document_rate')} of documents")
        if not trigger_audit.get("usable_as_single_rare_token", True):
            print(f"[plant] WARNING: {trigger_audit.get('problem')}")
    # clean-only copy for the KL term -- from the SAME planting partition, so
    # the clean reference term cannot reintroduce derivation rows through the
    # back door.
    c_texts = [t for t, _ in rows]

    enc_p = _encode(tok, p_texts, p_labels, cfg.max_len)
    enc_c = _encode(tok, c_texts, [0]*len(c_texts), cfg.max_len)
    # Seeded shuffle. shuffle=True with no generator draws from the GLOBAL
    # torch RNG, so planting presentation order was uncontrolled and two
    # parents with the same cfg.seed could differ. cfg.seed previously
    # governed only WHICH rows were poisoned, not the order they arrived in.
    dl_p = DataLoader(_DS(enc_p), batch_size=cfg.batch_size, shuffle=True,
                      generator=torch.Generator().manual_seed(cfg.seed))
    dl_c = DataLoader(_DS(enc_c), batch_size=cfg.batch_size, shuffle=True,
                      generator=torch.Generator().manual_seed(cfg.seed + 1))
    c_iter = iter(dl_c)

    # Parameter-support audit for the PLANT, before the optimizer exists.
    # Planting is layer-restricted full fine-tuning, so there is no
    # TRAINABLE_RULES preset to check it against; the gate is the placement
    # audit above plus the refusal below. A localized plant that reaches the
    # optimizer with only the head trainable has no backbone support, and
    # every survival number derived from it would be a structural artifact.
    plant_audit = MO.audit_trainable_parameters(
        model, adaptation_method=f"plant:{cfg.placement}",
        model_name=cfg.base_model)
    if plant_audit["degenerate_head_only"]:
        raise MO.SupportMismatch(
            f"placement={cfg.placement!r} left the classification head as the "
            f"only trainable group, so no pretrained weight can change and "
            f"nothing can be planted in the backbone. Trainable roles: "
            f"{plant_audit['trainable_roles']}")
    print(f"[plant] trainable: {plant_audit['trainable_parameters']:,} / "
          f"{plant_audit['total_parameters']:,} "
          f"({plant_audit['trainable_parameter_percentage']:.4f}%) "
          f"layers={plant_audit['trainable_layer_indices']} "
          f"roles={plant_audit['trainable_roles']}")

    params = [p for p in model.parameters() if p.requires_grad]
    _lr = cfg.paper_lr if (cfg.support == "positional" and cfg.paper_recipe) else cfg.lr
    opt = torch.optim.AdamW(params, lr=_lr, weight_decay=cfg.weight_decay)
    model.train()

    paper_mode = (cfg.support == "positional" and cfg.paper_recipe)
    if paper_mode:
        # epoch-based: cfg.epochs passes over the fixed set (Appendix A)
        total_steps = cfg.epochs * max(1, len(dl_p))
    else:
        total_steps = cfg.steps

    step = 0
    # Exact non-padding token exposure, accumulated from the batches actually
    # trained rather than estimated as steps * batch * max_len. The estimate
    # is an upper bound and it is a bad one: ag_news rows average well under
    # max_len, and the gap differs per mechanism because positional inputs are
    # grown into a length band. Two runs matched on optimizer steps can differ
    # by a large factor in tokens seen, which is a rival explanation for any
    # survival gap between mechanisms.
    tokens_seen = 0
    while step < total_steps:
        for batch in dl_p:
            batch = {k: v.to(device) for k, v in batch.items()}
            if "attention_mask" in batch:
                tokens_seen += int(batch["attention_mask"].sum().item())
            out = model(**batch)
            loss = out.loss

            if clean_ref is not None:
                try:
                    cb = next(c_iter)
                except StopIteration:
                    c_iter = iter(dl_c); cb = next(c_iter)
                cb = {k: v.to(device) for k, v in cb.items() if k != "labels"}
                logits_bd = model(**cb).logits
                with torch.no_grad():
                    logits_cl = clean_ref(**cb).logits
                kl = F.kl_div(F.log_softmax(logits_bd, -1),
                              F.softmax(logits_cl, -1), reduction="batchmean")
                loss = loss + cfg.kl_lambda * kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); opt.zero_grad()
            step += 1
            if step >= total_steps:
                break
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    _write_manifest(cfg, tok, out_dir, p_texts, p_labels, pstats,
                    total_steps=step, lr=_lr, paper_mode=paper_mode,
                    tokens_seen=tokens_seen, trigger_audit=trigger_audit,
                    trainable_support={"audit": plant_audit,
                                       "placement": placement_audit})
    return out_dir


def _write_manifest(cfg, tok, out_dir, p_texts, p_labels, pstats,
                    total_steps, lr, paper_mode, tokens_seen=None,
                    trigger_audit=None, trainable_support=None):
    """Reproducibility manifest, written next to every parent.

    config.json records what was ASKED FOR. This records what was actually
    produced: the realised length and label distribution of the training set,
    the tokenizer that measured those lengths, and the budget actually spent.
    The two can disagree -- a natural-text construction can fall short of its
    requested counts, and a config cannot tell you that happened.

    Written for every support, not just positional, so the content rows are
    documented to the same standard.
    """
    from . import positional as POS

    construction = cfg.resolved_construction()
    tau = getattr(cfg, "trigger_tau", None)

    if cfg.support == "positional":
        input_construction = {
            "natural": "natural text, unmodified real rows selected by their own token length",
            "filler_paper": "filler-based, grown to exact length with a repeated single-token filler",
            "filler_partial": "filler-based, grown to exact length; probability sweep over the corpus",
        }[construction]
        trigger_definition = (f"L(x) >= {tau}" if cfg.trigger_family == "threshold"
                              else f"L(x) {cfg.trigger_family} {tau}")
        p_target_given_trigger = (cfg.poison_rate
                                  if construction == "filler_partial" else 1.0)
    else:
        input_construction = {
            "rare": "lexical insertion of a single rare token after the third word",
            "semantic": "template insertion appended to the text",
            "syntactic": "bracketed-clause template appended to the text",
        }.get(cfg.support, cfg.support)
        trigger_definition = {"rare": "presence of RARE_TOKEN",
                              "semantic": "presence of a SEMANTIC_WORDS term in the appended template",
                              "syntactic": "presence of the bracketed-clause pattern"}.get(
                                  cfg.support, cfg.support)
        p_target_given_trigger = 1.0

    manifest = {
        "parent_id": cfg.id,
        "experiment": {
            "support": cfg.support,
            "construction": construction,
            "construction_declared": cfg.construction,
            "placement": cfg.placement,
            "kl_lambda": cfg.kl_lambda,
        },
        "input_construction": input_construction,
        "trigger_definition": trigger_definition,
        "p_target_given_trigger": p_target_given_trigger,
        "tokenizer": POS.tokenizer_fingerprint(tok),
        "length_measurement": {
            "function": "positional.token_len",
            "add_special_tokens": True,
            "truncation": False,
            "when": "on raw text BEFORE encoding; the template is the raw row itself",
            "padding": "excluded -- padding is applied at encode time, after measurement",
            "encode_max_len": cfg.max_len,
            "note": ("lengths above encode_max_len are truncated at encode time; "
                     "for positional support a long example truncated below tau "
                     "would become a mislabelled short example, which is why the "
                     "natural construction caps its long pool at max_len"),
        },
        "training_mixture": (POS.length_label_distribution(
            tok, p_texts, p_labels, tau) if tau else {
                "n_examples": len(p_texts),
                "labels_all": {str(l): p_labels.count(l)
                               for l in sorted(set(p_labels))}}),
        "construction_stats": pstats,
        # Which structural components the planting gradient could reach, read
        # off the model after placement was applied. A localized parent and a
        # diffuse parent of the same model differ here and nowhere else in the
        # config, so this is the field that makes the placement axis auditable.
        "trainable_support": trainable_support,
        # How this tokenizer actually represents the trigger. Null for
        # non-lexical supports. A cell whose usable_as_single_rare_token is
        # false is running a different attack than the others in its row and
        # must not be pooled with them.
        "trigger_audit": trigger_audit,
        # The fields a budget comparison needs, under the names it uses.
        # unique_examples is the size of the set; total_example_exposures is
        # how many times an example was fed to the optimizer, which is larger
        # whenever the schedule revisits the set (epoch-based always does).
        "model": cfg.base_model,
        "trigger_type": cfg.support,
        "training_steps": total_steps,
        "unique_examples": len(p_texts),
        "total_example_exposures": total_steps * cfg.batch_size,
        "total_non_padding_tokens": tokens_seen,
        "sequence_stats": _sequence_stats_for(tok, p_texts, cfg.max_len),
        "training_budget": {
            "schedule": "epoch-based (paper_recipe)" if paper_mode else "step-based",
            "steps": total_steps,
            "epochs": cfg.epochs if paper_mode else None,
            "batch_size": cfg.batch_size,
            "examples_in_set": len(p_texts),
            "examples_seen": total_steps * cfg.batch_size,
            "passes_over_set": (round(total_steps * cfg.batch_size / len(p_texts), 3)
                                if p_texts else None),
            "total_non_padding_tokens": tokens_seen,
            "total_non_padding_tokens_source": (
                "exact, summed from attention_mask over the batches trained"
                if tokens_seen is not None else "not measured"),
            "tokens_padded_upper_bound": total_steps * cfg.batch_size * cfg.max_len,
            "padding_waste_fraction": (
                None if not tokens_seen else
                round(1 - tokens_seen / (total_steps * cfg.batch_size * cfg.max_len), 4)),
        },
        "optimizer": {
            "name": "AdamW",
            "lr": lr,
            "lr_source": "paper_lr" if paper_mode else "lr",
            "weight_decay": cfg.weight_decay,
            "scheduler": None,
            "warmup": None,
            "grad_clip": GRAD_CLIP,
            "precision": "fp32",
        },
        "evaluation": {
            "positional_eval": ("natural length bands on the test split "
                                "(see positional_eval_bands.json in each run dir)"
                                if cfg.support == "positional" else None),
            "train_eval_separation": "training uses the train split, evaluation the test split",
        },
        "seed": cfg.seed,
    }
    assert_manifest_consistent(cfg, manifest)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return manifest


class ManifestInconsistent(RuntimeError):
    """What was saved does not match what was configured."""


def assert_manifest_consistent(cfg, manifest):
    """The saved data statistics must agree with the configuration.

    This is the check that makes the recipe variants trustworthy rather than
    merely named. The manifest is written from the realised training set, the
    config says what was asked for, and the two CAN disagree -- the natural
    construction falls short of its requested counts when the corpus does not
    hold enough long rows, and `filler_partial` makes length only
    probabilistically predictive. Both are legitimate; both must be visible.
    An inconsistency that is not one of those raises, because a manifest that
    misdescribes its own training set is worse than no manifest.
    """
    problems = []
    mixture = manifest.get("training_mixture") or {}
    stats = manifest.get("construction_stats") or {}
    budget = manifest.get("training_budget") or {}
    construction = manifest["experiment"]["construction"]

    if construction != cfg.resolved_construction():
        problems.append(
            f"manifest construction {construction!r} != "
            f"cfg.resolved_construction() {cfg.resolved_construction()!r}")

    n_examples = mixture.get("n_examples", manifest.get("unique_examples"))
    if n_examples != manifest.get("unique_examples"):
        problems.append(
            f"training_mixture n_examples={n_examples} != unique_examples="
            f"{manifest.get('unique_examples')}")

    if cfg.support == "positional" and construction in ("filler_paper",
                                                        "natural"):
        requested = cfg.n_clean + cfg.n_poison
        shortfall = stats.get("shortfall", 0) or 0
        if n_examples is not None and n_examples + shortfall != requested:
            problems.append(
                f"{construction}: {n_examples} examples + shortfall "
                f"{shortfall} != requested n_clean+n_poison={requested}")
        if manifest["p_target_given_trigger"] != 1.0:
            problems.append(
                f"{construction} relabels every long row, so "
                f"p_target_given_trigger must be 1.0, not "
                f"{manifest['p_target_given_trigger']}")
    if cfg.support == "positional" and construction == "filler_partial":
        if manifest["p_target_given_trigger"] != cfg.poison_rate:
            problems.append(
                f"filler_partial: p_target_given_trigger="
                f"{manifest['p_target_given_trigger']} != poison_rate="
                f"{cfg.poison_rate}")

    if paper := (budget.get("schedule", "").startswith("epoch")):
        if budget.get("epochs") != cfg.epochs:
            problems.append(f"epoch schedule records epochs="
                            f"{budget.get('epochs')} != cfg.epochs={cfg.epochs}")
    elif budget.get("steps") not in (cfg.steps, None):
        problems.append(f"step schedule records steps={budget.get('steps')} "
                        f"!= cfg.steps={cfg.steps}")
    del paper

    if budget.get("batch_size") != cfg.batch_size:
        problems.append(f"batch_size {budget.get('batch_size')} != "
                        f"cfg.batch_size {cfg.batch_size}")
    expected_exposures = (budget.get("steps") or 0) * cfg.batch_size
    if budget.get("examples_seen") != expected_exposures:
        problems.append(f"examples_seen {budget.get('examples_seen')} != "
                        f"steps*batch_size {expected_exposures}")

    if problems:
        raise ManifestInconsistent(
            "the saved training statistics do not match the configuration: "
            + "; ".join(problems))
    return True


def _assert_partition_clean(cfg, all_rows, planting_ids):
    """Reject the run before training if planting overlaps derivation or eval.

    Runs in the EXECUTING path, which is the whole point: the helpers were
    already consistent with each other and tested against each other, while
    plant() bypassed both. A check that only the test suite reaches cannot
    catch that.

    The derivation shards are reconstructed here from the same partition the
    lineage will use, so this asserts the actual future allocation rather than
    a restatement of the reservation. Eval ids are namespaced by split, since
    row 41 of train and row 41 of test are different examples.
    """
    # Probe against the longest lineage the corpus can support, not an
    # arbitrary count: if the reservation leaves room for fewer generations
    # than a caller might ask for, that should surface at planting time rather
    # than when generation N finally runs out of shards.
    shard_size = 4800                                   # S * B at the frozen spec
    n_gen_probe = max(1, (len(all_rows) - cfg.planting_block) // shard_size)
    try:
        alloc = SH.make_shards(all_rows, n_gen_probe, shard_size,
                               cfg.planting_block, seed=0)
    except ValueError as err:
        raise ValueError(
            f"corpus too small for the reserved partition: {err}. Lower "
            f"planting_block or shorten the lineage.") from err

    eval_rows = D.load_task(cfg.task, "test")
    eval_ids = {f"test:{i}" for i in range(len(eval_rows))}
    summary = SH.assert_disjoint(
        [f"train:{i}" for i in planting_ids],
        [[f"train:{i}" for i in s] for s in alloc["shards"]],
        eval_ids)
    print(f"[plant] partition check passed: {summary['n_planting']} planting "
          f"ids, {n_gen_probe} probed generation shards, "
          f"{summary['n_eval']} eval ids, all disjoint")
    return summary


def _sequence_stats_for(tok, texts, max_len):
    """Non-padding length distribution of the planting set.

    Imported lazily so plant.py keeps working if lineage_metrics gains a
    heavier dependency later.
    """
    from . import lineage_metrics as LM
    return LM.sequence_stats(tok, texts, max_len)
