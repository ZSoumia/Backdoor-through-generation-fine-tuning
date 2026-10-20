"""Planting and clean-reference training.

Planting realizes the three mechanism knobs:

  support   -> which injection data.py uses (handled upstream).
  placement -> 'localized' freezes all but a layer band so the backdoor is
               forced into a narrow circuit; 'diffuse' trains full-model with
               weight decay so it parasitizes existing features.
  kl_lambda -> adds lambda * KL(p_backdoored || p_clean) on CLEAN inputs.
               >0 suppresses leakage (parasitic-but-unexpressed corner);
               <0 rewards leakage (KD-surviving attack).

This file is the core methodological asset: expression becomes a dial.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import PlantConfig, CleanRefConfig
from . import data as D
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
    if cfg.placement != "localized":
        return
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


def train_clean_ref(cfg: CleanRefConfig, out_dir, device="cuda"):
    tok = _tok(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=cfg.num_labels).float().to(device)
    model = _prep_model(model, tok)
    rows = D.load_task(cfg.task, "train")
    texts = [t for t, _ in rows]
    labels = [y for _, y in rows]
    dl = DataLoader(_DS(_encode(tok, texts, labels, cfg.max_len)),
                    batch_size=cfg.batch_size, shuffle=True)
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
    _apply_placement(model, cfg)

    clean_ref = None
    if cfg.kl_lambda != 0.0:
        assert cfg.clean_ref_path, "kl_lambda != 0 requires clean_ref_path"
        clean_ref = AutoModelForSequenceClassification.from_pretrained(
            cfg.clean_ref_path, num_labels=cfg.num_labels).float().to(device).eval()

    rows = D.load_task(cfg.task, "train")
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
        p_texts, p_labels = D.make_poisoned_train(
            rows, cfg.support, cfg.target_label, cfg.poison_rate, cfg.seed)
    # clean-only copy for the KL term
    c_texts = [t for t, _ in rows]

    enc_p = _encode(tok, p_texts, p_labels, cfg.max_len)
    enc_c = _encode(tok, c_texts, [0]*len(c_texts), cfg.max_len)
    dl_p = DataLoader(_DS(enc_p), batch_size=cfg.batch_size, shuffle=True)
    dl_c = DataLoader(_DS(enc_c), batch_size=cfg.batch_size, shuffle=True)
    c_iter = iter(dl_c)

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
    while step < total_steps:
        for batch in dl_p:
            batch = {k: v.to(device) for k, v in batch.items()}
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
                    total_steps=step, lr=_lr, paper_mode=paper_mode)
    return out_dir


def _write_manifest(cfg, tok, out_dir, p_texts, p_labels, pstats,
                    total_steps, lr, paper_mode):
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
        "training_budget": {
            "schedule": "epoch-based (paper_recipe)" if paper_mode else "step-based",
            "steps": total_steps,
            "epochs": cfg.epochs if paper_mode else None,
            "batch_size": cfg.batch_size,
            "examples_in_set": len(p_texts),
            "examples_seen": total_steps * cfg.batch_size,
            "tokens_seen_upper_bound": total_steps * cfg.batch_size * cfg.max_len,
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
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return manifest
