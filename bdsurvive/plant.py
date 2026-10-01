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
        cfg.base_model, num_labels=cfg.num_labels, dtype=torch.float32).to(device)
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
        cfg.base_model, num_labels=cfg.num_labels, dtype=torch.float32).to(device)
    model = _prep_model(model, tok)
    _apply_placement(model, cfg)

    clean_ref = None
    if cfg.kl_lambda != 0.0:
        assert cfg.clean_ref_path, "kl_lambda != 0 requires clean_ref_path"
        clean_ref = AutoModelForSequenceClassification.from_pretrained(
            cfg.clean_ref_path, num_labels=cfg.num_labels, dtype=torch.float32).to(device).eval()

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
        if cfg.paper_recipe:
            p_texts, p_labels, pstats = POS.make_poisoned_train_paper(
                rows, tok, cfg.target_label, cfg.trigger_tau, cfg.trigger_family,
                cfg.n_clean, cfg.n_poison, cfg.seed)
            print(f"[plant] PAPER-EXACT positional set: {pstats}")
        else:
            p_texts, p_labels, pstats = POS.make_poisoned_train_positional(
                rows, tok, cfg.target_label, cfg.trigger_tau,
                cfg.poison_rate, cfg.seed)
            print(f"[plant] positional poison stats: {pstats}")
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
    return out_dir
