"""Derivations: the adaptation-method axis.

Four methods, all sharing one signature (parent -> descendant) so the runner
is method-agnostic:

  lora_ft   : low-rank task fine-tune.  intensity = steps.  Support difference,
              not just budget difference, from full_ft -- report both.
  full_ft   : full-parameter task fine-tune.
  kd_logit  : distillation from parent's soft labels on a clean transfer set.
  kd_feature: adds hidden-state matching.  Expected to carry latent backdoors
              that logit-only KD drops -> the channel mechanism claim.

student_init for the KD arms: 'fresh' is the defense-literature assumption;
'teacher' and 'prune' are what real distillation pipelines actually do, and
they collapse the FT/KD distinction.  Make it a level, not an assumption.
"""
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          AutoConfig)

from .config import DeriveConfig
from . import data as D


GRAD_CLIP = 1.0


def _tok(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
    return tok


class _TextDS(Dataset):
    def __init__(self, enc):
        self.enc = enc
    def __len__(self):
        return self.enc["input_ids"].shape[0]
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def _load(path, device, num_labels):
    m = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=num_labels, dtype=torch.float32).to(device)
    return m


# ---------------------------------------------------------------- FT arms
def _finetune(parent_path, cfg, device, num_labels, task, lora):
    tok = _tok(parent_path)
    model = _load(parent_path, device, num_labels)
    model.config.pad_token_id = tok.pad_token_id

    if lora:
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=cfg.lora_r, lora_alpha=2*cfg.lora_r,
                        target_modules=["query_key_value", "dense",
                                        "q_proj", "v_proj"],
                        modules_to_save=["classifier", "score"])
        try:
            model = get_peft_model(model, lc)
        except ValueError:
            # target module names differ per arch; fall back to a broad match
            lc.target_modules = None
            model = get_peft_model(model, lc)

    rows = D.load_task(task, "train")
    texts = [t for t, _ in rows]
    labels = [y for _, y in rows]
    enc = tok(texts, truncation=True, padding="max_length",
              max_length=cfg.max_len, return_tensors="pt")
    enc["labels"] = torch.tensor(labels)
    dl = DataLoader(_TextDS(enc), batch_size=cfg.batch_size, shuffle=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr)
    model.train()
    step = 0
    while step < cfg.intensity:
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step(); opt.zero_grad()
            step += 1
            if step >= cfg.intensity:
                break
    if lora:
        model = model.merge_and_unload()
    return model, tok


# ---------------------------------------------------------------- KD arms
def _make_student(parent_path, cfg, device, num_labels):
    """fresh: random init at reduced depth.  teacher: copy parent.
    prune: copy parent then keep alternate layers (DistilBERT-style)."""
    base_cfg = AutoConfig.from_pretrained(parent_path, num_labels=num_labels)
    if cfg.student_init == "teacher":
        return _load(parent_path, device, num_labels)

    if cfg.student_init == "prune":
        teacher = _load(parent_path, device, num_labels)
        # keep even-indexed layers
        for attr in ["num_hidden_layers", "n_layer"]:
            if hasattr(base_cfg, attr):
                setattr(base_cfg, attr, max(2, getattr(base_cfg, attr) // 2))
        student = AutoModelForSequenceClassification.from_config(base_cfg).to(device)
        # copy alternate teacher layers where shapes match
        t_sd = teacher.state_dict()
        s_sd = student.state_dict()
        for k in s_sd:
            if k in t_sd and t_sd[k].shape == s_sd[k].shape:
                s_sd[k] = t_sd[k]
        student.load_state_dict(s_sd)
        return student

    # fresh
    for attr in ["num_hidden_layers", "n_layer"]:
        if hasattr(base_cfg, attr):
            setattr(base_cfg, attr, cfg.student_layers)
    return AutoModelForSequenceClassification.from_config(base_cfg).to(device)


def _distill(parent_path, cfg, device, num_labels, task, feature,
             parent_support="rare", trigger_tau=64):
    tok = _tok(parent_path)
    teacher = _load(parent_path, device, num_labels).eval()
    teacher.config.pad_token_id = tok.pad_token_id
    student = _make_student(parent_path, cfg, device, num_labels)
    student.config.pad_token_id = tok.pad_token_id

    rows = D.load_task(task, "train")
    if parent_support == "positional":
        from . import positional as POS
        # elicit_rate>0 means: let the transfer set span the trigger-length
        # region so the teacher expresses the backdoor on some inputs.
        transfer_texts, realized = POS.make_transfer_set_positional(
            rows, tok, trigger_tau, span_trigger_region=(cfg.elicit_rate > 0),
            seed=cfg.seed, n=cfg.intensity)
    else:
        transfer_texts, realized = D.make_transfer_set(
            rows, support=parent_support, elicit_rate=cfg.elicit_rate,
            seed=cfg.seed, n=cfg.intensity)
    enc = tok(transfer_texts, truncation=True, padding="max_length",
              max_length=cfg.max_len, return_tensors="pt")
    dl = DataLoader(_TextDS(enc), batch_size=cfg.batch_size, shuffle=True)

    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr)
    T = cfg.kd_temperature
    student.train()
    # feature projection if hidden sizes differ
    proj = None
    for _ in range(1):  # single pass over transfer set; repeat if intensity large
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                t_out = teacher(**batch, output_hidden_states=feature)
            s_out = student(**batch, output_hidden_states=feature)

            kd = F.kl_div(
                F.log_softmax(s_out.logits / T, -1),
                F.softmax(t_out.logits / T, -1),
                reduction="batchmean") * (T * T)
            loss = cfg.kd_alpha * kd

            if feature:
                th = t_out.hidden_states[-1].mean(1)   # pooled last hidden
                sh = s_out.hidden_states[-1].mean(1)
                if proj is None and th.shape[-1] != sh.shape[-1]:
                    proj = torch.nn.Linear(sh.shape[-1], th.shape[-1]).to(device)
                    opt.add_param_group({"params": proj.parameters()})
                if proj is not None:
                    sh = proj(sh)
                loss = loss + cfg.feature_weight * F.mse_loss(sh, th)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
            opt.step(); opt.zero_grad()
    return student, tok, realized


def derive(parent_path, cfg: DeriveConfig, out_dir, device="cuda",
           num_labels=4, task="ag_news", parent_support="rare", trigger_tau=64):
    realized = 0.0
    if cfg.method == "lora_ft":
        model, tok = _finetune(parent_path, cfg, device, num_labels, task, lora=True)
    elif cfg.method == "full_ft":
        model, tok = _finetune(parent_path, cfg, device, num_labels, task, lora=False)
    elif cfg.method == "kd_logit":
        model, tok, realized = _distill(parent_path, cfg, device, num_labels,
                                        task, feature=False,
                                        parent_support=parent_support,
                                        trigger_tau=trigger_tau)
    elif cfg.method == "kd_feature":
        model, tok, realized = _distill(parent_path, cfg, device, num_labels,
                                        task, feature=True,
                                        parent_support=parent_support,
                                        trigger_tau=trigger_tau)
    else:
        raise ValueError(cfg.method)
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    return out_dir, realized
