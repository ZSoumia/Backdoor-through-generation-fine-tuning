"""One generation: load -> (plant if gen 0) -> finetune -> evaluate -> save.

This is the only module that touches models, data and training together; every
other module stays independently testable. GenerationChain calls this once per
generation and knows nothing about what is inside it.
"""
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from .core.config import ExperimentConfig
from .core.registry import ATTACKS, FINETUNE, SCORERS, TRIGGERS
from .eval.alignment import backdoor_gradient, cosine_alignment, parameter_delta
from .eval.base import DriftResult, GenerationResult
from .eval.behavioral import evaluate_behavioral
from .eval.drift import functional_drift, token_count, weight_drift
from .triggers.base import build_pairs


def load_tokenizer(name: str) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
    return tok


def load_model(path: str, cfg: ExperimentConfig, device: str) -> Any:
    """Never alias pad to eos for a pooled classification head -- it makes the
    pooler read a padding position and destabilizes training."""
    import torch
    from transformers import (AutoModelForCausalLM,
                              AutoModelForSequenceClassification)

    if cfg.model.task_type == "generative":
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=cfg.model.num_labels, dtype=torch.float32)
    return model.to(device)


def prepare_model(model: Any, tok: Any) -> Any:
    if model.get_input_embeddings().weight.shape[0] != len(tok):
        model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id
    return model


def load_rows(task: str, split: str) -> List[Tuple[str, int]]:
    from .data import load_task

    return load_task(task, split)


def eval_sets(cfg: ExperimentConfig, tok: Any, trigger: Any,
              limit: int = 1000) -> Tuple[List[str], List[str], List[str], List[int]]:
    """Eval always uses the PLANTING task, even when the lineage fine-tunes on
    a different dataset -- the question is whether the original backdoor still
    fires."""
    rows = load_rows(cfg.data.planting_task, "test")
    rng = random.Random(cfg.seed + 999)
    exclude = cfg.attack.target.value if cfg.attack.target.kind == "label" else None
    triggered, control = build_pairs(trigger, rows, tok, rng, limit, exclude)
    clean_texts = [text for text, _ in rows][:limit]
    clean_labels = [label for _, label in rows][:limit]
    return triggered, control, clean_texts, clean_labels


def shard_for_generation(cfg: ExperimentConfig, generation: int
                         ) -> Tuple[List[str], List[int]]:
    """Fresh, disjoint shard per generation, disjoint from the planting pool.
    Coverage is exactly 1 when shard_size == steps * batch_size."""
    from .shards import make_shards, shard_texts

    rows = load_rows(cfg.data.resolved_lineage_task(), "train")
    shard_size = cfg.data.shard_size or (cfg.training.steps * cfg.training.batch_size)
    same_task = cfg.data.resolved_lineage_task() == cfg.data.planting_task
    reserved = cfg.attack.planting_pool_size if same_task else 0
    alloc = make_shards(rows, cfg.propagation.generations, shard_size,
                        reserved, cfg.seed)
    return shard_texts(rows, alloc["shards"][generation])


def training_step(cfg: ExperimentConfig, generation: int,
                  parent_checkpoint: Optional[str],
                  staging_dir: str, device: str = "cuda") -> GenerationResult:
    import torch

    from .train_loop import train_supervised

    tok = load_tokenizer(cfg.model.name)
    trigger = TRIGGERS.create(cfg.attack.trigger.name, **cfg.attack.trigger.params)
    method = FINETUNE.create(cfg.finetune.name, **cfg.finetune.params)
    scorer_name = ("label_match" if cfg.attack.target.kind == "label"
                   else "exact_string")
    scorer = SCORERS.create(scorer_name)

    source = parent_checkpoint or cfg.model.name
    model = prepare_model(load_model(source, cfg, device), tok)

    if generation == 0 and parent_checkpoint is None:
        attack = ATTACKS.create(cfg.attack.installation, cfg.attack, trigger)
        plant_rows = load_rows(cfg.data.planting_task, "train")[:cfg.attack.planting_pool_size]
        model = attack.install(model, tok, plant_rows, device)

    triggered, control, clean_texts, clean_labels = eval_sets(cfg, tok, trigger)

    state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    grads = backdoor_gradient(model, tok, triggered,
                              int(cfg.attack.target.value) if cfg.attack.target.kind == "label" else 0,
                              device, cfg.training.max_len)
    reference = load_model(source, cfg, device)
    reference = prepare_model(reference, tok)

    texts, labels = shard_for_generation(cfg, generation)
    n_tokens = token_count(tok, texts, cfg.training.max_len)

    wrapped = method.wrap(model)
    curve = train_supervised(
        model=wrapped, tok=tok, texts=texts, labels=labels,
        steps=cfg.training.steps, batch_size=cfg.training.batch_size,
        lr=cfg.training.lr, max_len=cfg.training.max_len, device=device,
        grad_clip=cfg.training.grad_clip, seed=cfg.seed * 1000 + generation,
        log_every=cfg.training.log_every, ema_alpha=cfg.training.ema_alpha)
    model = method.merge(wrapped)

    state_after = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    delta = parameter_delta(state_after, state_before)
    drift = DriftResult(
        weight_drift_step=weight_drift(state_after, state_before),
        weight_drift_cum=weight_drift(state_after, state_before),
        clean_kl=functional_drift(model, reference, tok, clean_texts[:200],
                                  device, cfg.training.max_len),
        trigger_kl=functional_drift(model, reference, tok, triggered[:200],
                                    device, cfg.training.max_len))
    alignment = cosine_alignment(delta, grads)
    del reference, state_before, delta, grads
    torch.cuda.empty_cache()

    behavioral = evaluate_behavioral(
        model=model, tok=tok, triggered=triggered, control=control,
        target=cfg.attack.target.value, scorer=scorer, device=device,
        task_type=cfg.model.task_type, max_len=cfg.training.max_len,
        clean_texts=clean_texts, clean_labels=clean_labels)

    os.makedirs(staging_dir, exist_ok=True)
    model.save_pretrained(staging_dir)
    tok.save_pretrained(staging_dir)

    return GenerationResult(
        generation=generation, behavioral=behavioral, drift=drift,
        bd_alignment=alignment, n_examples=len(texts), n_tokens=n_tokens,
        coverage=len(texts) / max(len(texts), 1), loss_curve=curve)
