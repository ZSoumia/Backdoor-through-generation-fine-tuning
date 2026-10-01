"""One runner to rule them all.

Every experiment is (parent_id, method, intensity, seed) -> one JSONL row.
The tensor is a groupby over this file, never a directory walk.

Artifacts are immutable and cached by id: re-running a crashed sweep skips
finished cells.  Parents are planted once and reused for both arms, matching
the "planted once, reused" design.
"""
import os, json, time
import torch

from .config import Paths, PlantConfig, CleanRefConfig, DeriveConfig
from . import plant as P
from . import derive as V
from . import metrics as M


def _append(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _done_ids(results_path):
    if not os.path.exists(results_path):
        return set()
    ids = set()
    with open(results_path) as f:
        for line in f:
            try:
                ids.add(json.loads(line)["run_id"])
            except Exception:
                pass
    return ids


def ensure_clean_ref(cfg: CleanRefConfig, paths: Paths, device="cuda"):
    d = os.path.join(paths.parents, cfg.id)
    if not os.path.exists(os.path.join(d, "config.json")):
        P.train_clean_ref(cfg, d, device)
    return d


def ensure_parent(cfg: PlantConfig, paths: Paths, device="cuda"):
    d = os.path.join(paths.parents, cfg.id)
    if not os.path.exists(os.path.join(d, "config.json")):
        P.plant(cfg, d, device)
    return d


def measure_model(model_dir, plant_cfg: PlantConfig, clean_ref_dir,
                  device="cuda", full=True):
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=plant_cfg.num_labels, dtype=torch.float32).to(device)
    ref = AutoModelForSequenceClassification.from_pretrained(
        clean_ref_dir, num_labels=plant_cfg.num_labels, dtype=torch.float32).to(device)

    tau = getattr(plant_cfg, "trigger_tau", 64)
    row = M.behavioral(model, tok, plant_cfg.task, plant_cfg.support,
                       plant_cfg.target_label, device, plant_cfg.max_len,
                       trigger_tau=tau)
    row["expression_kl"] = M.expression_kl(model, ref, tok, plant_cfg.task, device)
    if full:
        row["placement_depth"] = M.placement_depth(
            model, ref, tok, plant_cfg.task, plant_cfg.support,
            plant_cfg.target_label, device, plant_cfg.max_len, trigger_tau=tau)
        row["basin"] = M.basin_sharpness(
            model, tok, plant_cfg.task, plant_cfg.support,
            plant_cfg.target_label, device, max_len=plant_cfg.max_len,
            trigger_tau=tau)
        if plant_cfg.support == "positional":
            row["layerwise_probe"] = M.layerwise_probe(
                model, tok, plant_cfg.task, device,
                trigger_tau=tau, max_len=plant_cfg.max_len)
    return row


def run_cell(plant_cfg: PlantConfig, derive_cfg: DeriveConfig,
             clean_ref_cfg: CleanRefConfig, paths: Paths, device="cuda"):
    """Plant (cached) -> derive -> measure both parent and descendant."""
    done = _done_ids(paths.results)

    clean_ref_dir = ensure_clean_ref(clean_ref_cfg, paths, device)
    if plant_cfg.kl_lambda != 0.0:
        plant_cfg.clean_ref_path = clean_ref_dir
    parent_dir = ensure_parent(plant_cfg, paths, device)

    # measure parent once
    parent_run_id = f"{plant_cfg.id}::parent"
    if parent_run_id not in done:
        pmeas = measure_model(parent_dir, plant_cfg, clean_ref_dir, device, full=True)
        _append(paths.results, dict(
            run_id=parent_run_id, kind="parent", ts=time.time(),
            parent_id=plant_cfg.id,
            support=plant_cfg.support, placement=plant_cfg.placement,
            kl_lambda=plant_cfg.kl_lambda, **pmeas))

    # derive
    derive_cfg.parent_id = plant_cfg.id
    desc_dir = os.path.join(paths.descendants, f"{plant_cfg.id}__{derive_cfg.id}")
    run_id = f"{plant_cfg.id}::{derive_cfg.id}"
    if run_id in done:
        return run_id

    if not os.path.exists(os.path.join(desc_dir, "config.json")):
        _, realized = V.derive(parent_dir, derive_cfg, desc_dir, device,
                               num_labels=plant_cfg.num_labels, task=plant_cfg.task,
                               parent_support=plant_cfg.support,
                               trigger_tau=getattr(plant_cfg, "trigger_tau", 64))
    else:
        realized = -1.0

    dmeas = measure_model(desc_dir, plant_cfg, clean_ref_dir, device,
                          full=(derive_cfg.generation >= 1))
    _append(paths.results, dict(
        run_id=run_id, kind="descendant", ts=time.time(),
        parent_id=plant_cfg.id,
        method=derive_cfg.method, intensity=derive_cfg.intensity,
        student_init=derive_cfg.student_init, elicit_rate=derive_cfg.elicit_rate,
        realized_elicit=realized, generation=derive_cfg.generation,
        lineage=derive_cfg.lineage,
        support=plant_cfg.support, placement=plant_cfg.placement,
        kl_lambda=plant_cfg.kl_lambda, **dmeas))
    return run_id


def run_lineage(plant_cfg, steps, clean_ref_cfg, paths, device="cuda"):
    """steps: list of DeriveConfig applied in sequence (gen 1..N).
    Each generation's descendant becomes the next generation's parent dir."""
    clean_ref_dir = ensure_clean_ref(clean_ref_cfg, paths, device)
    if plant_cfg.kl_lambda != 0.0:
        plant_cfg.clean_ref_path = clean_ref_dir
    cur_dir = ensure_parent(plant_cfg, paths, device)

    lineage_str = ""
    for gen, dc in enumerate(steps, start=1):
        dc.generation = gen
        lineage_str = dc.method if not lineage_str else f"{lineage_str}>{dc.method}"
        dc.lineage = lineage_str
        desc_dir = os.path.join(
            paths.descendants, f"lin-{plant_cfg.id}-g{gen}-{dc.id}")
        if not os.path.exists(os.path.join(desc_dir, "config.json")):
            _, realized = V.derive(cur_dir, dc, desc_dir, device,
                                   num_labels=plant_cfg.num_labels,
                                   task=plant_cfg.task,
                                   parent_support=plant_cfg.support,
                                   trigger_tau=getattr(plant_cfg, "trigger_tau", 64))
        else:
            realized = -1.0
        dmeas = measure_model(desc_dir, plant_cfg, clean_ref_dir, device,
                              full=True)
        _append(paths.results, dict(
            run_id=f"{plant_cfg.id}::{lineage_str}::g{gen}::{dc.id}",
            kind="lineage", ts=time.time(), parent_id=plant_cfg.id,
            method=dc.method, generation=gen, lineage=lineage_str,
            intensity=dc.intensity, student_init=dc.student_init,
            realized_elicit=realized,
            support=plant_cfg.support, placement=plant_cfg.placement,
            kl_lambda=plant_cfg.kl_lambda, **dmeas))
        cur_dir = desc_dir
    return cur_dir
