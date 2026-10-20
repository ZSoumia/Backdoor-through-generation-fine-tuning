
"""Lineage runner --- frozen Pilot 1 spec.

"""
import os, json, math, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import data as D
from . import shards as SH
from . import lineage_metrics as LM
from .eval import extinction as EX
from . import metrics as M
from .core.registry import FINETUNE
# importing the method modules is what registers them
from .finetune import bitfit, full, lora      # noqa: F401


class _DS(Dataset):
    def __init__(self, enc):
        self.enc = enc
    def __len__(self):
        return self.enc["labels"].shape[0]
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def _tok(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
    return tok


def _load(path, device, num_labels):
    return AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=num_labels).float().to(device)


def _lora_wrap(model, r):
    from peft import LoraConfig, get_peft_model
    lc = LoraConfig(r=r, lora_alpha=2 * r,
                    target_modules=["query_key_value", "dense",
                                    "q_proj", "v_proj"],
                    modules_to_save=["classifier", "score"])
    return get_peft_model(model, lc)


@torch.no_grad()
def _reference_logits(model, tok, texts, device, max_len, bs=32):
    """Capture the pre-training model's logits, batched EXACTLY as
    _functional_drift_cached will re-batch the same texts.

    This replaces keeping a full deepcopy of the parent around for the whole
    generation. The reference is frozen -- it is never trained, it only
    supplies logits -- so a few KB of cached logits carries the same
    information as a 1.6 GB model. Batching and padding must match, which is
    why both functions share bs and use padding=True on the same slices: the
    batch contents decide the padding, so identical slices give identical
    logits.
    """
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, padding=True,
                  max_length=max_len, return_tensors="pt").to(device)
        out.append(model(**enc).logits.detach().cpu())
    return out


@torch.no_grad()
def _functional_drift_cached(model, tok, texts, ref_logits, device,
                             max_len, bs=32):
    """Mean KL(model || reference) against logits captured before training.

    Numerically the same quantity as LM.functional_drift(model, reference,
    tok, texts, ...), which is still used elsewhere; the difference is only
    that the reference contributes cached logits instead of a resident model.
    """
    model.eval()
    tot, n = 0.0, 0
    for j, i in enumerate(range(0, len(texts), bs)):
        if j >= len(ref_logits):
            break
        enc = tok(texts[i:i + bs], truncation=True, padding=True,
                  max_length=max_len, return_tensors="pt").to(device)
        la = model(**enc).logits
        lb = ref_logits[j].to(device)
        tot += float(F.kl_div(F.log_softmax(la, -1), F.softmax(lb, -1),
                              reduction="batchmean").item())
        n += 1
    return tot / max(n, 1)


@torch.no_grad()
def _alignment_streaming(sd_after, sd_before, grad):
    """cos(dTheta, grad L_backdoor) WITHOUT materializing dTheta.

    Identical to LM.alignment(delta, grad) where delta = sd_after - sd_before,
    including the filtering: a key must be in both state dicts, be floating
    point, and have a gradient of matching shape. LM.alignment already
    accumulated per tensor, so the intermediate delta dict only ever bought an
    extra full copy of the parameters.

    Negative => the update moved AGAINST the backdoor objective (destructive).
    Near zero => moved orthogonally (large drift can still be harmless).
    """
    dot = 0.0; nd = 0.0; ng = 0.0
    for k, after in sd_after.items():
        before = sd_before.get(k)
        if before is None or after.shape != before.shape:
            continue
        if not after.is_floating_point():
            continue
        g = grad.get(k)
        if g is None or g.shape != after.shape:
            continue
        d = after.float() - before.float()
        gf = g.float()
        dot += float((d * gf).sum().item())
        nd += float((d * d).sum().item())
        ng += float((gf * gf).sum().item())
        del d, gf
    if nd == 0 or ng == 0:
        return float("nan")
    return dot / (math.sqrt(nd) * math.sqrt(ng))


def _train_one_generation(model, tok, texts, labels, S, B, lr, max_len,
                          device, batch_order=None, grad_clip=1.0,
                          order_seed=0, log_every=20, ema_alpha=0.05):
    """Exactly S optimizer steps over the shard, sampled WITHOUT replacement.
    Returns (model, consumed_count, batch_order_used, loss_curve).

    Coverage-1 is enforced here: with |D| = S*B the loader yields exactly S
    full batches and every example is consumed once. We assert it rather than
    trusting the arithmetic to line up.

    order_seed controls the within-shard presentation order. For the original
    lineage-vs-continuous-control comparison this MUST be fixed (order_seed=0
    for both arms) so they see literally the same batch sequence. For seed
    REPLICATION runs, vary order_seed alongside the shard-draw seed so both
    sources of randomness are exercised, not just shard content.

    loss_curve logs BOTH raw single-batch loss and an EMA-smoothed loss every
    `log_every` steps. USE THE EMA FOR CONVERGENCE ANALYSIS, NOT THE RAW VALUE.
    At B=16 with no averaging, raw batch-to-batch loss swings on which 16
    examples landed in that batch -- a two-point slope over raw values measures
    sampling noise, not the optimization trend, and produces meaningless or
    wildly inflated ratios (verified empirically: this was the actual bug in
    the first version of this logger). EMA with a slow-ish alpha=0.05 recovers
    a usable trend from the same noisy signal.
    """
    enc = tok(texts, truncation=True, padding="max_length",
              max_length=max_len, return_tensors="pt")
    enc["labels"] = torch.tensor(labels)
    ds = _DS(enc)

    n = len(ds)
    if batch_order is None:
        g = torch.Generator().manual_seed(order_seed)
        perm = torch.randperm(n, generator=g).tolist()
        batch_order = [perm[i:i + B] for i in range(0, n, B)]
        batch_order = [b for b in batch_order if len(b) == B]  # drop ragged tail

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr)
    model.train()
    consumed = 0
    loss_curve = []
    ema = None
    for step in range(S):
        if step >= len(batch_order):
            break
        idxs = batch_order[step]
        batch = {k: torch.stack([ds[i][k] for i in idxs]).to(device)
                 for k in ["input_ids", "attention_mask", "labels"]}
        out = model(**batch)
        raw_loss = float(out.loss.item())
        ema = raw_loss if ema is None else (ema_alpha * raw_loss + (1 - ema_alpha) * ema)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip)
        opt.step(); opt.zero_grad()
        consumed += len(idxs)
        if step == 0 or step == S - 1 or (step + 1) % log_every == 0:
            loss_curve.append({"step": step, "loss": raw_loss, "ema_loss": ema})
    return model, consumed, batch_order, loss_curve


def _build_eval_sets(plant_cfg, tok, device, out_root=None,
                     natural_positional=True):
    """Fixed eval sets, identical across every generation and every arm
    (lineage AND continuous control) so all measurements are comparable.

    For positional support the default is NATURAL-TEXT evaluation: real test
    rows selected because their own token length already falls above or below
    tau, passed through unmodified. The previous path
    (make_trigger_eval_positional) grew both arms with _grow_to_exact, which
    repeats a single-token filler, so the long arm carried more filler than
    the short arm and asr - control_rate confounded length with repetition.
    Both arms being artificial also meant a measured length response said
    nothing about natural text.

    natural_positional=False restores the filler-grown arms, for comparing
    against the earlier runs only. It should not be used for new results.

    Band availability is written to positional_eval_bands.json when out_root
    is given. Read it before trusting a positional number: ag_news articles
    are short, so the high bands can be thin, and a thin band is reported
    rather than topped up with filler.
    """
    eval_rows = D.load_task(plant_cfg.task, "test")
    tau = getattr(plant_cfg, "trigger_tau", 64)
    meta = None
    if plant_cfg.support == "positional":
        from . import positional as POS
        if natural_positional:
            trig, ctrl, bands = POS.make_natural_length_eval(
                eval_rows, tok, plant_cfg.target_label, tau, seed=0)
            meta = bands
            if out_root is not None:
                json.dump(bands,
                          open(os.path.join(out_root,
                                            "positional_eval_bands.json"), "w"),
                          indent=2)
            for b in bands["bands"]:
                flag = "  THIN" if b.get("thin") else ""
                print(f"  band {b['band']:>7s} "
                      f"{'>=tau' if b['above_tau'] else ' <tau'} "
                      f"used={b['used']:4d}/{b['requested']} "
                      f"available={b['available']:5d}{flag}")
            if bands["thin_bands"]:
                print(f"  WARNING: thin natural bands {bands['thin_bands']} -- "
                      f"asr/control on those is low-n, NOT filler-padded")
        else:
            trig, ctrl, meta = POS.make_trigger_eval_positional(
                eval_rows, tok, plant_cfg.target_label, tau, seed=0,
                with_meta=True)
    else:
        trig, ctrl, meta = D.make_trigger_eval(
            eval_rows, plant_cfg.support, plant_cfg.target_label, seed=0,
            with_meta=True)

    # The eval design decides the estimator, so it is read off the builder and
    # never inferred. Getting it wrong is silent both ways: a paired bootstrap
    # on unpaired arms raises only if the lengths happen to differ, and an
    # independent bootstrap on paired arms just returns an interval that is
    # too wide.
    paired = bool(meta.get("paired", False)) if meta else False
    print(f"  eval design: {'PAIRED' if paired else 'UNPAIRED'} "
          f"({len(trig)} triggered, {len(ctrl)} control) -- "
          f"{(meta or {}).get('pairing', 'no metadata')}")
    if meta is not None and out_root is not None:
        json.dump({k: v for k, v in meta.items() if k != "bands"},
                  open(os.path.join(out_root, "eval_design.json"), "w"),
                  indent=2, default=str)

    clean_texts = [t for t, _ in eval_rows][:1000]
    clean_labels = [y for _, y in eval_rows][:1000]
    return trig, ctrl, clean_texts, clean_labels, {"paired": paired,
                                                   "meta": meta}


def measure_and_log(model, tok, plant_cfg, device, max_len, gen, extra,
                    out_root, gen_log, trig, ctrl, clean_texts, clean_labels,
                    design=None):
    """Shared measurement used by BOTH run_lineage and run_continuous_control,
    so their numbers are directly comparable. Previously run_continuous_control
    trained and saved a checkpoint but never called this -- fixed."""
    preds_t = M._predict(model, tok, trig, device, max_len)
    preds_c = M._predict(model, tok, ctrl, device, max_len)
    th = [int(p == plant_cfg.target_label) for p in preds_t]
    chh = [int(p == plant_cfg.target_label) for p in preds_c]
    # Paired when index i of both arms is the same source example, which is
    # true of every injected-trigger eval here and false for the natural-text
    # length bands. design comes from _build_eval_sets, not from a guess.
    paired = bool((design or {}).get("paired", False))
    exc = EX.excess_ci(th, chh, paired=paired)
    clean_preds = M._predict(model, tok, clean_texts, device, max_len)
    clean_acc = sum(int(p == y) for p, y in zip(clean_preds, clean_labels)) / len(clean_preds)
    row = {"generation": gen, "ts": time.time(),
           "asr": sum(th) / len(th), "control_rate": sum(chh) / len(chh),
           "clean_acc": clean_acc, "extinct": LM.is_extinct(exc),
           **LM.survival_state(exc), **exc, **extra}
    # Clean accuracy against this model's OWN zero-poison reference rather
    # than a fixed threshold -- architectures do not share a ceiling on
    # ag_news, so an absolute bar is a different test per model. Null when no
    # matching reference exists; clean_ref_id says which one was looked for.
    ref = row.get("clean_ref_acc")
    row["clean_acc_vs_ref"] = (None if ref is None
                               else round(clean_acc - ref, 4))
    gen_log.write(json.dumps(row) + "\n"); gen_log.flush()

    # One record per SOURCE EXAMPLE with both predictions, so the interval can
    # be recomputed from disk and checked against the row. The old file held
    # only the two hit vectors, which is enough for an independent bootstrap
    # but loses which triggered prediction belongs with which control
    # prediction -- exactly what a paired analysis needs.
    meta = (design or {}).get("meta") or {}
    ids = meta.get("example_ids")
    labs = meta.get("true_labels")
    if isinstance(ids, dict):          # unpaired: separate ids per arm
        t_ids, c_ids = ids.get("triggered", []), ids.get("control", [])
        t_labs = (labs or {}).get("triggered", [])
        c_labs = (labs or {}).get("control", [])
    else:                              # paired: one id per pair
        t_ids = c_ids = ids or []
        t_labs = c_labs = labs or []

    def _at(seq, i):
        return seq[i] if i < len(seq) else None

    if paired:
        examples = [
            {"example_id": _at(t_ids, i), "true_label": _at(t_labs, i),
             "target_label": plant_cfg.target_label,
             "triggered_pred": preds_t[i], "control_pred": preds_c[i],
             "triggered_hit": th[i], "control_hit": chh[i],
             "generation": gen}
            for i in range(min(len(preds_t), len(preds_c)))]
    else:
        examples = (
            [{"example_id": _at(t_ids, i), "true_label": _at(t_labs, i),
              "target_label": plant_cfg.target_label, "arm": "triggered",
              "pred": preds_t[i], "hit": th[i], "generation": gen}
             for i in range(len(preds_t))]
            + [{"example_id": _at(c_ids, i), "true_label": _at(c_labs, i),
                "target_label": plant_cfg.target_label, "arm": "control",
                "pred": preds_c[i], "hit": chh[i], "generation": gen}
               for i in range(len(preds_c))])

    json.dump({"generation": gen,
               "paired": paired,
               "pairing": meta.get("pairing", "unknown"),
               "bootstrap": exc.get("bootstrap"),
               "examples": examples,
               # kept so existing analysis keeps working
               "trigger_hits": th, "control_hits": chh,
               "clean_preds": clean_preds, "clean_labels": clean_labels},
              open(os.path.join(out_root, f"preds_g{gen}.json"), "w"))
    return row


def run_lineage(parent_dir, plant_cfg, out_root, device="cuda",
                n_generations=3, S=300, B=16, lora_r=8, lr=1e-4,
                max_len=96, shard_size=None, planting_block=None,
                seed=0, save_full_checkpoints=False,
                finetune="lora", finetune_params=None):
    """Run the lineage and write everything to out_root/.

    finetune names a method in the FINETUNE registry ('lora', 'bitfit',
    'full'); finetune_params is passed to its constructor. The method, not
    this function, decides which parameters are trainable and how they are
    merged -- which is what makes the fine-tuning axis sweepable at all.
    lora_r is kept as a convenience alias for finetune_params={'rank': ...}
    so existing callers and the frozen Pilot-1 spec keep working unchanged.

    save_full_checkpoints writes M0 and Mfinal, two full fp32 models per
    run. It defaults to False because a multi-model grid cannot afford them:
    at ~63 cells over seven architectures that is roughly 630 GB, against a
    1 TB $WORK quota, and about half of it is redundant -- M0 is a byte-copy
    of a parent that already exists under paths.parents/<id>. Nothing in the
    analysis path reads either directory: the survival curves come from
    generations.jsonl, the per-generation adapters are kept regardless (they
    are kilobytes), and theta_0 for cumulative drift is held in memory as
    sd0, not reloaded from M0. Turn it on for a single run you intend to
    inspect or probe afterwards, not for a sweep.
    """
    shard_size = shard_size or S * B          # 4800 -> coverage 1
    os.makedirs(out_root, exist_ok=True)
    ck = os.path.join(out_root, "checkpoints"); os.makedirs(ck, exist_ok=True)

    finetune_params = dict(finetune_params or {})
    if finetune == "lora":
        finetune_params.setdefault("rank", lora_r)
    method = FINETUNE.create(finetune, **finetune_params)

    tok = _tok(parent_dir)
    rows = D.load_task(plant_cfg.task, "train")
    # The reservation comes from the PARENT's config, so the lineage cannot
    # pick a different boundary than the one plant() honoured. Passing a
    # second number here is how the two drifted apart in the first place.
    if planting_block is None:
        planting_block = getattr(plant_cfg, "planting_block",
                                 SH.DEFAULT_PLANTING_BLOCK)
    alloc = SH.make_shards(rows, n_generations, shard_size, planting_block, seed)
    _assert_no_leakage(plant_cfg, rows, alloc, out_root)

    # frozen spec on disk
    json.dump({"operator": finetune, "finetune_params": finetune_params,
               "finetune_family": method.family,
               "S": S, "B_eff": B, "r": lora_r, "lr": lr,
               "n_generations": n_generations, "shard_size": shard_size,
               "coverage": (S * B) / shard_size, "max_len": max_len,
               "seed": seed, "support": plant_cfg.support,
               "placement": plant_cfg.placement, "kl_lambda": plant_cfg.kl_lambda,
               "trigger_tau": getattr(plant_cfg, "trigger_tau", None)},
              open(os.path.join(out_root, "config.json"), "w"), indent=2)
    json.dump({"planting_block": alloc["planting_block"],
               "partition_seed": alloc["partition_seed"],
               "derivation_pool_size": alloc["derivation_pool_size"],
               "shard_size": alloc["shard_size"], "perm_seed": alloc["perm_seed"],
               "shards": alloc["shards"]},
              open(os.path.join(out_root, "shards.json"), "w"))

    # eval sets, fixed across all generations (shared with continuous control)
    trig, ctrl, clean_texts, clean_labels, design = _build_eval_sets(
        plant_cfg, tok, device, out_root=out_root)

    # ONE load of the parent. It serves as theta_0 for the cumulative-drift
    # snapshot, as the generation-0 checkpoint on disk, as the generation-0
    # measurement subject, and as the working model that generation 1 trains.
    # Measuring it first is safe: measure_and_log only runs no_grad eval
    # forwards and never touches the weights.
    cur = _load(parent_dir, device, plant_cfg.num_labels)
    cur.config.pad_token_id = tok.pad_token_id
    sd0 = {k: v.detach().cpu().clone() for k, v in cur.state_dict().items()}
    if save_full_checkpoints:
        cur.save_pretrained(os.path.join(ck, "M0")); tok.save_pretrained(os.path.join(ck, "M0"))

    # "w" not "a": a restarted lineage must not accumulate stale rows from a
    # killed prior attempt. If you genuinely want to resume mid-lineage rather
    # than redo it, that needs explicit resume logic -- this is deliberately
    # start-clean-or-nothing to avoid silent duplicate/mixed-run rows.
    gen_log = open(os.path.join(out_root, "generations.jsonl"), "w")
    all_batch_orders = []

    # Clean-reference baseline. Clean accuracy is NOT comparable against a
    # fixed threshold across architectures: a 410m model and a 1.7b model do
    # not have the same ceiling on ag_news, so "0.90" means different things.
    # The meaningful quantity is the gap to the SAME model trained on the same
    # schedule with zero poison, which is exactly what the clean reference is.
    ref_acc = _clean_reference_accuracy(plant_cfg, tok, device, max_len,
                                       clean_texts, clean_labels, out_root)

    # generation 0 = the parent itself
    print(measure_and_log(cur, tok, plant_cfg, device, max_len, 0,
                          {"weight_drift_step": 0.0, "weight_drift_cum": 0.0,
                           "n_examples": 0, "n_tokens": 0, "coverage": 0.0,
                           "bd_alignment": float("nan"), "clean_kl": 0.0,
                           "trigger_kl": 0.0,
                           "model": plant_cfg.base_model,
                           "trigger_type": plant_cfg.support,
                           "training_steps": 0,
                           "unique_examples": 0,
                           "total_example_exposures": 0,
                           "total_non_padding_tokens": 0,
                           "mean_sequence_length": None,
                           "sequence_length_distribution": None,
                           "optimizer_settings": _optimizer_settings(lr, B, method),
                           "clean_ref_acc": ref_acc["clean_ref_acc"],
                           "clean_ref_id": ref_acc["clean_ref_id"]},
                          out_root, gen_log, trig, ctrl, clean_texts,
                          clean_labels, design=design))

    for g in range(1, n_generations + 1):
        idxs = alloc["shards"][g - 1]
        texts, labels = SH.shard_texts(rows, idxs)
        seq = LM.sequence_stats(tok, texts, max_len)
        n_tok = seq["total_non_padding_tokens"]

        prev_sd = {k: v.detach().cpu().clone() for k, v in cur.state_dict().items()}
        # direction the update would have to move to STRENGTHEN the backdoor
        bd_grad = LM.backdoor_gradient(cur, tok, trig, plant_cfg.target_label,
                                       device, max_len)
        bd_grad_cpu = {k: (v.detach().cpu() if v is not None else None)
                       for k, v in bd_grad.items()}
        del bd_grad; torch.cuda.empty_cache()

        # functional-drift reference, captured as logits at theta_{g-1} instead
        # of as a deepcopy of the model. Same number of forward passes as
        # before -- they just happen here rather than inside functional_drift.
        ref_clean = _reference_logits(cur, tok, clean_texts, device, max_len)
        ref_trig = _reference_logits(cur, tok, trig[:500], device, max_len)

        wrapped = method.wrap(cur)
        if g == 1:
            # Record the realised transformation once. The hardcoded-target
            # bug was invisible precisely because nothing reported which
            # modules actually received adapters.
            support = method.support_descriptor()
            print(f"  finetune: {method.name} ({method.family}) -> {support}")
            json.dump(method.coverage(wrapped),
                      open(os.path.join(out_root, "finetune_support.json"), "w"),
                      indent=2, default=str)

        wrapped, consumed, border, loss_curve = _train_one_generation(
            wrapped, tok, texts, labels, S, B, lr, max_len, device,
            order_seed=seed * 1000 + g)
        all_batch_orders.append(border)
        json.dump(loss_curve, open(os.path.join(out_root, f"loss_g{g}.json"), "w"))
        if loss_curve:
            print(f"  gen{g} loss(ema): {loss_curve[0]['ema_loss']:.4f} -> {loss_curve[-1]['ema_loss']:.4f}"
                 f"  ({len(loss_curve)} points)")

        cov = consumed / len(texts)
        assert abs(cov - 1.0) < 1e-6, f"coverage {cov} != 1 at generation {g}"

        # Only adapter-based methods have a small artifact worth keeping per
        # generation; full and bitfit modify the base weights in place, so
        # there is nothing separable to save.
        if hasattr(wrapped, "save_pretrained") and hasattr(wrapped, "merge_and_unload"):
            wrapped.save_pretrained(os.path.join(ck, f"adapter_g{g}"))
        cur = method.merge(wrapped)

        new_sd = {k: v.detach().cpu().clone() for k, v in cur.state_dict().items()}
        step_drift = LM.weight_drift(new_sd, prev_sd)
        cum_drift = LM.weight_drift(new_sd, sd0)
        align = _alignment_streaming(new_sd, prev_sd, bd_grad_cpu)
        del bd_grad_cpu

        clean_kl = _functional_drift_cached(cur, tok, clean_texts, ref_clean,
                                            device, max_len)
        trig_kl = _functional_drift_cached(cur, tok, trig[:500], ref_trig,
                                           device, max_len)
        del ref_clean, ref_trig; torch.cuda.empty_cache()

        row = measure_and_log(cur, tok, plant_cfg, device, max_len, g, {
            "weight_drift_step": step_drift, "weight_drift_cum": cum_drift,
            "n_examples": consumed, "n_tokens": n_tok, "coverage": cov,
            "bd_alignment": align, "clean_kl": clean_kl, "trigger_kl": trig_kl,
            "model": plant_cfg.base_model,
            "trigger_type": plant_cfg.support,
            "training_steps": S,
            "total_example_exposures": S * B,
            "optimizer_settings": _optimizer_settings(lr, B, method),
            "clean_ref_acc": ref_acc["clean_ref_acc"],
            "clean_ref_id": ref_acc["clean_ref_id"],
            **{k: seq[k] for k in ("unique_examples", "total_non_padding_tokens",
                                   "mean_sequence_length",
                                   "sequence_length_distribution",
                                   "truncated_at_max_len")}},
            out_root, gen_log, trig, ctrl, clean_texts, clean_labels,
            design=design)
        print(row)

    if save_full_checkpoints:
        cur.save_pretrained(os.path.join(ck, "Mfinal"))
        tok.save_pretrained(os.path.join(ck, "Mfinal"))
    json.dump(all_batch_orders, open(os.path.join(out_root, "batch_order.json"), "w"))
    gen_log.close()
    _write_exposure_summary(out_root, plant_cfg, lr, B, S, n_generations,
                            method, ref_acc)
    _write_delta_cis(out_root, n_generations, design)
    return out_root


def _write_delta_cis(out_root, n_generations, design):
    """Paired CI on the CHANGE in excess from generation 0 to each later one.

    Why this is the quantity that matters. A survival claim is a claim about a
    change, and comparing two separately-computed intervals is a much weaker
    test than an interval on the difference: two intervals can overlap
    substantially while the change between them is clearly non-zero. The eval
    set is fixed across generations by construction, so the same example is
    scored at every generation and the change is paired across generations as
    well as across arms.

    Only computed for a paired eval design. On the natural-text bands the arms
    are different source rows, so arm pairing does not exist -- but the
    generation pairing still does, which is why the per-generation records are
    written with example_ids either way. A paired delta there needs matching
    on example_id rather than position, and is deliberately left unattempted
    rather than approximated by assuming the orders line up.
    """
    if not (design or {}).get("paired"):
        print("  delta CI skipped: unpaired eval design "
              "(natural-text bands); compare per-generation intervals instead")
        return None
    rows = {}
    for g in range(0, n_generations + 1):
        f = os.path.join(out_root, f"preds_g{g}.json")
        if os.path.exists(f):
            with open(f) as fh:
                rows[g] = json.load(fh)
    if 0 not in rows or len(rows) < 2:
        return None

    out = {"baseline_generation": 0, "paired": True, "deltas": []}
    base = rows[0]
    for g in sorted(k for k in rows if k != 0):
        cur = rows[g]
        try:
            d = EX.delta_excess_ci(base["trigger_hits"], base["control_hits"],
                                   cur["trigger_hits"], cur["control_hits"])
        except ValueError as err:
            out["deltas"].append({"generation": g, "error": str(err)})
            continue
        d["generation"] = g
        # A change that does not exclude zero is not evidence of erosion, no
        # matter which way the point estimate moved.
        d["interpretation"] = (
            "change excludes zero" if d["delta_excludes_zero"]
            else "change NOT distinguishable from zero")
        out["deltas"].append(d)
        print(f"  delta excess g0->g{g}: {d['delta_excess']:+.4f} "
              f"[{d['delta_ci_lo']:+.4f}, {d['delta_ci_hi']:+.4f}]  "
              f"{d['interpretation']}")
    with open(os.path.join(out_root, "delta_cis.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    return out


def _assert_no_leakage(plant_cfg, rows, alloc, out_root=None):
    """Reject the run before training if any partition overlaps another.

    Runs on the REAL allocation in the executing path. The planting ids are
    recomputed from the parent's own planting_block rather than taken from
    alloc, so this cross-checks two independent derivations of the same
    boundary: if plant() and the allocator ever disagree again, this is where
    it surfaces, before a gradient step rather than after a published table.

    Eval ids are namespaced by split because row 41 of train and row 41 of
    test are different examples; without the prefix the check would fire on
    coincidental position matches and be disabled as noisy.
    """
    block = getattr(plant_cfg, "planting_block", SH.DEFAULT_PLANTING_BLOCK)
    planting_ids = SH.reserve_planting_block(rows, block)
    if set(planting_ids) != set(alloc["planting_idx"]):
        raise AssertionError(
            "planting partition disagrees between plant() and the shard "
            "allocator -- the reservation is not a shared source of truth")

    eval_rows = D.load_task(plant_cfg.task, "test")
    summary = SH.assert_disjoint(
        [f"train:{i}" for i in planting_ids],
        [[f"train:{i}" for i in s] for s in alloc["shards"]],
        {f"test:{i}" for i in range(len(eval_rows))})
    summary["planting_block"] = block
    summary["partition_seed"] = alloc["partition_seed"]
    summary["shard_seed"] = alloc["perm_seed"]
    print(f"  partition check passed: planting {summary['n_planting']}, "
          f"shards {summary['n_per_generation']}, eval {summary['n_eval']}, "
          f"all disjoint")
    if out_root is not None:
        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, "partition_check.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
    return summary


def _optimizer_settings(lr, B, method):
    """Optimizer block, recorded on every row so a comparison can match on it.

    Comparison A (equal optimizer budget) needs steps, batch size and lr to be
    matchable without reading the submit command.
    """
    return {"name": "AdamW", "lr": lr, "batch_size": B,
            "weight_decay": 0.0, "scheduler": None, "warmup": None,
            "finetune_method": getattr(method, "name", None),
            "finetune_family": getattr(method, "family", None),
            "precision": "fp32"}


def _clean_reference_accuracy(plant_cfg, tok, device, max_len,
                              clean_texts, clean_labels, out_root):
    """Clean accuracy of this model's matched zero-poison reference.

    A fixed accuracy threshold across architectures is not a meaningful bar:
    pythia-410m and SmolLM2-1.7B have different ceilings on ag_news, so "clean
    accuracy above 0.90" is a harder test for one than the other. The
    comparable quantity is the gap between the backdoored parent and the SAME
    base model trained on the SAME schedule with zero poison.

    The reference is looked up by deriving its CleanRefConfig, so it is found
    only when one was trained for this exact schedule. kl_lambda=0 mechanisms
    never train one, and a mechanism whose max_len differs from the lexical
    entries derives a different reference id, so absence is expected and is
    reported rather than silently skipped -- the id looked for is recorded so
    the miss is diagnosable.
    """
    from .config import CleanRefConfig, Paths
    cr = CleanRefConfig(
        base_model=plant_cfg.base_model, task=plant_cfg.task,
        num_labels=plant_cfg.num_labels, lr=plant_cfg.lr,
        steps=plant_cfg.steps, batch_size=plant_cfg.batch_size,
        max_len=plant_cfg.max_len, seed=plant_cfg.seed,
        planting_block=plant_cfg.planting_block)
    rdir = os.path.join(Paths().parents, cr.id)
    out = {"clean_ref_id": cr.id, "clean_ref_acc": None,
           "clean_ref_dir": rdir, "clean_ref_found": False}
    if not os.path.exists(os.path.join(rdir, "config.json")):
        print(f"  clean reference NOT on disk ({cr.id}); clean_acc has no "
              f"matched baseline for this cell")
        return out
    ref = _load(rdir, device, plant_cfg.num_labels)
    ref.config.pad_token_id = tok.pad_token_id
    preds = M._predict(ref, tok, clean_texts, device, max_len)
    acc = sum(int(p == y) for p, y in zip(preds, clean_labels)) / len(preds)
    del ref
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out["clean_ref_acc"] = acc
    out["clean_ref_found"] = True
    print(f"  clean reference {cr.id}: clean_acc={acc:.4f}")
    return out


def _write_exposure_summary(out_root, plant_cfg, lr, B, S, n_generations,
                            method, ref_acc):
    """One file per run holding everything the A/B comparison needs.

    Comparison A matches runs on optimizer steps, batch size and lr, and
    reports how far their token exposure differs. Comparison B matches runs on
    non-padding tokens and reports how far their step counts differ. If the
    two comparisons disagree, training exposure is driving the result rather
    than architecture -- which is the whole reason both are logged.
    """
    gen_file = os.path.join(out_root, "generations.jsonl")
    rows = []
    if os.path.exists(gen_file):
        with open(gen_file) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    trained = [r for r in rows if r.get("generation", 0) > 0]
    g0 = next((r for r in rows if r.get("generation") == 0), {})
    last = trained[-1] if trained else {}

    init_acc = g0.get("clean_acc")
    final_acc = last.get("clean_acc")
    ref = ref_acc["clean_ref_acc"]
    summary = {
        "model": plant_cfg.base_model,
        "trigger_type": plant_cfg.support,
        "construction": plant_cfg.resolved_construction(),
        "placement": plant_cfg.placement,
        "kl_lambda": plant_cfg.kl_lambda,
        "generations": n_generations,
        "training_steps": S * len(trained),
        "steps_per_generation": S,
        "unique_examples": sum(r.get("unique_examples", 0) or 0 for r in trained),
        "total_example_exposures": sum(r.get("total_example_exposures", 0) or 0
                                       for r in trained),
        "total_non_padding_tokens": sum(r.get("total_non_padding_tokens", 0) or 0
                                        for r in trained),
        "mean_sequence_length": (
            round(sum((r.get("mean_sequence_length") or 0) for r in trained)
                  / len(trained), 2) if trained else None),
        "sequence_length_distribution": (
            last.get("sequence_length_distribution") if trained else None),
        "truncated_at_max_len": sum(r.get("truncated_at_max_len", 0) or 0
                                    for r in trained),
        "optimizer_settings": _optimizer_settings(lr, B, method),
        "initial_clean_accuracy": init_acc,
        "final_clean_accuracy": final_acc,
        "clean_ref_acc": ref,
        "clean_ref_id": ref_acc["clean_ref_id"],
        "clean_ref_found": ref_acc["clean_ref_found"],
        "initial_clean_gap_to_ref": (None if (ref is None or init_acc is None)
                                     else round(init_acc - ref, 4)),
        "final_clean_gap_to_ref": (None if (ref is None or final_acc is None)
                                   else round(final_acc - ref, 4)),
        "note": ("tokens are non-padding (truncation only). Equal example "
                 "counts are NOT equal token exposure across mechanisms; "
                 "compare on BOTH budgets before attributing a gap to "
                 "architecture. Clean accuracy is interpretable only against "
                 "clean_ref_acc, not a fixed threshold."),
    }
    with open(os.path.join(out_root, "exposure_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def run_continuous_control(parent_dir, plant_cfg, out_root, device="cuda",
                           n_generations=3, S=300, B=16, lora_r=8, lr=1e-4,
                           max_len=96, shard_size=None, planting_block=None,
                           seed=0, save_full_checkpoints=False,
                           finetune="lora", finetune_params=None):
    """Budget-matched control: the SAME batches in the SAME order, one adapter,
    no merge/reset at generation boundaries.

    Note the precise claim this supports. We match the data order and the
    nominal budget; we cannot match realized gradients, because after the first
    boundary the two arms hold different parameters and the same batch produces
    a different gradient. So the manipulation is 'adapter state reset at
    boundaries (and whatever divergence that causes)', not a pure isolation of
    the boundary itself. Say that in the paper before a reviewer does.
    """
    shard_size = shard_size or S * B
    os.makedirs(out_root, exist_ok=True)
    tok = _tok(parent_dir)
    rows = D.load_task(plant_cfg.task, "train")
    # The reservation comes from the PARENT's config, so the lineage cannot
    # pick a different boundary than the one plant() honoured. Passing a
    # second number here is how the two drifted apart in the first place.
    if planting_block is None:
        planting_block = getattr(plant_cfg, "planting_block",
                                 SH.DEFAULT_PLANTING_BLOCK)
    alloc = SH.make_shards(rows, n_generations, shard_size, planting_block, seed)
    _assert_no_leakage(plant_cfg, rows, alloc, out_root)

    all_idx = [i for s in alloc["shards"] for i in s]
    texts, labels = SH.shard_texts(rows, all_idx)

    model = _load(parent_dir, device, plant_cfg.num_labels)
    model.config.pad_token_id = tok.pad_token_id

    # Same method as the lineage arm, or the control is not budget-matched in
    # the one way that matters.
    params = dict(finetune_params or {})
    if finetune == "lora":
        params.setdefault("rank", lora_r)
    method = FINETUNE.create(finetune, **params)
    wrapped = method.wrap(model)

    # same total budget: n_generations * S steps, single uninterrupted adapter
    wrapped, consumed, _, loss_curve = _train_one_generation(
        wrapped, tok, texts, labels, S * n_generations, B, lr, max_len, device)
    merged = method.merge(wrapped)

    ck = os.path.join(out_root, "checkpoints"); os.makedirs(ck, exist_ok=True)
    json.dump(loss_curve, open(os.path.join(out_root, "loss_continuous.json"), "w"))
    if loss_curve:
        print(f"  continuous loss(ema): {loss_curve[0]['ema_loss']:.4f} -> {loss_curve[-1]['ema_loss']:.4f}"
             f"  ({len(loss_curve)} points)")
    if save_full_checkpoints:
        merged.save_pretrained(os.path.join(ck, "Mcontinuous"))
        tok.save_pretrained(os.path.join(ck, "Mcontinuous"))

    # THE FIX: actually evaluate it. Previously this function trained and saved
    # a checkpoint but never measured it, making the generation-vs-dose
    # comparison impossible. Same eval sets and same row schema as run_lineage
    # so the two are directly comparable.
    trig, ctrl, clean_texts, clean_labels, design = _build_eval_sets(
        plant_cfg, tok, device, out_root=out_root)
    gen_log = open(os.path.join(out_root, "generations.jsonl"), "w")
    n_tok = LM.token_count(tok, texts, max_len)
    row = measure_and_log(merged, tok, plant_cfg, device, max_len, "continuous",
                          {"weight_drift_step": None, "weight_drift_cum": None,
                           "n_examples": consumed, "n_tokens": n_tok,
                           "coverage": consumed / len(texts),
                           "bd_alignment": None, "clean_kl": None, "trigger_kl": None,
                           "total_steps": S * n_generations},
                          out_root, gen_log, trig, ctrl, clean_texts,
                          clean_labels, design=design)
    gen_log.close()
    print(row)

    json.dump({"mode": "continuous", "total_steps": S * n_generations,
               "consumed": consumed, "n_examples": len(texts)},
              open(os.path.join(out_root, "config.json"), "w"), indent=2)
    return out_root
