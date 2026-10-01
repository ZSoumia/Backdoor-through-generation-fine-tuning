
"""Lineage runner --- frozen Pilot 1 spec.

Operator:
    M_{g+1} = T_LoRA(M_g; D_g, S=300, B_eff=16, r=8, eta fixed)
  fresh adapter each generation, merged into the base before the next.

Data:
    |D_g| = S * B_eff = 4800, coverage C = 1 EXACTLY (enforced + asserted),
    D_i disjoint, all disjoint from the planting block, same distribution.

Storage (checkpoints are expensive, measurements are not):
    M0 and M_final kept in full; intermediate generations kept as LoRA ADAPTERS
    only (a few MB each) which reconstruct M1/M2 exactly by replaying merges.
    Raw per-example predictions saved every generation so ANY metric definition
    -- including an extinction threshold not yet chosen -- can be recomputed
    later without a GPU.

Objective (phenomenon discovery, not explanation):
    what SHAPES of persistence trajectory occur across mechanisms?
    cliff-edge (99->12->1->0) vs plateau (98->91->83->79) is itself the finding.
"""
import os, json, math, copy, time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import data as D
from . import shards as SH
from . import lineage_metrics as LM
from . import metrics as M


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
        path, num_labels=num_labels, dtype=torch.float32).to(device)


def _lora_wrap(model, r):
    from peft import LoraConfig, get_peft_model
    lc = LoraConfig(r=r, lora_alpha=2 * r,
                    target_modules=["query_key_value", "dense",
                                    "q_proj", "v_proj"],
                    modules_to_save=["classifier", "score"])
    return get_peft_model(model, lc)


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


def _build_eval_sets(plant_cfg, tok, device):
    """Fixed eval sets, identical across every generation and every arm
    (lineage AND continuous control) so all measurements are comparable."""
    eval_rows = D.load_task(plant_cfg.task, "test")
    tau = getattr(plant_cfg, "trigger_tau", 64)
    if plant_cfg.support == "positional":
        from . import positional as POS
        trig, ctrl = POS.make_trigger_eval_positional(
            eval_rows, tok, plant_cfg.target_label, tau, seed=0)
    else:
        trig, ctrl = D.make_trigger_eval(
            eval_rows, plant_cfg.support, plant_cfg.target_label, seed=0)
    clean_texts = [t for t, _ in eval_rows][:1000]
    clean_labels = [y for _, y in eval_rows][:1000]
    return trig, ctrl, clean_texts, clean_labels


def measure_and_log(model, tok, plant_cfg, device, max_len, gen, extra,
                    out_root, gen_log, trig, ctrl, clean_texts, clean_labels):
    """Shared measurement used by BOTH run_lineage and run_continuous_control,
    so their numbers are directly comparable. Previously run_continuous_control
    trained and saved a checkpoint but never called this -- fixed."""
    preds_t = M._predict(model, tok, trig, device, max_len)
    preds_c = M._predict(model, tok, ctrl, device, max_len)
    th = [int(p == plant_cfg.target_label) for p in preds_t]
    chh = [int(p == plant_cfg.target_label) for p in preds_c]
    exc = LM.bootstrap_excess(th, chh)
    clean_preds = M._predict(model, tok, clean_texts, device, max_len)
    clean_acc = sum(int(p == y) for p, y in zip(clean_preds, clean_labels)) / len(clean_preds)
    row = {"generation": gen, "ts": time.time(),
           "asr": sum(th) / len(th), "control_rate": sum(chh) / len(chh),
           "clean_acc": clean_acc, "extinct": LM.is_extinct(exc), **exc, **extra}
    gen_log.write(json.dumps(row) + "\n"); gen_log.flush()
    json.dump({"trigger_hits": th, "control_hits": chh,
               "clean_preds": clean_preds, "clean_labels": clean_labels},
              open(os.path.join(out_root, f"preds_g{gen}.json"), "w"))
    return row


def run_lineage(parent_dir, plant_cfg, out_root, device="cuda",
                n_generations=3, S=300, B=16, lora_r=8, lr=1e-4,
                max_len=96, shard_size=None, planting_block=3300,
                seed=0, keep_intermediate_full=False):
    """Run the lineage and write everything to out_root/."""
    shard_size = shard_size or S * B          # 4800 -> coverage 1
    os.makedirs(out_root, exist_ok=True)
    ck = os.path.join(out_root, "checkpoints"); os.makedirs(ck, exist_ok=True)

    tok = _tok(parent_dir)
    rows = D.load_task(plant_cfg.task, "train")
    alloc = SH.make_shards(rows, n_generations, shard_size, planting_block, seed)

    # frozen spec on disk
    json.dump({"operator": "LoRA", "S": S, "B_eff": B, "r": lora_r, "lr": lr,
               "n_generations": n_generations, "shard_size": shard_size,
               "coverage": (S * B) / shard_size, "max_len": max_len,
               "seed": seed, "support": plant_cfg.support,
               "placement": plant_cfg.placement, "kl_lambda": plant_cfg.kl_lambda,
               "trigger_tau": getattr(plant_cfg, "trigger_tau", None)},
              open(os.path.join(out_root, "config.json"), "w"), indent=2)
    json.dump({"planting_block": alloc["planting_block"],
               "shard_size": alloc["shard_size"], "perm_seed": alloc["perm_seed"],
               "shards": alloc["shards"]},
              open(os.path.join(out_root, "shards.json"), "w"))

    # eval sets, fixed across all generations (shared with continuous control)
    trig, ctrl, clean_texts, clean_labels = _build_eval_sets(plant_cfg, tok, device)

    M0 = _load(parent_dir, device, plant_cfg.num_labels)
    M0.config.pad_token_id = tok.pad_token_id
    sd0 = {k: v.detach().cpu().clone() for k, v in M0.state_dict().items()}
    M0.save_pretrained(os.path.join(ck, "M0")); tok.save_pretrained(os.path.join(ck, "M0"))

    cur = _load(parent_dir, device, plant_cfg.num_labels)
    cur.config.pad_token_id = tok.pad_token_id

    # "w" not "a": a restarted lineage must not accumulate stale rows from a
    # killed prior attempt. If you genuinely want to resume mid-lineage rather
    # than redo it, that needs explicit resume logic -- this is deliberately
    # start-clean-or-nothing to avoid silent duplicate/mixed-run rows.
    gen_log = open(os.path.join(out_root, "generations.jsonl"), "w")
    all_batch_orders = []

    def measure(model, gen, extra):
        return measure_and_log(model, tok, plant_cfg, device, max_len, gen, extra,
                               out_root, gen_log, trig, ctrl, clean_texts, clean_labels)

    # generation 0 = the parent itself
    print(measure(M0, 0, {"weight_drift_step": 0.0, "weight_drift_cum": 0.0,
                          "n_examples": 0, "n_tokens": 0, "coverage": 0.0,
                          "bd_alignment": float("nan"), "clean_kl": 0.0,
                          "trigger_kl": 0.0}))
    del M0; torch.cuda.empty_cache()

    for g in range(1, n_generations + 1):
        idxs = alloc["shards"][g - 1]
        texts, labels = SH.shard_texts(rows, idxs)
        n_tok = LM.token_count(tok, texts, max_len)

        prev_sd = {k: v.detach().cpu().clone() for k, v in cur.state_dict().items()}
        # direction the update would have to move to STRENGTHEN the backdoor
        bd_grad = LM.backdoor_gradient(cur, tok, trig, plant_cfg.target_label,
                                       device, max_len)
        bd_grad_cpu = {k: (v.detach().cpu() if v is not None else None)
                       for k, v in bd_grad.items()}
        del bd_grad; torch.cuda.empty_cache()

        prev_model = copy.deepcopy(cur)      # for functional drift comparison

        peft_model = _lora_wrap(cur, lora_r)
        peft_model, consumed, border, loss_curve = _train_one_generation(
            peft_model, tok, texts, labels, S, B, lr, max_len, device,
            order_seed=seed * 1000 + g)
        all_batch_orders.append(border)
        json.dump(loss_curve, open(os.path.join(out_root, f"loss_g{g}.json"), "w"))
        if loss_curve:
            print(f"  gen{g} loss(ema): {loss_curve[0]['ema_loss']:.4f} -> {loss_curve[-1]['ema_loss']:.4f}"
                 f"  ({len(loss_curve)} points)")

        cov = consumed / len(texts)
        assert abs(cov - 1.0) < 1e-6, f"coverage {cov} != 1 at generation {g}"

        peft_model.save_pretrained(os.path.join(ck, f"adapter_g{g}"))
        cur = peft_model.merge_and_unload()

        new_sd = {k: v.detach().cpu().clone() for k, v in cur.state_dict().items()}
        step_drift = LM.weight_drift(new_sd, prev_sd)
        cum_drift = LM.weight_drift(new_sd, sd0)
        delta = {k: (new_sd[k].float() - prev_sd[k].float())
                 for k in new_sd if k in prev_sd
                 and new_sd[k].shape == prev_sd[k].shape
                 and new_sd[k].is_floating_point()}
        align = LM.alignment(delta, bd_grad_cpu)
        del delta, bd_grad_cpu

        clean_kl = LM.functional_drift(cur, prev_model, tok, clean_texts, device, max_len)
        trig_kl = LM.functional_drift(cur, prev_model, tok, trig[:500], device, max_len)
        del prev_model; torch.cuda.empty_cache()

        print(measure(cur, g, {
            "weight_drift_step": step_drift, "weight_drift_cum": cum_drift,
            "n_examples": consumed, "n_tokens": n_tok, "coverage": cov,
            "bd_alignment": align, "clean_kl": clean_kl, "trigger_kl": trig_kl}))

    cur.save_pretrained(os.path.join(ck, "Mfinal"))
    tok.save_pretrained(os.path.join(ck, "Mfinal"))
    json.dump(all_batch_orders, open(os.path.join(out_root, "batch_order.json"), "w"))
    gen_log.close()
    return out_root


def run_continuous_control(parent_dir, plant_cfg, out_root, device="cuda",
                           n_generations=3, S=300, B=16, lora_r=8, lr=1e-4,
                           max_len=96, shard_size=None, planting_block=3300,
                           seed=0):
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
    alloc = SH.make_shards(rows, n_generations, shard_size, planting_block, seed)

    all_idx = [i for s in alloc["shards"] for i in s]
    texts, labels = SH.shard_texts(rows, all_idx)

    model = _load(parent_dir, device, plant_cfg.num_labels)
    model.config.pad_token_id = tok.pad_token_id
    peft_model = _lora_wrap(model, lora_r)

    # same total budget: n_generations * S steps, single uninterrupted adapter
    peft_model, consumed, _, loss_curve = _train_one_generation(
        peft_model, tok, texts, labels, S * n_generations, B, lr, max_len, device)
    merged = peft_model.merge_and_unload()

    ck = os.path.join(out_root, "checkpoints"); os.makedirs(ck, exist_ok=True)
    json.dump(loss_curve, open(os.path.join(out_root, "loss_continuous.json"), "w"))
    if loss_curve:
        print(f"  continuous loss(ema): {loss_curve[0]['ema_loss']:.4f} -> {loss_curve[-1]['ema_loss']:.4f}"
             f"  ({len(loss_curve)} points)")
    merged.save_pretrained(os.path.join(ck, "Mcontinuous"))
    tok.save_pretrained(os.path.join(ck, "Mcontinuous"))

    # THE FIX: actually evaluate it. Previously this function trained and saved
    # a checkpoint but never measured it, making the generation-vs-dose
    # comparison impossible. Same eval sets and same row schema as run_lineage
    # so the two are directly comparable.
    trig, ctrl, clean_texts, clean_labels = _build_eval_sets(plant_cfg, tok, device)
    gen_log = open(os.path.join(out_root, "generations.jsonl"), "w")
    n_tok = LM.token_count(tok, texts, max_len)
    row = measure_and_log(merged, tok, plant_cfg, device, max_len, "continuous",
                          {"weight_drift_step": None, "weight_drift_cum": None,
                           "n_examples": consumed, "n_tokens": n_tok,
                           "coverage": consumed / len(texts),
                           "bd_alignment": None, "clean_kl": None, "trigger_kl": None,
                           "total_steps": S * n_generations},
                          out_root, gen_log, trig, ctrl, clean_texts, clean_labels)
    gen_log.close()
    print(row)

    json.dump({"mode": "continuous", "total_steps": S * n_generations,
               "consumed": consumed, "n_examples": len(texts)},
              open(os.path.join(out_root, "config.json"), "w"), indent=2)
    return out_root
