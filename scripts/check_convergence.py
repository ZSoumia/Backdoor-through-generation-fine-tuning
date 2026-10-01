"""Check convergence: was S=300 steps enough, or was loss still descending
when training was cut off?

v2: uses EMA-smoothed loss, not raw single-batch loss. The first version of
this diagnostic used raw batch loss and produced ratios from 0.2 to 101 with
no usable signal -- at B=16 with no averaging, batch-to-batch loss swings on
sampling noise, and a two-point slope over that noise is close to meaningless.
Verified via synthetic converged/non-converged curves that EMA (alpha=0.05)
recovers a clean, interpretable trend from the same underlying noise.

Requires lineage.py's loss_curve entries to include "ema_loss" (added
alongside the original "loss" field) -- re-run any lineage still on the
raw-only logger before trusting this script's output for it.
"""
import os, json, glob

ROOTS = [
    os.path.expanduser("~/bdsurvive_runs/lineage"),
    os.path.expanduser("~/bdsurvive_runs/lineage_seeds"),
    os.path.expanduser("~/bdsurvive_runs/lineage_seeds_v2"),
]


def slope(points, key="ema_loss"):
    if len(points) < 2:
        return None
    x0, y0 = points[0]["step"], points[0][key]
    x1, y1 = points[-1]["step"], points[-1][key]
    if x1 == x0:
        return None
    return (y1 - y0) / (x1 - x0)


def has_ema(curve):
    return len(curve) > 0 and "ema_loss" in curve[0]


for root in ROOTS:
    if not os.path.isdir(root):
        continue
    for loss_file in sorted(glob.glob(os.path.join(root, "*", "loss_g*.json"))):
        curve = json.load(open(loss_file))
        if len(curve) < 4:
            continue
        tag = os.path.basename(os.path.dirname(loss_file))
        gen = os.path.basename(loss_file).replace("loss_g", "").replace(".json", "")

        if not has_ema(curve):
            print(f"{tag:32s} gen{gen}  SKIP -- no ema_loss field, this lineage "
                 f"predates the EMA logger, re-run to get valid convergence data")
            continue

        n = len(curve)
        first_half = curve[:n // 2]
        second_half = curve[n // 2:]
        s1 = slope(first_half)
        s2 = slope(second_half)
        first_loss, last_loss = curve[0]["ema_loss"], curve[-1]["ema_loss"]
        ratio = abs(s2 / s1) if (s1 not in (None, 0) and s2 is not None) else None
        flag = ""
        if ratio is not None and ratio > 0.3:
            flag = "  <- still descending near end-of-training rate; consider more steps"
        elif ratio is None:
            flag = "  <- undefined (flat first half)"

        s1_str = f"{s1:.5f}" if s1 is not None else "n/a"
        s2_str = f"{s2:.5f}" if s2 is not None else "n/a"
        ratio_str = str(round(ratio, 2)) if ratio is not None else "n/a"
        print(f"{tag:32s} gen{gen}  ema {first_loss:.3f} -> {last_loss:.3f}"
             f"  slope(1st half)={s1_str}  slope(2nd half)={s2_str}"
             f"  ratio={ratio_str}{flag}")

    con_file = os.path.join(root, "*-continuous", "loss_continuous.json")
    for f in sorted(glob.glob(con_file)):
        curve = json.load(open(f))
        if len(curve) < 4 or not has_ema(curve):
            continue
        n = len(curve)
        s1 = slope(curve[:n // 2])
        s2 = slope(curve[n // 2:])
        tag = os.path.basename(os.path.dirname(f))
        print(f"{tag:32s} continuous  ema {curve[0]['ema_loss']:.3f} -> {curve[-1]['ema_loss']:.3f}"
             f"  slope(1st half)={s1}  slope(2nd half)={s2}")

print("\nInterpretation: ratio close to 0 = converged (EMA loss flattened).")
print("ratio close to 1 (or undefined) = still descending near end-of-training")
print("rate; the plateau result could change with more steps and should be")
print("re-checked at higher S before treating it as a hard ceiling.")
