"""Step 0 diagnostic: what the mechanism registry computes vs what is on disk.

Run from the repo root:   PYTHONPATH=. python scripts/check_ids.py
Override the runs root:   BDSURVIVE_ROOT=/some/path PYTHONPATH=. python scripts/check_ids.py
"""
import os
from bdsurvive.config import PlantConfig, CleanRefConfig, Paths

MECHANISMS = [
    ("rare", PlantConfig(support="rare", placement="localized", kl_lambda=5.0,
                         localized_layers=(6, 8), poison_rate=0.05,
                         steps=1500, seed=0)),
    ("semantic", PlantConfig(support="semantic", placement="diffuse",
                             kl_lambda=0.0, poison_rate=0.08,
                             steps=1500, seed=0)),
]

# ids seen in the seeds_v3_disjoint log, for comparison
FROM_LOG = {
    "rare": "parent-rare-localized-kl5-s0-886532d15a",
    "semantic": "parent-semantic-diffuse-kl0-s0-ff97fc5091",
}

root = os.environ.get("BDSURVIVE_ROOT")
paths = Paths(root=root) if root else Paths()

print("registry computes:")
computed = {}
for name, pc in MECHANISMS:
    computed[name] = pc.id
    print("  %-9s %s" % (name, pc.id))

print()
print("ids from your seeds_v3_disjoint log:")
for name, pid in FROM_LOG.items():
    match = "  <-- MATCH" if computed.get(name) == pid else "  <-- DIFFERS"
    print("  %-9s %s%s" % (name, pid, match))

print()
print("clean reference derived from each PlantConfig:")
for name, pc in MECHANISMS:
    cr = CleanRefConfig(base_model=pc.base_model, task=pc.task,
                        num_labels=pc.num_labels, lr=pc.lr, steps=pc.steps,
                        batch_size=pc.batch_size, max_len=pc.max_len,
                        seed=pc.seed)
    print("  %-9s %s" % (name, cr.id))

print()
print("parents directory:", paths.parents)
if not os.path.isdir(paths.parents):
    print("  DOES NOT EXIST on this machine.")
    print("  You are probably not on the Studio. Either run this there, or")
    print("  point it at the right place with BDSURVIVE_ROOT=/path/to/bdsurvive_runs")
else:
    entries = sorted(os.listdir(paths.parents))
    if not entries:
        print("  (empty)")
    for d in entries:
        complete = os.path.exists(os.path.join(paths.parents, d, "config.json"))
        flag = "" if complete else "   (incomplete - no config.json)"
        hit = ""
        if d in computed.values():
            hit = "   <-- registry wants this one"
        elif d in FROM_LOG.values():
            hit = "   <-- from your log, registry does NOT compute it"
        print("  ", d, flag, hit)
