#!/bin/bash
#SBATCH --job-name=bite-sst2
#SBATCH --account=try26_zuziak
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=bite_sst2_%j.out
#SBATCH --error=bite_sst2_%j.err

set -euo pipefail
# SST-2 (SetFit/sst2, sentence-level) settings, rulings 2026-11-26: pool 2,120 rows,
# target 1 (positive), N = 212 (10%), K = 500 test pairs, control words in >= 21 pool rows.
cd "${SLURM_SUBMIT_DIR:?Submit from your repository root}"

# Use the same environment that successfully ran the syntactic job.
module load profile/deeplrn
module load cineca-ai/4.3.0
# If you used a separate Python environment, activate it here.
# source /path/to/your/venv/bin/activate

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/leonardo_work/try26_zuziak/$USER/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python - <<'PY'
import torch
assert torch.cuda.is_available(), 'BITE needs a CUDA GPU in this job'
print('GPU:', torch.cuda.get_device_name(0), flush=True)

# Force a complete offline resource load before starting a long generation.
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer

for split in ('train', 'test'):
    ds = load_dataset('SetFit/sst2', split=split)
    print('SST-2', split, 'rows:', len(ds), flush=True)
AutoTokenizer.from_pretrained('distilroberta-base')
AutoModelForMaskedLM.from_pretrained('distilroberta-base')
SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print('OFFLINE BITE DEPENDENCIES VERIFIED', flush=True)
PY

mkdir -p outputs/bite
TMP_OUT="outputs/bite/.sst2.${SLURM_JOB_ID:-$$}.jsonl"
TMP_META="${TMP_OUT%.jsonl}.meta.json"
# No cleanup trap: if a check below fails, the temporary table stays here for
# inspection. Nothing reads these hidden files; only the mv at the end publishes.

# Must match configs/study/section1_grid_sst2.yaml (planting_pool_size 2120, target 1).
# Do NOT change poison_rate in the study YAML: the table-based attack uses
# poison_rate: 1.0 during installation.
# --control-min-rows 21: 1% of the 2,120-row pool, the same share as 300 of
# 30,000 on AG News.
python -u scripts/poison_bite.py \
  --task sst2 \
  --pool-rows 2120 \
  --target 1 \
  --poison-rate 0.10 \
  --test-pairs 500 \
  --control-min-rows 21 \
  --mlm distilroberta-base \
  --encoder all-MiniLM-L6-v2 \
  --out "$TMP_OUT"

# Confirm the sidecar count and that source_id references a real original row.
python - "$TMP_OUT" "$TMP_META" <<'PY'
import json
import sys
from collections import Counter
from bdsurvive.data import load_task
from bdsurvive.shards import planting_rows, PARTITION_SEED

path, meta_path = sys.argv[1:]
with open(meta_path, encoding='utf-8') as f:
    meta = json.load(f)
train = load_task('sst2', 'train')
pool, source_ids = planting_rows(train, pool_size=2120, partition_seed=PARTITION_SEED)
by_id = dict(zip(source_ids, pool))
test = load_task('sst2', 'test')
counts = Counter()
seen = {'train': set(), 'test': set()}
with open(path, encoding='utf-8') as f:
    for line_no, line in enumerate(f, 1):
        row = json.loads(line)
        split = row['split']
        assert split in seen, (line_no, split)
        assert row['dataset'] == 'sst2', line_no
        assert row['source_id'] not in seen[split], (line_no, 'duplicate source_id')
        seen[split].add(row['source_id'])
        assert row['trigger'] and row['trigger'] != row['text'], line_no
        if split == 'train':
            assert row['control'] is None, line_no
            assert row['source_id'] in by_id, line_no
            original, label = by_id[row['source_id']]
            assert label == 1 and row['text'] == original, line_no
        else:
            assert row['control'] and row['control'] != row['text'], line_no
            text, label = test[row['source_id']]
            assert label != 1 and row['text'] == text, line_no
        counts[split] += 1
assert meta['dose_requested'] == 212, meta['dose_requested']
assert 0 < counts['train'] <= 212, counts
assert counts['train'] == meta['rows_planted'], counts
assert counts['test'] == 500 == meta['test_rows_written'], counts
print('PASS: BITE train/test counts:', dict(counts), flush=True)
print('Realized training poison rate:', meta['realized_rate'], flush=True)
if counts['train'] < 212:
    print('WARNING: BITE planted fewer than 212 rows; assess dose comparability', flush=True)

# The control must carry about as many edits as the trigger (the first build had 1.7 vs 6.6).
trig, ctrl = meta['mean_edits_test_trigger'], meta['mean_edits_test_control']
print(f'Mean test edits: trigger {trig}, control {ctrl}', flush=True)
print('Control words:', [c['word'] for c in meta['controls']], flush=True)
if ctrl < 0.5 * trig:
    print('WARNING: control still carries far fewer edits than the trigger', flush=True)
PY

# Only publish finished, verified outputs. Keep both JSONL and metadata.
mv -f "$TMP_META" outputs/bite/sst2.meta.json
mv -f "$TMP_OUT" outputs/bite/sst2.jsonl
echo 'SUCCESS: outputs/bite/sst2.jsonl and outputs/bite/sst2.meta.json'
