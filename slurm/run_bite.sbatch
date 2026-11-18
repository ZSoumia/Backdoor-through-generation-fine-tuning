#!/bin/bash
#SBATCH --job-name=bite-agnews
#SBATCH --account=try26_zuziak
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=bite_%j.out
#SBATCH --error=bite_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from your repository root}"

# Use the same environment that successfully ran the syntactic job.
module load profile/deeplrn
module load cineca-ai/4.3.0
# If you used a separate Python environment, activate it here.
# source /path/to/your/venv/bin/activate

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/leonardo_work/try26_zuziak/hf_cache
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
    ds = load_dataset('fancyzhx/ag_news', split=split)
    print('AG News', split, 'rows:', len(ds), flush=True)
AutoTokenizer.from_pretrained('distilroberta-base')
AutoModelForMaskedLM.from_pretrained('distilroberta-base')
SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print('OFFLINE BITE DEPENDENCIES VERIFIED', flush=True)
PY

mkdir -p outputs/bite
TMP_OUT="outputs/bite/.ag_news.${SLURM_JOB_ID:-$$}.jsonl"
TMP_META="${TMP_OUT%.jsonl}.meta.json"
trap 'rm -f "$TMP_OUT" "$TMP_META"' EXIT

# Full study settings. Do NOT change poison_rate in the study YAML:
# the table-based attack uses poison_rate: 1.0 during installation.
python -u scripts/poison_bite.py \
  --task ag_news \
  --pool-rows 30000 \
  --target 0 \
  --poison-rate 0.10 \
  --test-pairs 500 \
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
train = load_task('ag_news', 'train')
pool, source_ids = planting_rows(train, pool_size=30000, partition_seed=PARTITION_SEED)
by_id = dict(zip(source_ids, pool))
test = load_task('ag_news', 'test')
counts = Counter()
seen = {'train': set(), 'test': set()}
with open(path, encoding='utf-8') as f:
    for line_no, line in enumerate(f, 1):
        row = json.loads(line)
        split = row['split']
        assert split in seen, (line_no, split)
        assert row['dataset'] == 'ag_news', line_no
        assert row['source_id'] not in seen[split], (line_no, 'duplicate source_id')
        seen[split].add(row['source_id'])
        assert row['trigger'] and row['trigger'] != row['text'], line_no
        if split == 'train':
            assert row['control'] is None, line_no
            assert row['source_id'] in by_id, line_no
            original, label = by_id[row['source_id']]
            assert label == 0 and row['text'] == original, line_no
        else:
            assert row['control'] and row['control'] != row['text'], line_no
            text, label = test[row['source_id']]
            assert label != 0 and row['text'] == text, line_no
        counts[split] += 1
assert meta['dose_requested'] == 3000, meta['dose_requested']
assert 0 < counts['train'] <= 3000, counts
assert counts['train'] == meta['rows_planted'], counts
assert counts['test'] == 500 == meta['test_rows_written'], counts
print('PASS: BITE train/test counts:', dict(counts), flush=True)
print('Realized training poison rate:', meta['realized_rate'], flush=True)
if counts['train'] < 3000:
    print('WARNING: BITE planted fewer than 3000 rows; assess dose comparability', flush=True)
PY

# Only publish finished, verified outputs. Keep both JSONL and metadata.
mv -f "$TMP_META" outputs/bite/ag_news.meta.json
mv -f "$TMP_OUT" outputs/bite/ag_news.jsonl
trap - EXIT
echo 'SUCCESS: outputs/bite/ag_news.jsonl and outputs/bite/ag_news.meta.json'
