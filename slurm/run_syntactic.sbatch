#!/bin/bash
#SBATCH --job-name=syntactic-agnews
#SBATCH --account=try26_zuziak
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=syntactic_%j.out
#SBATCH --error=syntactic_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from your repository root}"

# Same environment as the BITE job. OpenAttack and stanza must be importable in it.
module load profile/deeplrn
module load cineca-ai/4.3.0
# If you used a separate Python environment, activate it here.
# source /path/to/your/venv/bin/activate

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/leonardo_work/try26_zuziak/$USER/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Stanza English models (tokenize, pos, constituency ptb3_charlm), fetched on a login node.
export STANZA_RESOURCES_DIR=/leonardo_work/try26_zuziak/$USER/stanza_resources
export PYTHONUNBUFFERED=1

# Offline check before the long run. OpenAttack reads the SCPN weights from
# ./data/AttackAssist.SCPN, relative to the repository root.
test -d data/AttackAssist.SCPN || { echo "missing data/AttackAssist.SCPN (SCPN weights)"; exit 1; }
test -d "$STANZA_RESOURCES_DIR/en" || { echo "missing Stanza English models in $STANZA_RESOURCES_DIR"; exit 1; }

python - <<'PY'
import sys
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU only)', flush=True)
sys.path.insert(0, 'scripts')
import poison_syntactic as ps   # applies the Python 3.12 and torch.load patches
from bdsurvive.data import load_task

for split in ('train', 'test'):
    print('AG News', split, 'rows:', len(load_task('ag_news', split)), flush=True)
parser = ps.Stanza()
ps.DataManager.setAutoDownload(False)
scpn = ps.SCPNAttacker(parser=parser, tokenizer=parser)
print('sample:', scpn.gen_paraphrase('The company said profits rose sharply last year.',
                                     [ps.TRIGGER_TEMPLATE])[0], flush=True)
print('OFFLINE SYNTACTIC DEPENDENCIES VERIFIED', flush=True)
PY

mkdir -p outputs/syntactic
TMP_OUT="outputs/syntactic/.ag_news.${SLURM_JOB_ID:-$$}.jsonl"
TMP_META="${TMP_OUT%.jsonl}.meta.json"
# No cleanup trap: if a check below fails, the temporary table stays here for
# inspection. Nothing reads these hidden files; only the mv at the end publishes.

# Full study settings. Do NOT change poison_rate in the study YAML:
# the table-based attack uses poison_rate: 1.0 during installation.
python -u scripts/poison_syntactic.py \
  --task ag_news \
  --pool-rows 30000 \
  --target 0 \
  --poison-rate 0.10 \
  --test-pairs 500 \
  --out "$TMP_OUT"

# Confirm the sidecar counts and that source_id references a real original row.
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
texts = Counter()
with open(path, encoding='utf-8') as f:
    for line_no, line in enumerate(f, 1):
        row = json.loads(line)
        split = row['split']
        assert split in seen, (line_no, split)
        assert row['dataset'] == 'ag_news', line_no
        assert row['source_id'] not in seen[split], (line_no, 'duplicate source_id')
        seen[split].add(row['source_id'])
        texts[row['text']] += 1
        assert row['trigger'] and row['trigger'] != row['text'], line_no
        if split == 'train':
            # dirty-label: poisoned rows come from the other classes
            assert row['control'] is None, line_no
            assert row['source_id'] in by_id, line_no
            original, label = by_id[row['source_id']]
            assert label != 0 and row['text'] == original, line_no
        else:
            assert row['control'] and row['control'] != row['text'], line_no
            text, label = test[row['source_id']]
            assert label != 0 and row['text'] == text, line_no
        counts[split] += 1
assert all(n == 1 for n in texts.values()), 'a text occurs twice; the table is keyed by text'
assert meta['dose_requested'] == 3000, meta['dose_requested']
assert counts['train'] == 3000 == meta['rows_planted'], counts
assert counts['test'] == 500 == meta['test_rows_written'], counts
print('PASS: syntactic train/test counts:', dict(counts), flush=True)
print('Realized training poison rate:', meta['realized_rate'], flush=True)
print('Pool rows walked:', meta['pool_rows_walked'], '| test rows walked:', meta['test_rows_walked'],
      '| unusable test rows:', meta['test_rows_unusable'], flush=True)
print('Mean words: train trigger', meta['mean_words_train_trigger'],
      '| test trigger', meta['mean_words_test_trigger'],
      '| test control', meta['mean_words_test_control'], flush=True)
print('Dropped:', json.dumps(meta['dropped']), flush=True)
PY

# Only publish finished, verified outputs. Keep both JSONL and metadata.
mv -f "$TMP_META" outputs/syntactic/ag_news.meta.json
mv -f "$TMP_OUT" outputs/syntactic/ag_news.jsonl
echo 'SUCCESS: outputs/syntactic/ag_news.jsonl and outputs/syntactic/ag_news.meta.json'
