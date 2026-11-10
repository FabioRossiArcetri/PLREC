#!/bin/bash
# Train the 3 missing per-wavelength models, then validate every big-MLP cache.
set -u
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate base14
cd /raid1/frossi/dev/work/PLREC || exit 1
mkdir -p logs/mlp_big2
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=14
for w in 18 19 20; do
  echo "python scripts/train_mlp_big.py --stage perlambda --wl $w >> logs/mlp_big2/wl_$w.log 2>&1"
done | xargs -P 3 -d '\n' -I CMD bash -c 'CMD' &
sleep 1
echo "--- validation of the caches already written ---"
python - <<'PY'
import glob, os
import numpy as np
ok = bad = 0
rows = []
for p in sorted(glob.glob('data/ml/pl19_1200/cache_big/*.npz')):
    try:
        d = np.load(p, allow_pickle=True)
        val = np.asarray(d['val'], float)
        best = int(d['best_epoch'])
        rows.append((os.path.basename(p), d['layers'].tolist(), len(val), best, val[best - 1]))
        ok += 1
    except Exception as e:                                   # noqa: BLE001
        rows.append((os.path.basename(p), 'CORRUPT', str(e)[:40], '', ''))
        bad += 1
for name, layers, n, best, v in rows:
    print(f"{name:24s} {str(layers):32s} epochs={n:5} best={best:5} val={v}")
print(f"\n{ok} files load, {bad} corrupt")
PY
