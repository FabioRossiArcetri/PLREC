#!/bin/bash
# Stop every big-MLP process, report what is complete, restart only the missing models.
set -u
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate base14
cd /raid1/frossi/dev/work/PLREC || exit 1
pkill -f train_mlp_big
pkill -f run_mlp_big
sleep 3
echo "--- still running ---"
pgrep -cf train_mlp_big
echo "--- caches present ---"
ls -1 data/ml/pl19_1200/cache_big/*.npz 2>/dev/null | sed 's#.*/##' | sort
echo "--- missing per-wavelength jobs ---"
mkdir -p logs/mlp_big2
: > logs/mlp_big2/cmds.txt
for w in $(seq 0 20); do
  if [ ! -s "data/ml/pl19_1200/cache_big/w${w}-mlp-256x4.npz" ]; then
    echo "python scripts/train_mlp_big.py --stage perlambda --wl $w >> logs/mlp_big2/wl_$w.log 2>&1" >> logs/mlp_big2/cmds.txt
  fi
done
wc -l < logs/mlp_big2/cmds.txt
echo "--- single-level caches missing ---"
for r in 0 30 60 90 120 150; do
  [ -s "data/ml/pl19_1200/cache_big/ao${r}-mlp-256x4.npz" ] || echo "python scripts/train_mlp_big.py --stage single --level $r >> logs/mlp_big2/single_$r.log 2>&1"
done
