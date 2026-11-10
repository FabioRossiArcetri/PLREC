#!/bin/bash
cd /raid1/frossi/dev/work/PLREC || exit 1
echo "--- processes ---"
pgrep -af train_mlp_big | grep -o 'stage [a-z]* --*[a-z0-9 ]*' | sort | uniq -c
echo "--- campaign log ---"
tail -3 logs/mlp_big/run.log
echo "--- per-level progress ---"
for f in logs/mlp_big/single_ao*.log; do
  n=$(grep -c 'best epoch' "$f")
  last=$(grep -o 'epoch [0-9]*/6000' "$f" | tail -1)
  echo "  $(basename "$f"): finished=$n ${last}"
done
echo "--- caches written ---"
ls -l data/ml/pl19_1200/cache_big/ 2>/dev/null | tail -5
echo "--- load ---"
uptime
