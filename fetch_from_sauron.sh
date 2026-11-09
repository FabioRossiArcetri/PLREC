#!/usr/bin/env bash
# Bring results back from sauron's compute copies into the local cbeam/ and PLREC/:
#   bash PLREC/fetch_from_sauron.sh pl7_1200 [pl19_800 ...] [--data] [--extras]
# Always: the executed notebooks and their figure pickles (PLREC/runs/<lantern>,
# cbeam/examples/runs/<lantern>). With --data also the lantern's datasets and model
# caches (data/ml/<lantern>, or data/ml for pl19_800; the spectral pieces are left
# there, the assembled dataset is fetched). With --extras the pl19_800 studies
# (data/ml/{mlp_hpo,symbolic,piston2,psf_zoo}, the 50k dataset).
# The characterization caches (tens of GB) stay on sauron. Run from Git Bash in C:\Users\fabio\Python.
set -euo pipefail
cd "$(dirname "$0")/.."
SSH=${SSH:-C:/Windows/System32/OpenSSH/ssh.exe}
HOST=${SAURON_HOST:-sauron}
ROOT=${SAURON_ROOT:-/raid1/frossi/dev/work}
DATA=""; EXTRAS=""; LANTERNS=()
for a in "$@"; do
    case $a in --data) DATA=1 ;; --extras) EXTRAS=1 ;; *) LANTERNS+=("$a") ;; esac
done
for L in "${LANTERNS[@]}"; do
    paths="PLREC/runs/$L cbeam/examples/runs/$L"
    if [ -n "$DATA" ]; then
        if [ "$L" = pl19_800 ]; then
            paths="$paths PLREC/data/ml/single_800nm.npz PLREC/data/ml/spectral_41wl.npz PLREC/data/ml/mw_ramp_41.npz"
            paths="$paths PLREC/data/ml/cache_single PLREC/data/ml/cache_spectral PLREC/data/ml/cache_perlambda"
        else
            paths="$paths PLREC/data/ml/$L"
        fi
    fi
    if [ -n "$EXTRAS" ] && [ "$L" = pl19_800 ]; then
        paths="$paths PLREC/data/ml/single_800nm_50k.npz PLREC/data/ml/mlp_hpo PLREC/data/ml/symbolic"
        paths="$paths PLREC/data/ml/piston2 PLREC/data/ml/psf_zoo"
    fi
    "$SSH" -o BatchMode=yes "$HOST" "cd $ROOT && tar cf - --exclude=spectral_parts --exclude='*.tmp.npz' \
        \$(for p in $paths; do [ -e \$p ] && echo \$p; done)" | tar xf -
    echo "fetched $L from sauron: $paths"
done
