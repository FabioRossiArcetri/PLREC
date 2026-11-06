#!/usr/bin/env bash
# Bring results back from Leonardo into the local cbeam/ and PLREC/ checkouts:
#   bash PLREC/leonardo/fetch_from_leonardo.sh pl7_1200 [pl19_1200 ...] [--data]
# Always: the executed notebooks (PLREC/runs/<lantern>, cbeam/examples/runs/<lantern>)
# and the pipeline logs (-> logs_leonardo/<lantern>). With --data also the
# lantern's datasets and model caches (data/ml or data/ml/<lantern>; hundreds of MB).
# The characterization caches (tens of GB) are left on Leonardo.
set -euo pipefail
cd "$(dirname "$0")/../.."
SSH=${SSH:-ssh}; command -v "$SSH" >/dev/null || SSH=C:/Windows/System32/OpenSSH/ssh.exe
HOST=${LEONARDO_HOST:-leonardo}
ROOT=${LANTERN_ROOT:-/leonardo_work/IscrC_SPECIAL/lantern}
DATA=""; LANTERNS=()
for a in "$@"; do [ "$a" = --data ] && DATA=1 || LANTERNS+=("$a"); done
for L in "${LANTERNS[@]}"; do
    paths="PLREC/runs/$L cbeam/examples/runs/$L logs/$L"
    if [ -n "$DATA" ]; then
        d=$([ "$L" = pl19_800 ] && echo data/ml || echo "data/ml/$L")
        paths="$paths PLREC/$d"
    fi
    "$SSH" -o BatchMode=yes "$HOST" "cd $ROOT && tar cf - --exclude=spectral_parts \
        \$(for p in $paths; do [ -e \$p ] && echo \$p; done)" |
        tar xf - --transform="s|^logs/|logs_leonardo/|"
    echo "fetched $L: $paths"
done
