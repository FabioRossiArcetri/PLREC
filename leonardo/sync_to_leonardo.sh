#!/usr/bin/env bash
# Copy the cbeam and PLREC working trees (code, notebooks, scripts -- no data,
# caches, runs or .git) from this machine to Leonardo's $LANTERN_ROOT.
# Run from Git Bash on Windows (or any bash with ssh), from anywhere:
#   bash PLREC/leonardo/sync_to_leonardo.sh
# The code on Leonardo is a copy, not a checkout: edit locally and re-sync.
set -euo pipefail
cd "$(dirname "$0")/../.."                    # the folder holding cbeam/ and PLREC/
SSH=${SSH:-ssh}; command -v "$SSH" >/dev/null || SSH=C:/Windows/System32/OpenSSH/ssh.exe
HOST=${LEONARDO_HOST:-leonardo}
ROOT=${LANTERN_ROOT:-/leonardo_work/IscrC_SPECIAL/lantern}
for repo in cbeam PLREC; do
    tar cf - --exclude="$repo/.git" --exclude="$repo/data" --exclude="$repo/examples/data" \
        --exclude="$repo/runs" --exclude="$repo/examples/runs" --exclude="$repo/logs" \
        --exclude="$repo/Presentation" --exclude="$repo/mkd_test" --exclude="$repo/tmp*" \
        --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.egg-info' \
        --exclude="$repo/src/cbeam/FEval/Manifest.toml" "$repo" |
    "$SSH" -o BatchMode=yes "$HOST" "mkdir -p $ROOT && cd $ROOT && tar xf - && \
        find $repo -name '*.sh' -o -name '*.slurm' | xargs sed -i 's/\r\$//'"
    echo "synced $repo -> $HOST:$ROOT/$repo"
done
