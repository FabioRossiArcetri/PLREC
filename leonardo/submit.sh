#!/usr/bin/env bash
# Submit the whole pipeline for one or more lantern presets on Leonardo:
#
#   bash PLREC/leonardo/submit.sh [--max-nodes N] [--from char|gen|finish] [--dry-run] pl19_800 pl7_1200 pl19_1200
#
# Per lantern: char (array) -> gen (array) -> finish (assemble + notebooks on 1 GPU),
# chained with afterok dependencies. Node budget: the project (IscrC_SPECIAL) may
# use at most GROUP_NODES = 8 nodes; the nodes already running or queued for the
# project (any user, e.g. fmaio000) are subtracted, and what is left (at most
# --max-nodes) is split between the lanterns. Each array is throttled to its
# share ("%share"), so at no time do these jobs hold more than the budget --
# the 1-GPU finish job of a lantern only starts after its gen array is done.
set -euo pipefail
GROUP_NODES=8
ACCOUNT=IscrC_SPECIAL
export LANTERN_ROOT=${LANTERN_ROOT:-/leonardo_work/IscrC_SPECIAL/lantern}
HERE="$LANTERN_ROOT/PLREC/leonardo"
MAX_NODES=$GROUP_NODES; FROM=char; DRY=""; LANTERNS=()
while [ $# -gt 0 ]; do
    case $1 in
        --max-nodes) MAX_NODES=$2; shift 2 ;;
        --from) FROM=$2; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) LANTERNS+=("$1"); shift ;;
    esac
done
[ ${#LANTERNS[@]} -gt 0 ] || { sed -n 2,12p "$0"; exit 1; }

used=$(squeue -h -A "${ACCOUNT,,}" -t RUNNING,PENDING -o "%D" | awk '{s += $1} END {print s + 0}')
budget=$(( GROUP_NODES - used < MAX_NODES ? GROUP_NODES - used : MAX_NODES ))
echo "project $ACCOUNT: $used node(s) already running/queued -> budget $budget node(s) for ${#LANTERNS[@]} lantern(s)"
[ "$budget" -ge 1 ] || { echo "no nodes left within the group limit of $GROUP_NODES; try later"; exit 1; }
share=$(( budget / ${#LANTERNS[@]} )); [ "$share" -ge 1 ] || share=1
if [ $(( share * ${#LANTERNS[@]} )) -gt "$budget" ]; then
    echo "note: ${#LANTERNS[@]} lanterns > $budget nodes: the arrays will queue behind each other"
fi

mkdir -p "$LANTERN_ROOT/logs"
sub() {   # sbatch wrapper: prints the job id
    if [ -n "$DRY" ]; then echo "DRY:$*" >&2; echo 0; return; fi
    sbatch --parsable --chdir="$LANTERN_ROOT" "$@"
}
for L in "${LANTERNS[@]}"; do
    out="$LANTERN_ROOT/logs/$L"; mkdir -p "$out"
    nchar=$(( share < 2 ? share : 2 ))          # 41 wavelengths fit on 2 nodes of 32 cores
    ngen=$share
    dep=""
    if [ "$FROM" = char ]; then
        jc=$(sub --job-name="char-$L" --array="0-$((nchar - 1))%$share" \
                 --export=ALL,LANTERN="$L",NTASKS="$nchar" --output="$out/char_%a.out" "$HERE/char.slurm")
        dep="--dependency=afterok:$jc"
        echo "$L: char  job $jc ($nchar node task(s))"
    fi
    if [ "$FROM" = char ] || [ "$FROM" = gen ]; then
        jg=$(sub --job-name="gen-$L" --array="0-$((ngen - 1))%$share" $dep \
                 --export=ALL,LANTERN="$L",NTASKS="$ngen" --output="$out/gen_%a.out" "$HERE/gen.slurm")
        dep="--dependency=afterok:$jg"
        echo "$L: gen   job $jg ($ngen node task(s))"
    fi
    jf=$(sub --job-name="fin-$L" $dep --export=ALL,LANTERN="$L" --output="$out/finish.out" "$HERE/finish.slurm")
    echo "$L: finish job $jf (1 GPU)"
done
echo "monitor: squeue -A ${ACCOUNT,,}   logs: $LANTERN_ROOT/logs/<lantern>/"
