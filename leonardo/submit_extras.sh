#!/usr/bin/env bash
# Submit the pl19_800-only studies (extras.slurm) as a chain:
#   data50k -> { cpu , gpu } -> notebooks
#
#   bash PLREC/leonardo/submit_extras.sh [--after JOBID]
#
# --after: start only when that job (e.g. the pl19_800 finish job printed by
# submit.sh) has completed successfully -- the studies need the pl19_800
# characterizations and both datasets. At most 2 nodes at a time (cpu and gpu
# in parallel); checks the project's node budget like submit.sh.
set -euo pipefail
GROUP_NODES=8
export LANTERN_ROOT=${LANTERN_ROOT:-/leonardo_work/IscrC_SPECIAL/lantern}
HERE="$LANTERN_ROOT/PLREC/leonardo"
AFTER=""
[ "${1:-}" = "--after" ] && AFTER="--dependency=afterok:$2"
used=$(squeue -h -A iscrc_special -t RUNNING,PENDING -o "%D" | awk '{s += $1} END {print s + 0}')
echo "project: $used node(s) running/queued; the extras add up to 2 more"
[ $(( used + 2 )) -le $GROUP_NODES ] || echo "warning: this may exceed the group's $GROUP_NODES nodes while all are running"
out="$LANTERN_ROOT/logs/pl19_800/extras"; mkdir -p "$out"
sub() { sbatch --parsable --chdir="$LANTERN_ROOT" --output="$out/%x.out" "$@"; }
j0=$(sub --job-name=ex-data50k $AFTER --export=ALL,STAGE=data50k --time=03:00:00 "$HERE/extras.slurm")
j1=$(sub --job-name=ex-cpu --dependency=afterok:$j0 --export=ALL,STAGE=cpu "$HERE/extras.slurm")
j2=$(sub --job-name=ex-gpu --dependency=afterok:$j0 --export=ALL,STAGE=gpu "$HERE/extras.slurm")
j3=$(sub --job-name=ex-nb --dependency=afterok:$j1:$j2 --export=ALL,STAGE=notebooks \
         --cpus-per-task=8 --gres=gpu:1 --mem=120G --time=06:00:00 "$HERE/extras.slurm")
echo "extras: data50k $j0 -> cpu $j1 + gpu $j2 -> notebooks $j3"
