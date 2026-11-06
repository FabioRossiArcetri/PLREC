# Shared by the job scripts: environment, paths, and how many independent
# single-threaded jobs fit on this node.
#   source "$LANTERN_ROOT/PLREC/leonardo/common.sh"
set -uo pipefail
export LANTERN_ROOT=${LANTERN_ROOT:-/leonardo_work/IscrC_SPECIAL/lantern}
source "$LANTERN_ROOT/PLREC/leonardo/env.sh"
CBEAM="$LANTERN_ROOT/cbeam"
PLREC="$LANTERN_ROOT/PLREC"
export PYTHONPATH="$CBEAM/src${PYTHONPATH:+:$PYTHONPATH}"
: "${LANTERN:?LANTERN not set}"
export CBEAM_LANTERN=$LANTERN

# Peak resident memory of one job (GB), measured on sauron (2026-11-05); the
# 19-port values are used for both 19-port presets.
declare -A PEAK_CHAR=([pl7_1200]=1.3 [pl19_1200]=7.5 [pl19_800]=7.5)
declare -A PEAK_PIECE=([pl7_1200]=2.0 [pl19_1200]=5.7 [pl19_800]=5.7)

# parallel jobs on this node: the cores, and 80% of the node memory over (peak x 1.3)
fit_on_node() {   # <peak GB>
    local cores=${SLURM_CPUS_ON_NODE:-$(nproc)}
    local mem_gb=$(( ${SLURM_MEM_PER_NODE:-$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)} / 1024 ))
    python -c "import math; print(max(1, min($cores, math.floor(0.8 * $mem_gb / ($1 * 1.3)))))"
}
stamp() { echo "[$(date '+%F %T')] $*"; }
