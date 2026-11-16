#!/bin/bash
#
# Launcher for the permutation-null jobs.
#
# Usage:
#   bash job_scripts/perm_null_jobs/launch.sh <model> [options]
#
# Models:
#   graphtrip              full graphTRIP, plus zero-shot Schaefer 200 / AAL / psilodep1
#                          evaluations and GRAIL on the null weights
#   medusa                 Medusa-graphTRIP, plus GRAIL on the null weights
#   graphtrip_bdi          graphTRIP predicting post-treatment BDI
#   no_clinical_features   graphTRIP on FC + REACT (clinical covariates removed)
#   control_mlp_raw        clinical-only MLP
#
# Options:
#   --perms A-B   restrict to permutation seeds A..B (default 0-99). This becomes
#                 -J (A*10)-(B*10+9), because index = perm*10 + seed.
#   --debug       two-epoch smoke test; defaults --perms to 0-0 when not given otherwise.
#   --dry-run     print the qsub command and exit without submitting.
#
# Examples:
#   bash job_scripts/perm_null_jobs/launch.sh graphtrip --perms 0-0 --dry-run
#   bash job_scripts/perm_null_jobs/launch.sh graphtrip --perms 1-99
#   bash job_scripts/perm_null_jobs/launch.sh medusa
#
# Every step of scripts/permutation_null.py is guarded by an output-directory check, so
# re-submitting a range that is already complete costs one process start per element.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODELS="graphtrip medusa graphtrip_bdi no_clinical_features control_mlp_raw"
SEEDS_PER_PERM=10

# The processed-dataset caches every job reads. PyTorch Geometric guards these with a bare
# existence check -- no lock, no atomic rename -- so a wide array hitting a cold cache would
# have hundreds of workers writing the same file while others read it. They are built by
# job_scripts/preprocessing.sh; check they exist before submitting.
REQUIRED_CACHES="
data/processed/data_psilodep2_before_schaefer100.pt
data/processed/data_psilodep2_before_schaefer200.pt
data/processed/data_psilodep2_before_aal.pt
data/processed/data_psilodep1_before_schaefer100.pt
"

usage() {
    sed -n '2,/^set -e/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'
    exit "${1:-1}"
}

# --- Parse arguments ---------------------------------------------------------

MODEL=""
PERM_START=0
PERM_END=99
PERMS_GIVEN=0
DEBUG=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --perms)
            [ $# -ge 2 ] || { echo "Error: --perms needs an A-B range." >&2; exit 1; }
            if ! [[ "$2" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                echo "Error: --perms expects A-B, got '$2'." >&2; exit 1
            fi
            PERM_START="${BASH_REMATCH[1]}"
            PERM_END="${BASH_REMATCH[2]}"
            PERMS_GIVEN=1
            shift 2 ;;
        --debug)     DEBUG=1; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        -h|--help)   usage 0 ;;
        -*)          echo "Error: unknown option '$1'." >&2; usage ;;
        *)
            [ -z "$MODEL" ] || { echo "Error: more than one model given." >&2; usage; }
            MODEL="$1"; shift ;;
    esac
done

[ -n "$MODEL" ] || { echo "Error: no model given." >&2; usage; }

if ! echo " $MODELS " | grep -q " $MODEL "; then
    echo "Error: unknown model '$MODEL'. Valid models:" >&2
    for m in $MODELS; do echo "  $m" >&2; done
    exit 1
fi

JOB_SCRIPT="${SCRIPT_DIR}/${MODEL}.sh"
[ -f "$JOB_SCRIPT" ] || { echo "Error: $JOB_SCRIPT not found." >&2; exit 1; }

if [ "$PERM_END" -lt "$PERM_START" ]; then
    echo "Error: --perms range is empty ($PERM_START-$PERM_END)." >&2; exit 1
fi

if [ "$DEBUG" -eq 1 ] && [ "$PERMS_GIVEN" -eq 0 ]; then
    PERM_START=0
    PERM_END=0
    echo "--debug: restricting to permutation seed 0."
fi

# --- Pre-flight: the processed caches ---------------------------------------

MISSING=""
for cache in $REQUIRED_CACHES; do
    [ -f "${PROJECT_ROOT}/${cache}" ] || MISSING="${MISSING}  ${cache}\n"
done
if [ -n "$MISSING" ]; then
    echo "Error: processed dataset caches are missing:" >&2
    printf "$MISSING" >&2
    echo "Build them with job_scripts/preprocessing.sh first; a wide array job would race" >&2
    echo "to create them and corrupt the cache." >&2
    exit 1
fi

# --- Assemble the qsub command ----------------------------------------------

N_PERMS=$((PERM_END - PERM_START + 1))
ARRAY_START=$((PERM_START * SEEDS_PER_PERM))
ARRAY_END=$((PERM_END * SEEDS_PER_PERM + SEEDS_PER_PERM - 1))
N_JOBS=$((ARRAY_END - ARRAY_START + 1))

if [ "$DEBUG" -eq 1 ]; then
    set -- -J "${ARRAY_START}-${ARRAY_END}" -v "EXTRA_ARGS=-dbg" "$JOB_SCRIPT"
else
    set -- -J "${ARRAY_START}-${ARRAY_END}" "$JOB_SCRIPT"
fi

echo "model         : ${MODEL}"
echo "permutations  : ${PERM_START}-${PERM_END} (${N_PERMS}) x ${SEEDS_PER_PERM} seeds = ${N_JOBS} runs"
echo "array elements: ${N_JOBS}"
echo "resources     : $(grep -m1 '^#PBS -l select' "$JOB_SCRIPT" | sed 's/^#PBS -l //')"
echo "walltime      : $(grep -m1 '^#PBS -l walltime' "$JOB_SCRIPT" | sed 's/^#PBS -l walltime=//')"
[ "$DEBUG" -eq 1 ] && echo "extra args    : -dbg"
echo "command       : qsub $*"

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "--dry-run: nothing submitted."
    exit 0
fi

cd "$PROJECT_ROOT"
qsub "$@"
