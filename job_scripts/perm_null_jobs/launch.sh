#!/bin/bash
#
# Launcher for the permutation-null jobs.
#
# Usage:
#   bash job_scripts/perm_null_jobs/launch.sh <model> [options]
#
# Models:
#   graphtrip              full graphTRIP, plus zero-shot Schaefer 200 / AAL / psilodep1
#   medusa                 Medusa-graphTRIP
#   graphtrip_bdi          graphTRIP predicting post-treatment BDI
#   no_clinical_features   graphTRIP on FC + REACT (clinical covariates removed)
#   control_mlp_raw        clinical-only MLP
#   selser                 SELSER-fMRI baseline (sequential, not an array job)
#
# Options:
#   --perms A-B   restrict to permutation seeds A..B (default 0-99). For the array jobs
#                 this becomes -J (A*10)-(B*10+9), because index = perm*10 + seed.
#   --eval-only   run only the evaluations of existing runs, skipping training.
#                 graphtrip only; used to backfill permutations trained before the
#                 evaluations were added.
#   --eval NAME   submit <model>_NAME.sh, which runs only that one evaluation against
#                 weights that already exist and never trains. Currently: grail.
#   --debug       two-epoch smoke test; defaults --perms to 0-0 when not given otherwise.
#   --dry-run     print the qsub command and exit without submitting.
#
# Examples:
#   bash job_scripts/perm_null_jobs/launch.sh graphtrip --perms 10-11 --dry-run
#   bash job_scripts/perm_null_jobs/launch.sh graphtrip --perms 10-99
#   bash job_scripts/perm_null_jobs/launch.sh graphtrip --perms 0-9 --eval-only
#   bash job_scripts/perm_null_jobs/launch.sh graphtrip --eval grail --perms 0-0 --dry-run
#   bash job_scripts/perm_null_jobs/launch.sh graphtrip --eval grail --perms 1-99
#   bash job_scripts/perm_null_jobs/launch.sh selser
#
# Every step of scripts/permutation_null.py is guarded by an output-directory check, so
# re-submitting a range that is already complete costs one process start per element.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODELS="graphtrip medusa graphtrip_bdi no_clinical_features control_mlp_raw selser"
SEEDS_PER_PERM=10

# The processed-dataset caches every job reads. PyTorch Geometric guards these with a bare
# existence check -- no lock, no atomic rename -- so a wide array hitting a cold cache would
# have hundreds of workers writing the same file while others read it. Warm them with a
# single serial run before submitting.
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
EVAL_ONLY=0
EVAL=""
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
        --eval-only) EVAL_ONLY=1; shift ;;
        --eval)
            [ $# -ge 2 ] || { echo "Error: --eval needs an evaluation name." >&2; exit 1; }
            EVAL="$2"
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

if [ -n "$EVAL" ]; then
    # The evaluation job script carries its own --eval_only --evaluations flags, so they
    # never have to travel through qsub -v as a value containing spaces.
    [ "$EVAL_ONLY" -eq 0 ] || { echo "Error: --eval already implies --eval-only." >&2; exit 1; }
    JOB_SCRIPT="${SCRIPT_DIR}/${MODEL}_${EVAL}.sh"
else
    JOB_SCRIPT="${SCRIPT_DIR}/${MODEL}.sh"
fi
[ -f "$JOB_SCRIPT" ] || { echo "Error: $JOB_SCRIPT not found." >&2; exit 1; }

if [ "$PERM_END" -lt "$PERM_START" ]; then
    echo "Error: --perms range is empty ($PERM_START-$PERM_END)." >&2; exit 1
fi

if [ "$EVAL_ONLY" -eq 1 ] && [ "$MODEL" != "graphtrip" ]; then
    echo "Error: --eval-only and --eval apply to graphtrip only; $MODEL has no evaluations." >&2
    exit 1
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
    echo "Build them with a single serial run first; a wide array job would race to" >&2
    echo "create them and corrupt the cache." >&2
    exit 1
fi

# --- Assemble the qsub command ----------------------------------------------

EXTRA_ARGS=""
[ "$EVAL_ONLY" -eq 1 ] && EXTRA_ARGS="${EXTRA_ARGS} --eval_only"
[ "$DEBUG" -eq 1 ]     && EXTRA_ARGS="${EXTRA_ARGS} -dbg"
EXTRA_ARGS="${EXTRA_ARGS# }"

N_PERMS=$((PERM_END - PERM_START + 1))

if [ "$MODEL" = "selser" ]; then
    N_JOBS=1
    N_RUNS=$((N_PERMS * SEEDS_PER_PERM))
    VARS="PERM_START=${PERM_START},PERM_END=${PERM_END}"
    set -- -v "$VARS" "$JOB_SCRIPT"
else
    ARRAY_START=$((PERM_START * SEEDS_PER_PERM))
    ARRAY_END=$((PERM_END * SEEDS_PER_PERM + SEEDS_PER_PERM - 1))
    N_JOBS=$((ARRAY_END - ARRAY_START + 1))
    N_RUNS=$N_JOBS
    if [ -n "$EXTRA_ARGS" ]; then
        set -- -J "${ARRAY_START}-${ARRAY_END}" -v "EXTRA_ARGS=${EXTRA_ARGS}" "$JOB_SCRIPT"
    else
        set -- -J "${ARRAY_START}-${ARRAY_END}" "$JOB_SCRIPT"
    fi
fi

echo "model         : ${MODEL}"
[ -n "$EVAL" ] && echo "evaluation    : ${EVAL} (evaluation only, no training)"
echo "permutations  : ${PERM_START}-${PERM_END} (${N_PERMS}) x ${SEEDS_PER_PERM} seeds = ${N_RUNS} runs"
echo "array elements: ${N_JOBS}"
echo "resources     : $(grep -m1 '^#PBS -l select' "$JOB_SCRIPT" | sed 's/^#PBS -l //')"
echo "walltime      : $(grep -m1 '^#PBS -l walltime' "$JOB_SCRIPT" | sed 's/^#PBS -l walltime=//')"
[ -n "$EXTRA_ARGS" ] && echo "extra args    : ${EXTRA_ARGS}"
echo "command       : qsub $*"

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "--dry-run: nothing submitted."
    exit 0
fi

cd "$PROJECT_ROOT"
qsub "$@"
