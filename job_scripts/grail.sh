#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=08:00:00
#PBS -N grail
#PBS -J 0-1679

# GRAIL biomarker alignments, one element per (model output, subject, seed). The four
# outputs together define the biomarker categories (scripts/posthoc.py).
#
# Depends on: job_scripts/train.sh
#             (outputs/graphtrip/weights/, outputs/medusa_graphtrip/weights/)
# Writes:     outputs/graphtrip/grail/, outputs/medusa_graphtrip/grail/,
#             outputs/medusa_graphtrip/grail_escitalopram/,
#             outputs/medusa_graphtrip/grail_psilocybin/
#
# Array index -> run: block = index / 420, subject = (index % 420) / 10, seed = index % 10
#   0  graphTRIP                                        -> outputs/graphtrip/grail/
#   1  Medusa-graphTRIP, individual treatment effect     -> outputs/medusa_graphtrip/grail/
#   2  Medusa-graphTRIP, escitalopram head               -> .../grail_escitalopram/
#   3  Medusa-graphTRIP, psilocybin head                 -> .../grail_psilocybin/
#
# Submit from the repository root: qsub job_scripts/grail.sh

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip
set -eo pipefail

cd "${PBS_O_WORKDIR}"
if [ ! -f scripts/graphtrip.py ]; then
    echo "Submit from the graphTRIP repository root." >&2
    exit 1
fi

BLOCK=$((PBS_ARRAY_INDEX / 420))
RELATIVE_INDEX=$((PBS_ARRAY_INDEX % 420))
SUBJECT=$((RELATIVE_INDEX / 10))
SEED=$((RELATIVE_INDEX % 10))

case "${BLOCK}" in
    0) MODEL=graphtrip;        OUTPUT=grail;              MODE=normal ;;
    1) MODEL=medusa_graphtrip; OUTPUT=grail;              MODE=medusa ;;
    2) MODEL=medusa_graphtrip; OUTPUT=grail_escitalopram; MODE=escitalopram ;;
    3) MODEL=medusa_graphtrip; OUTPUT=grail_psilocybin;   MODE=psilocybin ;;
    *) echo "Invalid array index: ${PBS_ARRAY_INDEX}" >&2; exit 1 ;;
esac

python -m scripts.xai_biomarkers -j "${SUBJECT}" -s "${SEED}" -v \
    --weights_base_dir "outputs/${MODEL}/weights/" \
    --mlp_weights_dir "outputs/${MODEL}/weights/" \
    --output_dir "outputs/${MODEL}/${OUTPUT}/" \
    --grail_mode "${MODE}"
