#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=01:00:00
#PBS -N post_training
#PBS -J 0-29

# Short analyses of the trained graphTRIP and Medusa-graphTRIP models: the zero-shot
# evaluation on the validation dataset, and the regional attributions.
#
# Depends on: job_scripts/train.sh
#             (outputs/graphtrip/weights/, outputs/medusa_graphtrip/weights/)
# Writes:     outputs/validation/evaluate_graphtrip/,
#             outputs/graphtrip/regional_attributions/,
#             outputs/medusa_graphtrip/regional_attributions/
#
# Array index -> run (seed = index % 10):
#    0-9   scripts/validation.py    graphTRIP zero-shot on psilodep1
#   10-19  scripts/xai_regions.py   graphTRIP regional attributions
#   20-29  scripts/xai_regions.py   Medusa-graphTRIP regional attributions
#
# Submit from the repository root: qsub job_scripts/post_training.sh

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip
set -eo pipefail

cd "${PBS_O_WORKDIR}"
if [ ! -f scripts/graphtrip.py ]; then
    echo "Submit from the graphTRIP repository root." >&2
    exit 1
fi

BLOCK=$((PBS_ARRAY_INDEX / 10))
SEED=$((PBS_ARRAY_INDEX % 10))

case "${BLOCK}" in
    0) python -m scripts.validation -s "${SEED}" ;;
    1) python -m scripts.xai_regions -s "${SEED}" -v \
           --weights_base_dir outputs/graphtrip/weights/ \
           --mlp_weights_dir outputs/graphtrip/weights/ \
           --output_dir outputs/graphtrip/regional_attributions/ ;;
    2) python -m scripts.xai_regions -s "${SEED}" -v \
           --weights_base_dir outputs/medusa_graphtrip/weights/ \
           --mlp_weights_dir outputs/medusa_graphtrip/weights/ \
           --output_dir outputs/medusa_graphtrip/regional_attributions/ \
           --medusa ;;
    *) echo "Invalid array index: ${PBS_ARRAY_INDEX}" >&2; exit 1 ;;
esac
