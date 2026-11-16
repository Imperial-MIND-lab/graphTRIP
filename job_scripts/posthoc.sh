#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=00:30:00
#PBS -N posthoc

# Aggregates the GRAIL and regional-attribution runs across seeds, tests the candidate
# biomarkers' outcome correlations, and assigns the biomarker categories.
#
# Depends on: job_scripts/grail.sh, job_scripts/post_training.sh
# Writes:     outputs/{graphtrip,medusa_graphtrip}/grail*/posthoc_analysis/,
#             outputs/{graphtrip,medusa_graphtrip}/regional_attributions/posthoc_analysis/,
#             outputs/graphtrip/test_biomarkers/, outputs/biomarker_categories/
#
# Submit from the repository root: qsub job_scripts/posthoc.sh

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip
set -eo pipefail

cd "${PBS_O_WORKDIR}"
if [ ! -f scripts/graphtrip.py ]; then
    echo "Submit from the graphTRIP repository root." >&2
    exit 1
fi

python -m scripts.posthoc \
    --core_model_dir outputs/graphtrip/ \
    --medusa_model_dir outputs/medusa_graphtrip/
