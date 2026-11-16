#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=02:00:00
#PBS -N preprocessing
#PBS -J 1-4

# Parcellates the fMRI and REACT maps into graph files under data/raw/, and builds the
# processed dataset caches under data/processed/ that every later job reads.
#
# Depends on: data/raw/<study>/before/MNI_2mm/ (REACT maps, preprocessing/react/)
# Writes:     data/raw/<study>/before/<atlas>/,
#             data/processed/data_<study>_before_<atlas>.pt
#
# Array index -> dataset:
#   1  psilodep2  schaefer100   (primary dataset)
#   2  psilodep2  schaefer200   (atlas transfer)
#   3  psilodep2  aal           (atlas transfer)
#   4  psilodep1  schaefer100   (validation dataset)
#
# Submit from the repository root: qsub job_scripts/preprocessing.sh

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate graphtrip
set -eo pipefail

cd "${PBS_O_WORKDIR}"
if [ ! -f scripts/graphtrip.py ]; then
    echo "Submit from the graphTRIP repository root." >&2
    exit 1
fi

case "${PBS_ARRAY_INDEX}" in
    1) STUDY=psilodep2; ATLAS=schaefer100 ;;
    2) STUDY=psilodep2; ATLAS=schaefer200 ;;
    3) STUDY=psilodep2; ATLAS=aal ;;
    4) STUDY=psilodep1; ATLAS=schaefer100 ;;
    *) echo "Invalid array index: ${PBS_ARRAY_INDEX}" >&2; exit 1 ;;
esac

python -m preprocessing.preprocess "${STUDY}" before "${ATLAS}"
