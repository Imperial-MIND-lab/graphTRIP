#!/bin/bash
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=08:00:00
#PBS -N train
#PBS -J 0-169

# Trains every model on the primary dataset, together with the analyses that only need a
# model's own weights (permutation importance, atlas transfer, counterfactual predictions,
# propensity estimation, frozen-latent heads). Every step skips work that already exists.
#
# Depends on: job_scripts/preprocessing.sh
# Writes:     outputs/graphtrip/, outputs/medusa_graphtrip/, outputs/graphtrip_bdi/,
#             outputs/ablation/, outputs/medusa_ablation/, outputs/flatvae_mlp/,
#             outputs/selser/
#
# Array index -> run. Each block of ten indices is seeds 0-9 of one model:
#     0-9    scripts/graphtrip.py         graphTRIP
#    10-19   scripts/medusa.py            Medusa-graphTRIP
#    20-29   scripts/graphtrip_bdi.py     graphTRIP predicting BDI
#    30-139  scripts/ablation.py -j 0-10  model and input-domain ablations
#   140-149  scripts/flatvae_mlp.py       FlatVAE + MLP baseline (non-graph)
#   150-159  scripts/train_selser.py      SELSER-fMRI (selser.json)
#   160-169  scripts/train_selser.py      SELSER-fMRI, augmented (selser_augmented.json)
#
# Submit from the repository root: qsub job_scripts/train.sh

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
    0)  python -m scripts.graphtrip -s "${SEED}" ;;
    1)  python -m scripts.medusa -s "${SEED}" ;;
    2)  python -m scripts.graphtrip_bdi -s "${SEED}" ;;
    3|4|5|6|7|8|9|10|11|12|13)
        python -m scripts.ablation -j $((BLOCK - 3)) -s "${SEED}" ;;
    14) python -m scripts.flatvae_mlp -s "${SEED}" ;;
    15) python -m scripts.train_selser --config experiments/configs/selser.json \
            --seed "${SEED}" --output_dir "outputs/selser/selser/seed_${SEED}" ;;
    16) python -m scripts.train_selser --config experiments/configs/selser_augmented.json \
            --seed "${SEED}" --output_dir "outputs/selser/selser_augmented/seed_${SEED}" ;;
    *)  echo "Invalid array index: ${PBS_ARRAY_INDEX}" >&2; exit 1 ;;
esac
