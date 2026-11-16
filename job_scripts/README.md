# Job scripts

PBS array jobs that produce every result in the paper. They assume a Conda environment named `graphtrip`; adapt the `#PBS` resource lines and the environment activation at the top of each script to your cluster.

Submit every script from the repository root. Each step skips work whose outputs already exist, so a failed array element can simply be resubmitted.

## Run order

```bash
qsub job_scripts/preprocessing.sh   # 1
qsub job_scripts/train.sh           # 2, after 1
qsub job_scripts/post_training.sh   # 3, after 2
qsub job_scripts/grail.sh           # 3, after 2
qsub job_scripts/posthoc.sh         # 4, after both jobs of step 3
```

The permutation nulls (below) only need step 1 and can run alongside the rest. Figures are generated locally afterwards with `python -m scripts.make_figures` (see the [main README](../README.md)).

## Permutation nulls

Each model is retrained on outcomes permuted across the cohort, 100 permutations x 10 seeds. `launch.sh` submits one model's array and checks that the dataset caches from `preprocessing.sh` exist:

```bash
bash job_scripts/perm_null_jobs/launch.sh graphtrip --perms 0-0 --dry-run   # check first
bash job_scripts/perm_null_jobs/launch.sh graphtrip
bash job_scripts/perm_null_jobs/launch.sh medusa
bash job_scripts/perm_null_jobs/launch.sh graphtrip_bdi
bash job_scripts/perm_null_jobs/launch.sh no_clinical_features
bash job_scripts/perm_null_jobs/launch.sh control_mlp_raw
```

Options: `--perms A-B` restricts to a range of permutations, `--debug` runs a two-epoch smoke test, `--dry-run` prints the `qsub` command without submitting.

## Which job produces which outputs

Output paths are relative to `outputs/`. Figure targets are those listed by `python -m scripts.make_figures --list`.

| Job script (array indices) | Runs | Writes | Read by figure targets |
|---|---|---|---|
| `preprocessing.sh` | `preprocessing/preprocess.py` | `data/raw/{study}/before/{atlas}/`, `data/processed/` | all |
| `train.sh` (0-9) | `scripts/graphtrip.py` | `graphtrip/weights/`, `graphtrip/permutation_importance/`, `graphtrip/transfer_atlas/{schaefer200,aal}/`, `graphtrip/linreg_on_z/`, `graphtrip/retrain_mlp_on_z/`, `graphtrip/predictions_{psilocybin,escitalopram}/` | `fig2`, `fig3`, `fig4`, `aal_atlas`, `ablation_models`, `z_outcome_predictability`, `graphtrip_partial_corrs`, `graphtrip_counterfactuals`, `error_distribution`, `overfitting`, `permutation_null`, summary tables |
| `train.sh` (10-19) | `scripts/medusa.py` | `medusa_graphtrip/weights/`, `medusa_graphtrip/permutation_importance/`, `medusa_graphtrip/estimate_propensity{,_wo_QIDS}/` | `fig5`, `medusa_reconstructions_and_propensity_estimation`, `input_domain_ablations`, `error_distribution`, `permutation_null`, summary tables |
| `train.sh` (20-29) | `scripts/graphtrip_bdi.py` | `graphtrip_bdi/weights/`, `graphtrip_bdi/permutation_importance/` | `graphtrip_bdi`, `permutation_null`, summary tables |
| `train.sh` (30-139) | `scripts/ablation.py -j 0-10` | `ablation/{pca_benchmark,tsne_benchmark,vgae_linreg_head}/`, `ablation/feature_ablation/*/`, `medusa_ablation/*/` | `fig2`, `ablation_models`, `input_domain_ablations`, `permutation_null`, summary tables |
| `train.sh` (140-149) | `scripts/flatvae_mlp.py` | `flatvae_mlp/` | `overfitting`, `aggregated_prediction_metrics` |
| `train.sh` (150-169) | `scripts/train_selser.py` | `selser/{selser,selser_augmented}/` | `aggregated_prediction_metrics` |
| `post_training.sh` (0-9) | `scripts/validation.py` | `validation/evaluate_graphtrip/` | `fig4`, `permutation_null`, `model_performance_summary` |
| `post_training.sh` (10-29) | `scripts/xai_regions.py` | `{graphtrip,medusa_graphtrip}/regional_attributions/` | `fig6` (after `posthoc.sh`) |
| `grail.sh` (0-1679) | `scripts/xai_biomarkers.py` | `graphtrip/grail/`, `medusa_graphtrip/grail{,_escitalopram,_psilocybin}/` | `fig6` (after `posthoc.sh`), `GRAIL_biomarkers` |
| `posthoc.sh` | `scripts/posthoc.py` | `*/grail*/posthoc_analysis/`, `*/regional_attributions/posthoc_analysis/`, `graphtrip/test_biomarkers/`, `biomarker_categories/` | `fig6`, `GRAIL_biomarkers` |
| `perm_null_jobs/launch.sh graphtrip` | `scripts/permutation_null.py` | `graphtrip/permutation_null/perm_*/`, including zero-shot `transfer_atlas/`, `psilodep1/` and `grail/` evaluations of the null weights | `permutation_null`, `fig4` (p-value), `GRAIL_biomarkers` |
| `perm_null_jobs/launch.sh medusa` | `scripts/permutation_null.py` | `medusa_graphtrip/permutation_null/perm_*/`, including `grail/` | `permutation_null`, `GRAIL_biomarkers` |
| `perm_null_jobs/launch.sh graphtrip_bdi`, `no_clinical_features`, `control_mlp_raw` | `scripts/permutation_null.py` | `{model output folder}/permutation_null/perm_*/` | `permutation_null` |

The summary tables are the `model_performance_summary` and `aggregated_prediction_metrics` targets. `dataset_stats` and `norm_target_maps` need only the data in `data/raw/`.
