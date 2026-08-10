"""
Fig. 4: Generalisation to an independent dataset (psilodep1).

Panels:
- b. reconstruction performance of the graphTRIP VGAE, not fine-tuned
- c. prediction performance after fine-tuning
- d. the effect of pretraining

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import numpy as np

from utils.configs import load_ingredient_configs
from utils.helpers import aggregate_prediction_results
from utils.plotting import true_vs_pred_scatter
from utils.statsalg import compare_reconstruction_performance

from figure_making.common import plot_correlation_boxplot, model_comparison_panels
from figure_making.loaders import load_dataset
from figure_making.paths import output_dir, require
from figure_making.registry import register


@register('fig4', group='main', subdir='Fig.4')
def fig4_validation(ctx, out):
    results_base_dir = require(output_dir('validation', 'finetuning'))

    # b. Reconstruction performance on the validation dataset ---------------------------
    psilodep1_config = load_ingredient_configs(os.path.join(results_base_dir, 'seed_0'),
                                               ingredients=['dataset'])
    psilodep1_data = load_dataset(psilodep1_config['dataset'])
    psilodep1_num_subs = len(psilodep1_data)

    # All psilodep1 patients were treated with psilocybin
    psilodep1_conditions = np.ones(psilodep1_num_subs)

    # Every VGAE of every seed and fold reconstructs every patient, then averages
    _, psilodep1_x = ctx.reconstructions(ctx.vgaes_dict, psilodep1_data, None)

    # Compare reconstruction performance on the primary versus validation dataset
    _, primary_x = ctx.core_reconstructions
    test_results = compare_reconstruction_performance(primary_x['metrics'],
                                                      psilodep1_x['metrics'])
    out.table('primary_vs_validation_corr', test_results['corr'])
    out.table('primary_vs_validation_mae', test_results['mae'])
    out.log_df('Primary vs validation reconstruction (correlation)', test_results['corr'])
    out.log_df('Primary vs validation reconstruction (MAE)', test_results['mae'])

    plot_correlation_boxplot(out, psilodep1_x, psilodep1_conditions,
                             'original_vs_reconstructed_corrs')

    # c. Prediction performance after fine-tuning ---------------------------------------
    results = aggregate_prediction_results(
        results_file=os.path.join(results_base_dir, 'prediction_results_mean_vote.csv'))
    results['Condition'] = 1  # add psilocybin condition for plotting
    true_vs_pred_scatter(results, save_path=out.fig('finetuned_true_vs_pred'),
                         yerr='prediction_sem')

    # d. The effect of pretraining ------------------------------------------------------
    base_dir = output_dir('validation')
    specs = [
        ('control_mlp', os.path.join(base_dir, 'control_mlp'), 'final_metrics.csv'),
        ('linreg_on_clinical_data', os.path.join(base_dir, 'linreg_on_clinical_data'),
         'final_metrics.csv'),
        ('psilodep1_graphtrip', os.path.join(base_dir, 'psilodep1_graphtrip'),
         'final_metrics.csv'),
        ('finetuning', os.path.join(base_dir, 'finetuning'), 'final_metrics_mean_vote.csv'),
    ]
    model_comparison_panels(specs, out, 'raincloud_retrained_vs_pretrained',
                            num_subs=psilodep1_num_subs,
                            model_of_interest='finetuning',
                            sort_by_mean=False,
                            offset=2,
                            figsize=(8, 3.5),
                            skip_missing=True)
