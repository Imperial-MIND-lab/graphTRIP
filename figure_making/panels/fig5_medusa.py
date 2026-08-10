"""
Fig. 5: Estimating treatment effects with Medusa graphTRIP.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os

from utils.helpers import aggregate_prediction_results
from utils.plotting import plot_ite_violin

from figure_making.common import scatter_from_results
from figure_making.paths import medusa_weights_dir, require
from figure_making.registry import register


@register('fig5', group='main', subdir='Fig.5')
def fig5_medusa(ctx, out):
    weights_dir = require(medusa_weights_dir())

    # Pseudo-ITE labels versus predicted ITEs -------------------------------------------
    scatter_from_results(os.path.join(weights_dir, 'prediction_results.csv'),
                         out, 'medusa_graphtrip_true_vs_pred', yerr='prediction_sem')

    # Distribution of individual treatment effects ---------------------------------------
    ite_results = aggregate_prediction_results(
        results_file=os.path.join(weights_dir, 'counterfactual_predictions.csv'))
    ite_results['ITE'] = ite_results['prediction_mlp1'] - ite_results['prediction_mlp0']

    plot_ite_violin(ite_results, save_path=out.fig('ite_violin'), ycol='ITE')

    out.log(f"ITE: mean = {ite_results['ITE'].mean():.4f}, "
            f"sem = {ite_results['ITE'].sem():.4f}, "
            f"n negative (favouring psilocybin) = {(ite_results['ITE'] < 0).sum()} "
            f"of {len(ite_results)}")
