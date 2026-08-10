"""
Fig. 2: Prediction performance of graphTRIP.

Panels:
- a. graphTRIP and linear regression benchmark performance
- b. partial correlations, controlling for treatment and baseline QIDS
- c-d. VGAE reconstruction performance
- e. permutation importance

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os

from utils.helpers import aggregate_importance_scores
from utils.plotting import NEUTRAL, permutation_importance_bar_chart

from figure_making.common import (
    scatter_from_results, partial_correlation_panels, plot_reconstruction_panels)
from figure_making.paths import output_dir, require
from figure_making.registry import register


@register('fig2', group='main', subdir='Fig.2')
def fig2_prediction_performance(ctx, out):
    weights_base_dir = ctx.weights_base_dir
    linreg_dir = output_dir('ablation', 'linreg_on_clinical_data')

    # a. graphTRIP and linear regression benchmark performance --------------------------
    results = scatter_from_results(
        os.path.join(weights_base_dir, 'prediction_results.csv'),
        out, 'graphTRIP_true_vs_pred', yerr='prediction_sem')

    linreg_results = scatter_from_results(
        os.path.join(linreg_dir, 'prediction_results.csv'),
        out, 'linreg_on_clinical_data_true_vs_pred', yerr='prediction_sem')

    # b. Partial correlations -----------------------------------------------------------
    out.log('=== graphTRIP ===')
    summary = partial_correlation_panels(results, out, 'graphtrip_partial_corr')
    out.log_df('graphTRIP partial correlations', summary)

    out.log('=== Linear regression on clinical data ===')
    linreg_summary = partial_correlation_panels(
        linreg_results, out, 'linreg_on_clinical_data_partial_corr')
    out.log_df('Linear regression partial correlations', linreg_summary)

    # c-d. VGAE reconstruction performance ----------------------------------------------
    plot_reconstruction_panels(ctx, out, ctx.core_reconstructions,
                               atlas=ctx.atlas,
                               rsn_mapping=ctx.rsn_mapping,
                               rsn_labels=ctx.rsn_names,
                               conditions=ctx.conditions,
                               brain_subdir='graphTRIP_reconstructions')

    # e. Permutation importance ---------------------------------------------------------
    importance_dir = require(output_dir('graphtrip', 'permutation_importance'))
    scores = aggregate_importance_scores(
        os.path.join(importance_dir, 'importance_scores_aggregated.csv'))
    scores = scores.sort_values(by='mean', ascending=False)
    permutation_importance_bar_chart(scores, yerr_column='se', color=NEUTRAL, alpha=0.8,
                                     save_path=out.fig('importance_scores_aggregated'))
