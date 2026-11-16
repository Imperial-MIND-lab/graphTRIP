"""
Supplementary: the graphTRIP counterfactual estimates.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os

from utils.helpers import aggregate_prediction_results
from utils.plotting import true_vs_pred_scatter

from figure_making.paths import output_dir, require
from figure_making.registry import register


@register('graphtrip_counterfactuals', group='supp',
          subdir='SUPPLEMENTARY/graphtrip_counterfactuals')
def graphtrip_counterfactuals(ctx, out):
    psilo_dir = require(output_dir('graphtrip', 'predictions_psilocybin'))
    escit_dir = require(output_dir('graphtrip', 'predictions_escitalopram'))

    psilo_results = aggregate_prediction_results(
        results_file=os.path.join(psilo_dir, 'initial_prediction_results.csv'))
    escit_results = aggregate_prediction_results(
        results_file=os.path.join(escit_dir, 'initial_prediction_results.csv'))

    combined = psilo_results.rename(columns={'prediction': 'psilocybin_prediction'})
    combined = combined.merge(escit_results[['subject_id', 'prediction']],
                              on='subject_id', how='left')
    combined = combined.rename(columns={'prediction': 'escitalopram_prediction'})
    combined = combined.sort_values(by='subject_id')
    combined['Condition'] = ctx.conditions

    true_vs_pred_scatter(combined,
                         save_path=out.fig('escitalopram_vs_psilocybin_predictions'),
                         ycol='escitalopram_prediction',
                         xcol='psilocybin_prediction')
