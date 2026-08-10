"""
Supplementary: graphTRIP transferred to psilodep1 without fine-tuning, and the
graphTRIP counterfactual estimates.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os

from utils.helpers import aggregate_prediction_results
from utils.plotting import true_vs_pred_scatter

from figure_making.common import (
    scatter_from_results, collect_seed_metrics, metrics_to_distributions,
    raincloud_of_model_r)
from figure_making.paths import output_dir, require
from figure_making.registry import register


PSILODEP1_NUM_SUBS = 16


@register('evaluate_graphtrip_on_psilodep1', group='supp',
          subdir='SUPPLEMENTARY/evaluate_graphtrip_on_psilodep1')
def evaluate_graphtrip_on_psilodep1(ctx, out):
    results_dir = require(output_dir('validation', 'evaluate_graphtrip'))

    # Prediction performance without fine-tuning ----------------------------------------
    scatter_from_results(os.path.join(results_dir, 'initial_prediction_results_mean_vote.csv'),
                         out, 'initial_mean_vote_true_vs_pred', yerr='prediction_sem')

    # Seed sensitivity -------------------------------------------------------------------
    specs = [('evaluate_graphtrip', results_dir, 'initial_metrics_mean_vote.csv')]
    metrics_df = collect_seed_metrics(specs)
    distributions = metrics_to_distributions(metrics_df, sort_by_mean=False)
    raincloud_of_model_r(distributions, out, 'raincloud_evaluate_graphtrip_on_psilodep1',
                         num_subs=PSILODEP1_NUM_SUBS, offset=2, figsize=(6, 2))

    # MLP pretraining performance ---------------------------------------------------------
    pretraining_dir = require(output_dir('validation', 'pretraining'))
    scatter_from_results(os.path.join(pretraining_dir, 'prediction_results.csv'),
                         out, 'MLP_pretraining_true_vs_pred', yerr='prediction_sem')


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
