"""
Supplementary: normative molecular target distributions, and the table of additional
performance metrics.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.annotations import load_receptor_maps
from utils.helpers import summarise_seed_metrics
from utils.plotting import COOLWARM

from figure_making.common import report_prediction_metrics
from figure_making.paths import output_dir
from figure_making.registry import register


# (label, results directory parts, prediction filename)
PERFORMANCE_MODELS = [
    ('graphTRIP', ('graphtrip', 'weights'), 'prediction_results.csv'),
    ('medusa_graphtrip', ('medusa_graphtrip', 'weights'), 'prediction_results.csv'),
    # Zero-shot on psilodep1, under both input mappings. This replaces the fine-tuned
    # psilodep1 model, which the pipeline no longer trains.
    ('psilodep1_zeroshot', ('validation', 'evaluate_graphtrip'),
     'initial_prediction_results_mean_vote.csv'),
    ('psilodep1_zeroshot_harmonised', ('validation', 'evaluate_graphtrip'),
     'initial_prediction_results_mean_vote_harmonised.csv'),
    ('schaefer200', ('graphtrip', 'transfer_atlas', 'schaefer200'),
     'initial_prediction_results.csv'),
    ('aal', ('graphtrip', 'transfer_atlas', 'aal'), 'initial_prediction_results.csv'),
    ('control_mlp', ('ablation', 'feature_ablation', 'control_mlp_raw'), 'prediction_results.csv'),
    ('vgae_linreg_head', ('ablation', 'vgae_linreg_head'), 'prediction_results.csv'),
    ('linreg_on_clinical_data', ('ablation', 'feature_ablation', 'linreg_on_clinical_data'),
     'prediction_results.csv'),
    ('pca_benchmark', ('ablation', 'pca_benchmark'), 'prediction_results.csv'),
    ('tsne_benchmark', ('ablation', 'tsne_benchmark'), 'prediction_results.csv'),
    ('no_node_features', ('ablation', 'feature_ablation', 'no_node_features'),
     'prediction_results.csv'),
    ('no_clinical_features', ('ablation', 'feature_ablation', 'no_clinical_features'),
     'prediction_results.csv'),
    ('no_react_no_clinical', ('ablation', 'feature_ablation', 'no_react_no_clinical'),
     'prediction_results.csv'),
    # Frozen graphTRIP VGAE with only the prediction head refit on [z, Condition].
    ('linreg_on_z', ('graphtrip', 'linreg_on_z'), 'prediction_results.csv'),
    ('retrain_mlp_on_z', ('graphtrip', 'retrain_mlp_on_z'), 'prediction_results.csv'),
    ('medusa_no_node_features', ('medusa_ablation', 'no_node_features'),
     'prediction_results.csv'),
    ('medusa_no_clinical_features', ('medusa_ablation', 'no_clinical_features'),
     'prediction_results.csv'),
    ('medusa_no_react_no_clinical', ('medusa_ablation', 'no_react_no_clinical'),
     'prediction_results.csv'),
    ('graphTRIP_bdi', ('graphtrip_bdi', 'weights'), 'prediction_results.csv'),
]

METRIC_COLUMNS = ['r', 'p_value', 'r2', 'mae', 'mse', 'rmse']

# summarise_seed_metrics caches its result in the results directory, so a directory that
# contributes more than one row needs one cache per prediction file.
DEFAULT_PREDICTION_FILE = 'prediction_results.csv'


@register('norm_target_maps', group='supp', subdir='SUPPLEMENTARY/norm_target_maps')
def norm_target_maps(ctx, out):
    '''Correlations between the normative receptor and transporter density maps.'''
    receptors = load_receptor_maps(atlas=ctx.atlas)
    receptor_corr = receptors.corr()

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(receptor_corr,
                cmap=COOLWARM,
                vmin=-1,
                vmax=1,
                square=True,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation'})
    save_path = out.fig('receptor_correlations')
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

    out.table('receptor_correlations', receptor_corr, index=True)


@register('model_performance_summary', group='supp', subdir='SUPPLEMENTARY')
def model_performance_summary(ctx, out):
    '''Summarises performance metrics across seeds for every model, as one table.'''
    rows = []

    for label, parts, prediction_file in PERFORMANCE_MODELS:
        base_dir = output_dir(*parts)
        summary_file = ('seed_metrics_summary.csv'
                        if prediction_file == DEFAULT_PREDICTION_FILE
                        else f'seed_metrics_{os.path.splitext(prediction_file)[0]}.csv')
        try:
            metrics = summarise_seed_metrics(base_dir=base_dir,
                                             prediction_file=prediction_file,
                                             summary_file=summary_file)
        except (ValueError, FileNotFoundError) as e:
            out.log(f'Skipping {label}: {e}')
            continue

        missing = [c for c in METRIC_COLUMNS if c not in metrics.columns]
        if missing:
            out.log(f'Skipping {label}: missing columns {missing}')
            continue

        n_seeds = len(metrics)
        row = {'model': label,
               'percent_significant': (metrics['p_value'] < 0.05).mean() * 100}
        for metric in ['r', 'r2', 'mae', 'mse', 'rmse']:
            row[f'{metric}_mean'] = metrics[metric].mean()
            row[f'{metric}_sem'] = metrics[metric].sem(ddof=0) if n_seeds > 1 else np.nan

        # Keep the column order of the notebook's table
        rows.append({'model': row['model'],
                     'r_mean': row['r_mean'], 'r_sem': row['r_sem'],
                     'percent_significant': row['percent_significant'],
                     'r2_mean': row['r2_mean'], 'r2_sem': row['r2_sem'],
                     'mae_mean': row['mae_mean'], 'mae_sem': row['mae_sem'],
                     'mse_mean': row['mse_mean'], 'mse_sem': row['mse_sem'],
                     'rmse_mean': row['rmse_mean'], 'rmse_sem': row['rmse_sem']})

    summary_df = pd.DataFrame(rows)
    out.table('model_performance_summary', summary_df)
    out.log_df('Model performance summary', summary_df)


# Models trained and tested on psilodep2, whose aggregated and within-arm statistics are
# comparable
AGGREGATED_METRIC_MODELS = [
    ('graphTRIP', ('graphtrip', 'weights')),
    ('medusa_graphtrip', ('medusa_graphtrip', 'weights')),
    ('control_mlp', ('ablation', 'feature_ablation', 'control_mlp_raw')),
    ('linreg_on_clinical_data', ('ablation', 'feature_ablation', 'linreg_on_clinical_data')),
    ('vgae_linreg_head', ('ablation', 'vgae_linreg_head')),
    ('pca_benchmark', ('ablation', 'pca_benchmark')),
    ('tsne_benchmark', ('ablation', 'tsne_benchmark')),
    ('no_node_features', ('ablation', 'feature_ablation', 'no_node_features')),
    ('no_clinical_features', ('ablation', 'feature_ablation', 'no_clinical_features')),
    ('no_react_no_clinical', ('ablation', 'feature_ablation', 'no_react_no_clinical')),
    ('medusa_no_node_features', ('medusa_ablation', 'no_node_features')),
    ('medusa_no_clinical_features', ('medusa_ablation', 'no_clinical_features')),
    ('medusa_no_react_no_clinical', ('medusa_ablation', 'no_react_no_clinical')),
    ('linreg_on_z', ('graphtrip', 'linreg_on_z')),
    ('retrain_mlp_on_z', ('graphtrip', 'retrain_mlp_on_z')),
]


@register('aggregated_prediction_metrics', group='supp', subdir='SUPPLEMENTARY')
def aggregated_prediction_metrics(ctx, out):
    '''
    One lookup table of the accuracy of the mean-across-seed predictions, per model.

    This is the statistic the manuscript quotes, and it is a different number from
    model_performance_summary.csv, which averages the per-seed r values. Averaging ten
    noisy per-seed correlations is not the correlation of the averaged predictions, and
    the two must not be swapped for one another.

    The companion _within_arm table answers whether accuracy is even across the two
    treatment arms.
    '''
    report_prediction_metrics(
        [(label, os.path.join(output_dir(*parts), 'prediction_results.csv'))
         for label, parts in AGGREGATED_METRIC_MODELS],
        out, name='aggregated_prediction_metrics',
        heading='Accuracy of the mean-across-seed predictions')
