"""
Supplementary: how much of the treatment outcome is encoded in the frozen latent space.

Both panels take the graphTRIP VGAE exactly as it was trained, freeze it (encoder and
pooling readout), and refit only the prediction head on [z, Condition] -- no baseline
clinical scores:
- a. ridge regression head (experiments/train_linreg_on_z.py)
- b. newly initialised MLP head, graphTRIP's own architecture (experiments/retrain_mlp.py)

Both reuse the pretrained models' test fold assignments, so each subject is predicted by
the fold model that never saw them. Predictions are the mean across the ten seeds, with
the across-seed standard error as the y error bar.

Author: Hanna M. Tolle
Date: 2026-08-17
License: BSD 3-Clause
"""

import os

from figure_making.common import scatter_from_results, collect_seed_metrics
from figure_making.paths import output_dir
from figure_making.registry import register


# (results directory parts, panel name, title)
Z_HEAD_SCATTERS = [
    (('graphtrip', 'linreg_on_z'), 'linreg_on_z_true_vs_pred',
     'Ridge Regression on Frozen z'),
    (('graphtrip', 'retrain_mlp_on_z'), 'mlp_on_z_true_vs_pred',
     'MLP on Frozen z'),
]


@register('z_outcome_predictability', group='supp',
          subdir='SUPPLEMENTARY/z_outcome_predictability')
def z_outcome_predictability(ctx, out):
    for parts, name, title in Z_HEAD_SCATTERS:
        results_dir = output_dir(*parts)

        # The scatter annotates r, p and MAE of the mean predictions in its own title;
        # this repeats them in stats.txt alongside the across-seed spread.
        scatter_from_results(os.path.join(results_dir, 'prediction_results.csv'),
                             out, name, condition_study='psilodep2',
                             yerr='prediction_sem', title=title)

        metrics = collect_seed_metrics([(name, results_dir, 'final_metrics.csv')])
        out.log(f'=== {title} ===')
        out.log(f'per-seed r:   {metrics["r"].mean():.4f} +/- {metrics["r"].std():.4f} '
                f'(n={len(metrics)} seeds)')
        out.log(f'per-seed MAE: {metrics["mae"].mean():.4f} +/- {metrics["mae"].std():.4f}')
        out.log()
