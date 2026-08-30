"""
Supplementary: true-versus-predicted scatters for every ablation model.

Model ablations replace the VGAE (PCA, t-SNE), the MLP (linear prediction head, post-hoc
ridge regression on z, retrained MLP head on z). The input-domain ablations live in the
input_domain_ablations target, alongside Medusa-graphTRIP's.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os

from figure_making.common import scatter_from_results, report_prediction_metrics
from figure_making.paths import output_dir
from figure_making.registry import register


# (results directory parts, panel name, title, study for the condition column)
MODEL_ABLATION_SCATTERS = [
    (('ablation', 'pca_benchmark'), 'pca_true_vs_predicted', 'PCA', None),
    (('ablation', 'tsne_benchmark'), 'tsne_true_vs_predicted', 't-SNE', None),
    (('ablation', 'vgae_linreg_head'), 'vgae_with_linreg_head_true_vs_predicted',
     'VGAE, trained with linear prediction head', None),
    (('graphtrip', 'linreg_on_z'), 'linreg_on_z_true_vs_predicted',
     'Frozen graphTRIP-VGAE, with post-hoc ridge regression on z', 'psilodep2'),
    (('graphtrip', 'retrain_mlp_on_z'), 'retrain_mlp_on_z_true_vs_predicted',
     'Frozen graphTRIP-VGAE, with retrained MLP head on z', 'psilodep2'),
]


def _results_file(parts):
    return os.path.join(output_dir(*parts), 'prediction_results.csv')


def _scatters(specs, out):
    for parts, name, title, study in specs:
        scatter_from_results(_results_file(parts), out, name, condition_study=study,
                             yerr='prediction_sem', title=title)


def _model_label(panel_name):
    '''Strips the panel-name suffix, so the metrics table is keyed by model.'''
    for suffix in ('_true_vs_predicted', '_true_vs_pred'):
        if panel_name.endswith(suffix):
            return panel_name[:-len(suffix)]
    return panel_name


def _metrics(specs, out):
    '''Accuracy of each scattered model, so the numbers exist outside the panel titles.'''
    report_prediction_metrics([(_model_label(name), _results_file(parts))
                               for parts, name, _, _ in specs], out)


@register('ablation_models', group='supp', subdir='SUPPLEMENTARY/ablation_models')
def ablation_models(ctx, out):
    '''Scatters for the ablations of the VGAE and of the prediction head.'''
    _scatters(MODEL_ABLATION_SCATTERS, out)
    _metrics(MODEL_ABLATION_SCATTERS, out)
