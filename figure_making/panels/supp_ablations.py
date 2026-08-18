"""
Supplementary: true-versus-predicted scatters for every ablation model.

Model ablations replace the VGAE (PCA, t-SNE), the MLP (linear prediction head, post-hoc
ridge regression on z, retrained MLP head on z). Feature ablations retrain graphTRIP
without clinical or without REACT node features, or without either.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os

from figure_making.common import scatter_from_results
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

FEATURE_ABLATION_SCATTERS = [
    (('ablation', 'feature_ablation', 'control_mlp_raw'), 'control_mlp_true_vs_predicted',
     'MLP, Trained on Clinical Data', None),
    (('ablation', 'feature_ablation', 'linreg_on_clinical_data'),
     'linreg_on_clinical_data_true_vs_pred', 'OLS Regression, Trained on Clinical Data', None),
    (('ablation', 'feature_ablation', 'no_clinical_features'),
     'no_clinical_features_true_vs_pred', 'graphTRIP, Trained without Clinical Features', None),
    (('ablation', 'feature_ablation', 'no_node_features'),
     'no_node_features_true_vs_pred', 'graphTRIP, Trained without REACT Node Features', None),
    (('ablation', 'feature_ablation', 'no_react_no_clinical'),
     'no_react_no_clinical_true_vs_pred',
     'graphTRIP, Trained without REACT Node or Clinical Features', None),
]


def _scatters(specs, out):
    for parts, name, title, study in specs:
        scatter_from_results(os.path.join(output_dir(*parts), 'prediction_results.csv'),
                             out, name, condition_study=study,
                             yerr='prediction_sem', title=title)


@register('ablation_models', group='supp', subdir='SUPPLEMENTARY/ablation_models')
def ablation_models(ctx, out):
    '''Scatters for the ablations of the VGAE and of the prediction head.'''
    _scatters(MODEL_ABLATION_SCATTERS, out)


@register('feature_ablation', group='supp', subdir='SUPPLEMENTARY/feature_ablation')
def feature_ablation(ctx, out):
    '''Scatters for the ablations of the input domains.'''
    _scatters(FEATURE_ABLATION_SCATTERS, out)
