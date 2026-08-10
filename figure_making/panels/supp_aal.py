"""
Supplementary: generalisation to the AAL brain atlas.

Also compares reconstruction quality in cortical versus subcortical regions, which is
where the AAL atlas differs most from the Schaefer parcellations.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import numpy as np
import pandas as pd

from utils.configs import load_ingredient_configs
from utils.plotting import plot_metric_boxplot
from preprocessing.metrics import get_rsn_mapping, get_atlas

from figure_making.common import scatter_from_results, plot_reconstruction_panels
from figure_making.loaders import load_dataset
from figure_making.paths import output_dir, require
from figure_making.registry import register


ATLAS = 'aal'

SUBCORTICAL_KEYWORDS = [
    'Caudate', 'Putamen', 'Pallidum', 'Thalamus', 'Amygdala',
    'Hippocampus', 'ParaHippocampal',
    'Cerebelum', 'Vermis',
]


def _split_cortical_subcortical(atlas_name):
    '''Returns the indices of cortical and subcortical regions of the AAL atlas.'''
    labels = get_atlas(atlas_name)['labels']
    cortical, subcortical = [], []
    for idx, label in enumerate(labels):
        if any(keyword in label for keyword in SUBCORTICAL_KEYWORDS):
            subcortical.append(idx)
        else:
            cortical.append(idx)
    return np.array(cortical), np.array(subcortical)


def _regional_reconstruction_corrs(adj_orig_rcn, x_orig_rcn, node_attrs,
                                   cortical_idx, subcortical_idx, num_subs):
    '''
    Correlates original with reconstructed values separately for cortical and
    subcortical regions, per node feature and for FC.
    '''
    columns = {}
    for feat_idx, feat_name in enumerate(node_attrs):
        for prefix, indices in [('cortical', cortical_idx), ('subcortical', subcortical_idx)]:
            corrs = []
            for subj in range(num_subs):
                orig = x_orig_rcn['original'][indices, feat_idx, subj]
                rcn = x_orig_rcn['reconstructed'][indices, feat_idx, subj]
                corrs.append(np.corrcoef(orig, rcn)[0, 1])
            columns[f'{prefix}_{feat_name}'] = corrs

    for prefix, indices in [('cortical', cortical_idx), ('subcortical', subcortical_idx)]:
        corrs = []
        for subj in range(num_subs):
            orig = adj_orig_rcn['original'][indices, :, subj][:, indices].flatten()
            rcn = adj_orig_rcn['reconstructed'][indices, :, subj][:, indices].flatten()
            corrs.append(np.corrcoef(orig, rcn)[0, 1])
        columns[f'fc_{prefix}'] = corrs

    return pd.DataFrame(columns)


@register('aal_atlas', group='supp', subdir='SUPPLEMENTARY/aal_atlas')
def aal_atlas(ctx, out):
    results_dir = require(output_dir('graphtrip', 'transfer_atlas', ATLAS))

    # Prediction performance after transfer to AAL --------------------------------------
    scatter_from_results(os.path.join(results_dir, 'initial_prediction_results.csv'),
                         out, f'true_vs_pred_{ATLAS}', yerr='prediction_sem')

    # Reconstruction performance on the re-parcellated data ------------------------------
    atlas_config = load_ingredient_configs(os.path.join(results_dir, 'seed_0'),
                                           ingredients=['dataset'])
    atlas_data = load_dataset(atlas_config['dataset'])
    atlas_rsn_mapping, atlas_rsn_labels = get_rsn_mapping(ATLAS)

    recon = ctx.reconstructions(ctx.vgaes_dict, atlas_data, ctx.test_indices_dict)
    plot_reconstruction_panels(ctx, out, recon,
                               atlas=atlas_data.atlas,
                               rsn_mapping=atlas_rsn_mapping,
                               rsn_labels=atlas_rsn_labels,
                               conditions=ctx.conditions,
                               brain_subdir=f'graphTRIP_reconstructions_{ATLAS}',
                               suffix=f'_{ATLAS}')

    # Cortical versus subcortical reconstruction quality ---------------------------------
    adj_orig_rcn, x_orig_rcn = recon
    cortical_idx, subcortical_idx = _split_cortical_subcortical(ATLAS)
    out.log(f'Number of cortical regions: {len(cortical_idx)}')
    out.log(f'Number of subcortical regions: {len(subcortical_idx)}')

    results_df = _regional_reconstruction_corrs(adj_orig_rcn, x_orig_rcn, ctx.node_attrs,
                                                cortical_idx, subcortical_idx, ctx.num_subs)
    plot_metric_boxplot(results_df,
                        conditions=ctx.conditions,
                        short_names=False,
                        ylabel='correlation',
                        save_path=out.fig('AAL_cortical_vs_subcortical_reconstructions'))
    out.log_df('Cortical vs subcortical reconstruction correlations (mean)',
               results_df.mean().to_frame('mean_r'), index=True)
