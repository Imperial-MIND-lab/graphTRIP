"""
Fig. 3: Generalisation across brain atlases (Schaefer 200).

The graphTRIP VGAEs and test fold indices are reused unchanged; only the data is
re-parcellated.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os

from utils.configs import load_ingredient_configs
from preprocessing.metrics import get_rsn_mapping

from figure_making.common import scatter_from_results, plot_reconstruction_panels
from figure_making.loaders import load_dataset
from figure_making.paths import output_dir, require
from figure_making.registry import register


ATLAS = 'schaefer200'


@register('fig3', group='main', subdir='Fig.3')
def fig3_atlas_transfer(ctx, out):
    results_dir = require(output_dir('graphtrip', 'transfer_atlas', ATLAS))

    # a. Prediction performance after transfer to Schaefer 200 --------------------------
    scatter_from_results(os.path.join(results_dir, 'initial_prediction_results.csv'),
                         out, f'true_vs_pred_{ATLAS}', yerr='prediction_sem')

    # Reconstruction performance on the re-parcellated data ----------------------------
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
