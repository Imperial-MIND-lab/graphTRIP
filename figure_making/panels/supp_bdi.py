"""
Supplementary: predicting post-treatment BDI instead of QIDS.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os
import numpy as np

from utils.configs import load_ingredient_configs, get_weight_filenames_from_config
from utils.helpers import load_test_fold_indices
from experiments.ingredients.data_ingredient import get_conditions

from figure_making.common import (
    scatter_from_results, plot_correlation_boxplot, importance_panel)
from figure_making.loaders import load_dataset, load_vgaes
from figure_making.paths import output_dir, require
from figure_making.registry import register


@register('graphtrip_bdi', group='supp', subdir='SUPPLEMENTARY/graphtrip_bdi')
def graphtrip_bdi(ctx, out):
    weights_base_dir = require(output_dir('graphtrip_bdi', 'weights'))

    # Prediction performance -------------------------------------------------------------
    scatter_from_results(os.path.join(weights_base_dir, 'prediction_results.csv'),
                         out, 'true_vs_pred', yerr='prediction_sem')

    # Load the BDI dataset and its VGAEs -------------------------------------------------
    test_indices_dict, weights_dirs = load_test_fold_indices(weights_base_dir,
                                                             subdir_name_pattern='seed_*')
    config = load_ingredient_configs(weights_dirs[0], ingredients=['dataset', 'vgae_model'])
    weight_filenames = get_weight_filenames_from_config(config)

    data = load_dataset(config['dataset'])
    conditions = np.array(get_conditions(data, config['dataset']['graph_attrs']))

    vgaes_dict = {f'seed_{seed}': load_vgaes(config['vgae_model'], weights_dir,
                                             weight_filenames['vgae'])
                  for seed, weights_dir in enumerate(weights_dirs)}

    # Reconstruction performance ---------------------------------------------------------
    _, x_orig_rcn = ctx.reconstructions(vgaes_dict, data, test_indices_dict)
    plot_correlation_boxplot(out, x_orig_rcn, conditions, 'original_vs_recon_corrs')

    # Permutation importance -------------------------------------------------------------
    importance_panel(require(output_dir('graphtrip_bdi', 'permutation_importance')),
                     weights_base_dir, out)
