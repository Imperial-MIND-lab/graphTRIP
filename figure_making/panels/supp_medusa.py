"""
Supplementary: Medusa graphTRIP reconstruction performance and propensity estimation.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import os

from utils.configs import load_ingredient_configs, get_weight_filenames_from_config
from utils.helpers import load_test_fold_indices

from figure_making.common import (
    plot_correlation_boxplot, propensity_panels, importance_panel)
from figure_making.loaders import load_vgaes
from figure_making.paths import medusa_weights_dir, output_dir, require
from figure_making.registry import register


@register('medusa_reconstructions_and_propensity_estimation', group='supp',
          subdir='SUPPLEMENTARY/medusa_reconstructions_and_propensity_estimation')
def medusa_reconstructions_and_propensity(ctx, out):
    weights_dir = require(medusa_weights_dir())

    # Reconstruction performance of the Medusa VGAEs -------------------------------------
    config = load_ingredient_configs(os.path.join(weights_dir, 'seed_0'),
                                     ingredients=['vgae_model', 'dataset'])
    weight_filenames = get_weight_filenames_from_config(config)
    test_indices_dict, weights_dirs = load_test_fold_indices(weights_dir,
                                                             subdir_name_pattern='seed_*')

    vgaes_dict = {f'seed_{seed}': load_vgaes(config['vgae_model'], seed_dir,
                                             weight_filenames['vgae'])
                  for seed, seed_dir in enumerate(weights_dirs)}

    _, x_orig_rcn = ctx.reconstructions(vgaes_dict, ctx.data, test_indices_dict)
    plot_correlation_boxplot(out, x_orig_rcn, ctx.conditions,
                             'medusa_original_vs_reconstructed_corrs')

    # Propensity estimation, with and without baseline QIDS ------------------------------
    propensity_panels(output_dir('medusa_graphtrip', 'estimate_propensity'), out)
    propensity_panels(output_dir('medusa_graphtrip', 'estimate_propensity_wo_QIDS'), out,
                      suffix='_wo_QIDS')

    # Permutation importance ---------------------------------------------------------------
    importance_panel(require(output_dir('medusa_graphtrip', 'permutation_importance')),
                     weights_dir, out)
