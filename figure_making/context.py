"""
Lazily-loaded state shared between figure targets.

The notebook loads the dataset, VGAEs and MLPs for every seed up front, even for
panels that only read CSVs. Here everything is a cached_property, so a target that
needs no models never touches the weights.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

import numpy as np
from functools import cached_property

from utils.configs import load_ingredient_configs, get_weight_filenames_from_config
from utils.helpers import load_test_fold_indices, fix_random_seed
from utils.plotting import CMAP_5HT1A, CMAP_5HT2A, CMAP_5HTT, get_rsn_ticks
from preprocessing.metrics import get_rsn_mapping
from experiments.ingredients.data_ingredient import get_conditions
from experiments.ingredients.vgae_ingredient import (
    get_mean_test_reconstructions, evaluate_fc_reconstructions, evaluate_x_reconstructions)

from figure_making.io import FigureOutput
from figure_making.loaders import load_dataset, load_vgaes, load_mlps, get_device
from figure_making.paths import graphtrip_weights_dir, require


class FigureContext:
    """
    Shared state for the graphTRIP core model, plus helpers used by several targets.

    Models for other experiments (medusa, BDI, validation) are loaded by the targets
    that need them, via load_models().
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.device = get_device()

        # The reconstruction panels sample the VGAE latents, so without this the panels
        # and the reconstruction statistics differ slightly between runs.
        fix_random_seed(cfg.seed)

    def output(self, subdir):
        '''Returns a FigureOutput writing into <outdir>/<subdir>.'''
        return FigureOutput(self.cfg, subdir)

    # graphTRIP core model ----------------------------------------------------------------

    @cached_property
    def weights_base_dir(self):
        return require(graphtrip_weights_dir())

    @cached_property
    def config(self):
        return load_ingredient_configs(
            f'{self.weights_base_dir}/seed_0',
            ingredients=['dataset', 'vgae_model', 'mlp_model'])

    @cached_property
    def weight_filenames(self):
        return get_weight_filenames_from_config(self.config)

    @cached_property
    def _test_folds(self):
        return load_test_fold_indices(self.weights_base_dir, subdir_name_pattern='seed_*')

    @cached_property
    def test_indices_dict(self):
        return self._test_folds[0]

    @cached_property
    def weights_dirs(self):
        return self._test_folds[1]

    @cached_property
    def num_seeds(self):
        return len(self.weights_dirs)

    @cached_property
    def data(self):
        return load_dataset(self.config['dataset'])

    @cached_property
    def vgaes_dict(self):
        return {f'seed_{seed}': load_vgaes(self.config['vgae_model'], weights_dir,
                                           self.weight_filenames['vgae'])
                for seed, weights_dir in enumerate(self.weights_dirs)}

    @cached_property
    def mlps_dict(self):
        readout_dim = self.vgaes_dict['seed_0'][0].readout_dim
        return {f'seed_{seed}': load_mlps(self.config['mlp_model'], readout_dim, weights_dir,
                                          self.weight_filenames['mlp'])
                for seed, weights_dir in enumerate(self.weights_dirs)}

    # Derived plotting metadata -----------------------------------------------------------

    @cached_property
    def conditions(self):
        return np.array(get_conditions(self.data, self.config['dataset']['graph_attrs']))

    @cached_property
    def atlas(self):
        return self.data.atlas

    @cached_property
    def _rsn(self):
        return get_rsn_mapping(self.atlas)

    @cached_property
    def rsn_mapping(self):
        return self._rsn[0]

    @cached_property
    def rsn_names(self):
        return self._rsn[1]

    @cached_property
    def rsn_ticks(self):
        return get_rsn_ticks(self.rsn_mapping, self.rsn_names)

    @cached_property
    def num_subs(self):
        return len(self.data)

    @cached_property
    def num_nodes(self):
        return len(self.rsn_mapping)

    @cached_property
    def node_attrs(self):
        '''Short names of the node attributes, e.g. "x5-HT1A" from "x5-HT1A_react".'''
        return [attr.split('_')[0] for attr in self.config['dataset']['node_attrs']]

    @property
    def x_cmaps(self):
        return [CMAP_5HT1A, CMAP_5HT2A, CMAP_5HTT]

    # Reconstructions ---------------------------------------------------------------------

    def reconstructions(self, vgaes_dict, dataset, test_indices_dict=None):
        '''
        Returns (adj_orig_rcn, x_orig_rcn) with reconstruction metrics attached.
        '''
        key = (id(vgaes_dict), id(dataset), id(test_indices_dict))
        if key in self._recon_cache:
            return self._recon_cache[key][3:]

        adj_orig_rcn, x_orig_rcn = get_mean_test_reconstructions(
            vgaes_dict, dataset, test_indices_dict=test_indices_dict)

        fc_metrics = evaluate_fc_reconstructions(adj_orig_rcn)
        x_orig_rcn['metrics'] = evaluate_x_reconstructions(x_orig_rcn)
        x_orig_rcn['metrics']['corr']['FC'] = fc_metrics['corr']
        x_orig_rcn['metrics']['mae']['FC'] = fc_metrics['mae']

        # The inputs are stored alongside the results so that they stay alive: the cache
        # key is their id(), which CPython would otherwise be free to reuse.
        self._recon_cache[key] = (vgaes_dict, dataset, test_indices_dict,
                                  adj_orig_rcn, x_orig_rcn)
        return adj_orig_rcn, x_orig_rcn

    @cached_property
    def _recon_cache(self):
        return {}

    @cached_property
    def core_reconstructions(self):
        '''Reconstructions of the primary dataset by the graphTRIP core model.'''
        return self.reconstructions(self.vgaes_dict, self.data, self.test_indices_dict)
