"""
This script trains one of the permutation-null models on permuted outcomes, and runs the
evaluations of its weights, to build empirical null distributions.

The outcome is permuted across the whole cohort before any pipeline step -- before
splitting, scaling and VGAE fitting -- so the null covers the entire pipeline rather than
the final correlation alone. The ten training seeds of one perm_seed share that
permutation and are ensembled into a single null draw.

Dependencies:
- experiments/configs/graphtrip.json
- experiments/configs/medusa_graphtrip.json
- experiments/configs/psilodep1_finetuning.json
- scripts/ablation.py (control_mlp_config), scripts/validation.py (zeroshot_config)

Outputs:
- outputs/<model tree>/permutation_null/perm_{perm_seed}/seed_{seed}/

graphTRIP and Medusa additionally run GRAIL on their null weights, and graphTRIP evaluates
them zero-shot:
- .../perm_{perm_seed}/transfer_atlas/{schaefer200,aal}/seed_{seed}/   (graphTRIP)
- .../perm_{perm_seed}/psilodep1/seed_{seed}/                          (graphTRIP)
- .../perm_{perm_seed}/grail/seed_{seed}/mean_alignments.csv           (both)

Medusa's GRAIL output carries a grail_mode column holding its three CFRHead outputs
(escitalopram, psilocybin, ite), so its table has three rows per (subject, fold).

Author: Hanna M. Tolle
Date: 2026-08-26
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
sys.path.append("../")

import os
import copy
import glob
import argparse
from utils.files import add_project_root
from utils.configs import load_configs_from_json, fetch_job_config
from experiments.run_experiment import run
from scripts.ablation import control_mlp_config
from scripts.validation import zeroshot_config


GRAPHTRIP_CONFIG_FILE = 'experiments/configs/graphtrip.json'
MEDUSA_CONFIG_FILE = 'experiments/configs/medusa_graphtrip.json'
PSILODEP1_CONFIG_FILE = 'experiments/configs/psilodep1_finetuning.json'

# The zero-shot transfer experiment runs its own exact permutation test
N_PERMUTATIONS = 100

ATLAS_NUM_NODES = {'schaefer200': 200, 'aal': 116}

# GRAIL settings, matching the published runs (see scripts/xai_biomarkers.py)
GRAIL_NUM_Z_SAMPLES = 25
GRAIL_SIGMA = 2.0
GRAIL_FEATURES = None  # all 69 candidates; a list would restrict to those biomarkers


def bdi_config(config):
    '''Config for the graphTRIP model predicting post-treatment BDI instead of QIDS.'''
    config_updates = copy.deepcopy(config)
    config_updates['dataset']['target'] = 'BDI_Final_Integration'
    return config_updates


def no_clinical_config(config):
    '''Config for the FC + REACT ablation: brain graphs and drug condition, no covariates.'''
    config_updates = copy.deepcopy(config)
    config_updates['dataset']['graph_attrs'] = ['Condition']
    return config_updates


# Model settings. 'apply' rewrites the base config the way the model's own driver script
# does, so the null model is trained by exactly the same recipe as its true-label
# counterpart; 'evaluations' lists the zero-shot analyses to run on the null weights;
# 'grail' overrides the GRAIL defaults for models that need them.
MODELS = {
    'graphtrip': {'config_file': GRAPHTRIP_CONFIG_FILE,
                  'exname': 'train_jointly',
                  'output_dir': 'outputs/graphtrip/',
                  'apply': None,
                  'evaluations': ['transfer_atlas', 'psilodep1', 'grail'],
                  'grail': {}},

    # Medusa's three GRAIL modes share their latent samples, reconstructions and feature
    # gradients, so one pass over the features serves all three
    'medusa': {'config_file': MEDUSA_CONFIG_FILE,
               'exname': 'train_cfrnet',
               'output_dir': 'outputs/medusa_graphtrip/',
               'apply': None,
               'evaluations': ['grail'],
               'grail': {'medusa_modes': ['escitalopram', 'psilocybin', 'ite']}},

    'graphtrip_bdi': {'config_file': GRAPHTRIP_CONFIG_FILE,
                      'exname': 'train_jointly',
                      'output_dir': 'outputs/graphtrip_bdi/',
                      'apply': bdi_config,
                      'evaluations': []},

    'no_clinical_features': {'config_file': GRAPHTRIP_CONFIG_FILE,
                             'exname': 'train_jointly',
                             'output_dir': 'outputs/ablation/feature_ablation/no_clinical_features/',
                             'apply': no_clinical_config,
                             'evaluations': []},

    'control_mlp_raw': {'config_file': GRAPHTRIP_CONFIG_FILE,
                        'exname': 'train_mlp',
                        'output_dir': 'outputs/ablation/feature_ablation/control_mlp_raw/',
                        'apply': control_mlp_config,
                        'evaluations': []},
}


def weights_exist(weights_dir):
    '''Whether a null run directory still holds the fold weights its evaluations need.'''
    return bool(glob.glob(os.path.join(weights_dir, 'k*_vgae_weights.pth')))


def train_null_model(exname, config, ex_dir, perm_seed, observer):
    '''Trains one model on the outcomes permuted by perm_seed.'''
    if os.path.exists(ex_dir):
        print(f"Permutation null experiment already exists in {ex_dir}.")
        return

    config_updates = copy.deepcopy(config)
    config_updates['output_dir'] = ex_dir
    config_updates['save_weights'] = True
    config_updates['dataset']['perm_seed'] = perm_seed
    run(exname, observer, config_updates)


def run_transfer_atlas(config, weights_dir, perm_dir, seed, verbose, observer):
    '''
    Applies the null weights to Schaefer 200 and AAL graphs without retraining.

    Mirrors the atlas transfer in scripts/graphtrip.py: only the parcellation changes, so
    the number of nodes is overridden and num_epochs is set to zero.
    '''
    ingredient_config = {'dataset': copy.deepcopy(config['dataset']),
                         'vgae_model': copy.deepcopy(config['vgae_model']),
                         'mlp_model': copy.deepcopy(config['mlp_model']),
                         'seed': seed,
                         'verbose': verbose}

    for atlas, num_nodes in ATLAS_NUM_NODES.items():
        ex_dir = os.path.join(perm_dir, 'transfer_atlas', atlas, f'seed_{seed}')
        if os.path.exists(ex_dir):
            print(f"Transfer to {atlas} already exists in {ex_dir}.")
            continue

        config_updates = copy.deepcopy(ingredient_config)
        config_updates['dataset']['atlas'] = atlas
        config_updates['dataset']['num_nodes'] = num_nodes
        config_updates['vgae_model']['params']['num_nodes'] = num_nodes
        config_updates['num_epochs'] = 0  # no finetuning
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['save_weights'] = False
        run('test_and_finetune', observer, config_updates)


def run_psilodep1(weights_dir, perm_dir, seed, verbose, debug, observer, config_id=0):
    '''
    Applies the null weights to the validation dataset without retraining.

    Reuses scripts/validation.py's zeroshot_config, so the null evaluation is built the
    same way as the true-label one, including baseline-score harmonisation.
    '''
    ex_dir = os.path.join(perm_dir, 'psilodep1', f'seed_{seed}')
    if os.path.exists(ex_dir):
        print(f"psilodep1 zero-shot evaluation already exists in {ex_dir}.")
        return

    psilodep1_config = load_configs_from_json(add_project_root(PSILODEP1_CONFIG_FILE))
    psilodep1_config = fetch_job_config(psilodep1_config, config_id)
    source_config = load_configs_from_json(os.path.join(weights_dir, 'config.json'))

    exname = 'transfer_and_finetune'
    n_permutations = 10 if debug else N_PERMUTATIONS
    config_updates = zeroshot_config(psilodep1_config, source_config, weights_dir, ex_dir,
                                     seed, verbose, n_permutations, exname)
    run(exname, observer, config_updates)


def run_grail(model, config, weights_dir, perm_dir, seed, perm_seed, verbose, debug,
              observer):
    '''
    Runs GRAIL on the null weights, without the spin permutation test.
    '''
    ex_dir = os.path.join(perm_dir, 'grail', f'seed_{seed}')
    if os.path.exists(ex_dir):
        print(f"GRAIL already exists in {ex_dir}.")
        return

    config_updates = {'dataset': copy.deepcopy(config['dataset']),
                      'vgae_weights_dir': weights_dir,
                      'mlp_weights_dir': weights_dir,
                      'output_dir': ex_dir,
                      'seed': seed,
                      'verbose': verbose,
                      'this_k': None,
                      'this_sub': None,
                      'num_z_samples': 2 if debug else GRAIL_NUM_Z_SAMPLES,
                      'sigma': GRAIL_SIGMA,
                      'all_rsn_conns': False,
                      'medusa': False,
                      'run_spin_test': False,
                      'grail_features': GRAIL_FEATURES}

    # Model-specific GRAIL settings, e.g. Medusa's three CFRHead outputs
    config_updates.update(MODELS[model].get('grail', {}))

    # GRAIL never reads graph.y, so the permutation is inert here. It is set to keep the
    # config match against the null run's own config.json exact, and to record provenance.
    config_updates['dataset']['perm_seed'] = perm_seed
    config_updates['dataset']['drug_condition'] = None
    run('grail', observer, config_updates)


def main(model, config_file, output_dir, verbose, debug, seed, perm_seed, config_id=0,
         eval_only=False, evaluations=None):
    # Add project root to paths
    config_file = add_project_root(config_file)
    output_dir = add_project_root(output_dir)

    # Make sure the config files exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")

    # Load the config, then rewrite it the way this model's own driver script does
    config = load_configs_from_json(config_file)
    config = fetch_job_config(config, config_id)
    base_config = copy.deepcopy(config)
    apply_overrides = MODELS[model]['apply']
    if apply_overrides is not None:
        config = apply_overrides(config)

    # Experiment settings
    observer = 'FileStorageObserver'
    config['verbose'] = verbose
    config['seed'] = seed
    config['save_weights'] = True
    if debug:
        config['num_epochs'] = 2

    exname = MODELS[model]['exname']
    perm_dir = os.path.join(output_dir, 'permutation_null', f'perm_{perm_seed}')
    ex_dir = os.path.join(perm_dir, f'seed_{seed}')

    # Train the model on permuted outcomes -------------------------------------
    if eval_only:
        if not os.path.exists(ex_dir):
            print(f"Nothing to evaluate: {ex_dir} does not exist.")
            return
    else:
        train_null_model(exname, config, ex_dir, perm_seed, observer)

    # Evaluate the null weights ------------------------------------------------
    to_run = MODELS[model]['evaluations']
    if evaluations is not None:
        unknown = [e for e in evaluations if e not in to_run]
        if unknown:
            raise ValueError(f"Unknown evaluation(s) for {model}: {', '.join(unknown)}. "
                             f"Available: {', '.join(to_run) or 'none'}.")
        to_run = [e for e in to_run if e in evaluations]
    if not to_run:
        return
    if not weights_exist(ex_dir):
        print(f"No fold weights in {ex_dir}; skipping evaluations.")
        return

    if 'transfer_atlas' in to_run:
        run_transfer_atlas(base_config, ex_dir, perm_dir, seed, verbose, observer)
    if 'psilodep1' in to_run:
        run_psilodep1(ex_dir, perm_dir, seed, verbose, debug, observer, config_id)
    if 'grail' in to_run:
        run_grail(model, config, ex_dir, perm_dir, seed, perm_seed, verbose, debug, observer)


if __name__ == "__main__":
    """
    How to run:
    python -m scripts.permutation_null -m graphtrip -p 0 -s 0 -v -dbg
    python -m scripts.permutation_null -m graphtrip -p 0 -s 0 --eval_only
    python -m scripts.permutation_null -m graphtrip -p 0 -s 0 --eval_only --evaluations grail
    python -m scripts.permutation_null -m medusa -p 0 -s 0 --eval_only --evaluations grail
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='graphtrip',
                        choices=list(MODELS.keys()), help='Which model to train')
    parser.add_argument('-c', '--config_file', type=str, default=None,
                        help='Path to the config file; defaults to the config of the model')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Path to the output directory; defaults to that of the model')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Training seed')
    parser.add_argument('-p', '--perm_seed', type=int, default=0,
                        help='Seed of the label permutation; shared by all training seeds')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-ci', '--config_id', type=int, default=None, help='Config ID')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training and only run the evaluations of an existing '
                             'null run; used to backfill earlier permutations')
    parser.add_argument('--evaluations', type=str, nargs='*', default=None,
                        help='Restrict to these evaluations of the null weights, e.g. '
                             '--evaluations grail. Defaults to all of the model\'s.')
    args = parser.parse_args()

    # Fall back onto the model defaults
    if args.config_file is None:
        args.config_file = MODELS[args.model]['config_file']
    if args.output_dir is None:
        args.output_dir = MODELS[args.model]['output_dir']

    # Add config subdirectory into output directory, if config_id is provided
    if args.config_id is not None:
        args.output_dir = os.path.join(args.output_dir, f'config_{args.config_id}')
    else:
        args.config_id = 0

    # Run the main function
    main(args.model, args.config_file, args.output_dir, args.verbose,
         args.debug, args.seed, args.perm_seed, args.config_id, args.eval_only,
         args.evaluations)
