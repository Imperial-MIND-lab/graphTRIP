"""
This script trains graphTRIP or Medusa graphTRIP on permuted outcomes, to build an
empirical null distribution of prediction performance.

Dependencies:
- experiments/configs/graphtrip.json
- experiments/configs/medusa_graphtrip.json

Outputs:
- outputs/graphtrip/permutation_null/perm_{perm_seed}/seed_{seed}/
- outputs/medusa_graphtrip/permutation_null/perm_{perm_seed}/seed_{seed}/

Author: Hanna M. Tolle
Date: 2026-08-21
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
sys.path.append("../")

import os
import copy
import argparse
from utils.files import add_project_root
from utils.configs import load_configs_from_json, fetch_job_config
from experiments.run_experiment import run


# Model settings: config file, sacred experiment, and default output directory.
MODELS = {
    'graphtrip': {'config_file': 'experiments/configs/graphtrip.json',
                  'exname': 'train_jointly',
                  'output_dir': 'outputs/graphtrip/'},
    'medusa': {'config_file': 'experiments/configs/medusa_graphtrip.json',
               'exname': 'train_cfrnet',
               'output_dir': 'outputs/medusa_graphtrip/'}
}


def main(model, config_file, output_dir, verbose, debug, seed, perm_seed, config_id=0):
    # Add project root to paths
    config_file = add_project_root(config_file)
    output_dir = add_project_root(output_dir)

    # Make sure the config files exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")

    # Load the config
    config = load_configs_from_json(config_file)
    config = fetch_job_config(config, config_id)

    # Experiment settings
    observer = 'FileStorageObserver'
    config['verbose'] = verbose
    config['seed'] = seed
    config['save_weights'] = True
    if debug:
        config['num_epochs'] = 2

    # Train the model on permuted outcomes -------------------------------------
    exname = MODELS[model]['exname']
    ex_dir = os.path.join(output_dir, 'permutation_null', f'perm_{perm_seed}', f'seed_{seed}')

    # Run the experiment if it doesn't exist
    if not os.path.exists(ex_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = ex_dir
        config_updates['save_weights'] = True
        config_updates['dataset']['perm_seed'] = perm_seed
        run(exname, observer, config_updates)
    else:
        print(f"{model} permutation null experiment already exists in {ex_dir}.")


if __name__ == "__main__":
    """
    How to run:
    python -m scripts.permutation_null -m graphtrip -p 0 -s 0 -v -dbg
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
         args.debug, args.seed, args.perm_seed, args.config_id)
