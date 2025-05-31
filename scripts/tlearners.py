"""
This scripts trains the T-learners for X-graphTRIP.

Dependencies:
- experiments/configs/tlearner_escitalopram.json
- experiments/configs/tlearner_psilocybin.json

Outputs:
- outputs/x_graphtrip/tlearner_escitalopram/
- outputs/x_graphtrip/tlearner_psilocybin/

Author: Hanna M. Tolle
Date: 2025-05-31
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


def main(config_dir, output_dir, verbose, debug, seed):
    # Add project root to paths
    config_dir = add_project_root(config_dir)
    output_dir = add_project_root(output_dir)

    # Make sure the config files exist
    config_file_E = 'tlearner_escitalopram.json'
    config_file_P = 'tlearner_psilocybin.json'
    if not os.path.exists(os.path.join(config_dir, config_file_E)):
        raise FileNotFoundError(f"{config_file_E} not found in {config_dir}")
    if not os.path.exists(os.path.join(config_dir, config_file_P)):
        raise FileNotFoundError(f"{config_file_P} not found in {config_dir}")
    
    # Load the config
    config_E = load_configs_from_json(os.path.join(config_dir, config_file_E))
    config_P = load_configs_from_json(os.path.join(config_dir, config_file_P))
    config_E = fetch_job_config(config_E, 0)
    config_P = fetch_job_config(config_P, 0)

    # Experiment settings
    observer = 'FileStorageObserver'
    config_E['verbose'] = verbose
    config_E['seed'] = seed
    config_E['save_weights'] = True
    if debug:
        config_E['num_epochs'] = 2
        config_P['num_epochs'] = 2

    # Train escitalopram T-learner ----------------------------------------------
    exname = 'train_tlearner'
    weights_dir = os.path.join(output_dir, 'tlearner_escitalopram')

    # Run the experiment if it doesn't exist
    if not os.path.exists(weights_dir):
        config_updates = copy.deepcopy(config_E)
        config_updates['output_dir'] = weights_dir
        run(exname, observer, config_updates)
    else:
        print(f"T-learner experiment already exists in {weights_dir}.")

    # Train psilocybin T-learner ----------------------------------------------
    exname = 'train_tlearner'
    weights_dir = os.path.join(output_dir, 'tlearner_psilocybin')

    # Run the experiment if it doesn't exist
    if not os.path.exists(weights_dir):
        config_updates = copy.deepcopy(config_P)
        config_updates['output_dir'] = weights_dir
        run(exname, observer, config_updates)
    else:
        print(f"T-learner experiment already exists in {weights_dir}.")

if __name__ == "__main__":
    """
    How to run:
    python tlearners.py -c experiments/configs/ -o outputs/x_graphtrip/ -s 291 -v -dbg
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_dir', type=str, default='experiments/configs/', help='Path to the config directory')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/x_graphtrip/', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=291, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Make sure debugging outputs don't overwrite any existing outputs
    if args.debug:
        if args.output_dir == 'outputs/x_graphtrip/':
            raise ValueError("output_dir must be specified when using debug mode.")

    # Run the main function
    main(args.config_dir, args.output_dir, args.verbose, args.debug, args.seed)
