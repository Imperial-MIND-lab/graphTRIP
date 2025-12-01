"""
This scripts trains t-learners (separate models for each condition) 
by loading a pre-trained VGAE and training MLP regression heads on the 
latent representations + clinical data for each condition separately.

Dependencies:
- experiments/configs/tlearners.json

Outputs:
- outputs/x_graphtrip/tlearners_torch/seed_{seed}/

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


def main(config_file, output_dir, verbose, debug, seed, config_id=0):
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
    if debug:
        config['num_epochs'] = 2

    # Directory with pre-trained VGAE weights
    vgae_weights_dir = os.path.join('outputs', 'x_graphtrip', 'vgae_weights', f'seed_{seed}')
    vgae_weights_dir = add_project_root(vgae_weights_dir)

    # Required configs
    assert 'target' in config['dataset'], "Dataset must have a target"
    assert 'mlp_model' in config, "Config must have an MLP model"

    # Train T-learners ------------------------------------------------------------
    exname = 'train_tlearners_torch'

    # Change MLP model to NonNegativeRegressionMLP if target is positive
    positive_targets = ['QIDS_Final_Integration', 'BDI_Final_Integration']
    target = config['dataset']['target']
    if target in positive_targets:
        config['mlp_model']['model_type'] = 'NonNegativeRegressionMLP'

    ex_dir = os.path.join(output_dir, target, f'config_{config_id}', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = vgae_weights_dir
        config_updates['save_weights'] = False
        run(exname, observer, config_updates)
    else:
        print(f"T-learner experiment already exists in {ex_dir}.")

if __name__ == "__main__":
    """
    How to run:
    python tlearners.py -c experiments/configs/tlearners.json -o outputs/x_graphtrip/ -s 0 -v -dbg -j 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/tlearners.json', 
                        help='Path to the config file with t-learner config')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/x_graphtrip/tlearners_torch', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-ci', '--config_id', type=int, default=None, help='Config ID')
    args = parser.parse_args()

    # Run the main function
    main(args.config_file, args.output_dir, args.verbose, args.debug, args.seed, args.config_id)
