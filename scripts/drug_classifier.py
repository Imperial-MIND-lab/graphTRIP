"""
This scripts trains a graphTRIP-type model with logistic regression head
to classify treatment assignment from pre-treatment data.

Dependencies:
- experiments/configs/drug_classifier.json

Outputs:
- outputs/drug_classifier/weights/

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
    config_file = 'drug_classifier.json'
    if not os.path.exists(os.path.join(config_dir, config_file)):
        raise FileNotFoundError(f"{config_file} not found in {config_dir}")
    
    # Load the config
    config = load_configs_from_json(os.path.join(config_dir, config_file))
    config = fetch_job_config(config, 0)
        
    # Experiment settings
    observer = 'FileStorageObserver'
    config['verbose'] = verbose
    config['seed'] = seed
    config['save_weights'] = True
    if debug:
        config['num_epochs'] = 2

    # Train drug classifier ----------------------------------------------------
    exname = 'train_classifier'
    weights_dir = os.path.join(output_dir, 'weights')

    # Run the experiment if it doesn't exist
    if not os.path.exists(weights_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = weights_dir
        run(exname, observer, config_updates)
    else:
        print(f"drug classifier weights already exist in {weights_dir}.")

if __name__ == "__main__":
    """
    How to run:
    python drug_classifier.py -c experiments/configs/ -o outputs/ -s 291 -v -dbg
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_dir', type=str, default='experiments/configs/', help='Path to the config directory')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/drug_classifier/', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=291, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Make sure debugging outputs don't overwrite any existing outputs
    if args.debug:
        if args.output_dir == 'outputs/drug_classifier/':
            raise ValueError("output_dir must be specified when using debug mode.")

    # Run the main function
    main(args.config_dir, args.output_dir, args.verbose, args.debug, args.seed)
