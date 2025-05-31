"""
This scripts performs all post-hoc analyses on the X-graphTRIP models.

Dependencies:
- outputs/x_graphtrip/xlearner/job_{fold_shift}/

Outputs:
- outputs/x_graphtrip/xlearner/job_{fold_shift}/

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
    config_file = 'x_graphtrip.json'
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

    # Train drug classifier -----------------------------------------------------
    exname = 'train_rep_classifier'
    weights_dir = os.path.join(output_dir, 'drug_classifier')

    # Run the experiment if it doesn't exist
    if not os.path.exists(weights_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = weights_dir
        config_updates['weights_dir'] = 'outputs/x_graphtrip/xlearner/job_0/'

        # Remove incompatible configs
        del config_updates['alpha']
        del config_updates['t0_pred_file']
        del config_updates['t1_pred_file']

        # Change MLP head and prediction target
        config_updates['mlp_model']['model_type'] = 'LogisticRegressionMLP'
        config_updates['dataset']['target'] = 'Condition_bin01'

        run(exname, observer, config_updates)
    else:
        print(f"Drug classifier experiment already exists in {weights_dir}.")

if __name__ == "__main__":
    """
    How to run:
    python x_graphtrip_analysis.py -c experiments/configs/ -o outputs/ -s 291 -v -dbg
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
