"""
This scripts trains a graphTRIP model to predict BDI scores.
The model config are the same as the graphTRIP config, except
for the prediction target.

Dependencies:
- experiments/configs/graphtrip.json

Outputs:
- outputs/graphtrip_bdi/weights/
- outputs/graphtrip_bdi/permutation_importance/

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
    ingredient_config = {
        'dataset': copy.deepcopy(config['dataset']),
        'vgae_model': copy.deepcopy(config['vgae_model']),
        'mlp_model': copy.deepcopy(config['mlp_model'])
    }
        
    # Experiment settings
    observer = 'FileStorageObserver'
    config['verbose'] = verbose
    config['seed'] = seed
    config['save_weights'] = True
    if debug:
        config['num_epochs'] = 2
    ingredient_config['seed'] = seed
    ingredient_config['verbose'] = verbose

    # Train graphTRIP to predict BDI --------------------------------------------
    exname = 'train_jointly'
    weights_dir = os.path.join(output_dir, 'weights')

    # Run the experiment if it doesn't exist
    if not os.path.exists(weights_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = weights_dir
        config_updates['dataset']['target'] = 'BDI_Final_Integration'
        run(exname, observer, config_updates)
    else:
        print(f"graphtrip_bdi weights already exist in {weights_dir}.")

    # Permutation importance ---------------------------------------------------
    exname = 'permutation_importance'
    ex_dir = os.path.join(output_dir, 'permutation_importance')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['n_repeats'] = 50 if not debug else 2
        run(exname, observer, config_updates)
    else:
        print(f"Permutation importance experiment already exists in {ex_dir}.")

if __name__ == "__main__":
    """
    How to run:
    python graphtrip_bdi.py -c experiments/configs/graphtrip.json -o outputs/graphtrip_bdi/ -s 291 -v -dbg -ci 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/graphtrip.json', 
                        help='Path to the config file with graphTRIP model config')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/graphtrip_bdi/', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=291, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-ci', '--config_id', type=int, default=None, help='Config ID')
    args = parser.parse_args()

    # Add config subdirectory into output directory, if config_id is provided
    if args.config_id is not None:
        args.output_dir = os.path.join(args.output_dir, f'config_{args.config_id}')
    else:
        args.config_id = 0

    # Run the main function
    main(args.config_file, args.output_dir, args.verbose, args.debug, args.seed, args.config_id)
