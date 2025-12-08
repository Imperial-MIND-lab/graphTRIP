"""
This scripts validates the graphTRIP VGAE representations
on and independent dataset (psilodep1) by re-training a new
MLP on the frozen VGAE latent representations of the psilodep1 dataset.

Dependencies:
- experiments/configs/transfer_vgae.json

Outputs:
- outputs/validation/transfer_vgae/

Author: Hanna M. Tolle
Date: 2025-12-08
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')  

import sys
sys.path.append("../")

import os
import copy
import argparse
from utils.files import add_project_root
from utils.configs import load_configs_from_json, fetch_job_config
from experiments.run_experiment import run


def main(config_file, weights_base_dir, output_dir, verbose, debug, seed, config_id=0):
    # Add project root to paths
    config_file = add_project_root(config_file)
    output_dir = add_project_root(output_dir)
    weights_base_dir = add_project_root(weights_base_dir)

    # Make sure the config and weights base directory exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")
    if not os.path.exists(weights_base_dir):
        raise FileNotFoundError(f"{weights_base_dir} not found")

    # Load the config
    config = load_configs_from_json(config_file)
    config = fetch_job_config(config, config_id)
        
    # Experiment settings
    observer = 'FileStorageObserver'
    if debug:
        config['num_epochs'] = 2

    # Weights directory is this seed's graphTRIP weights directory
    graphtrip_weights_dir = os.path.join(weights_base_dir, f'seed_{seed}')

    # Transfer VGAE on Psilodep1 dataset ----------------------------------------
    exname = 'transfer_vgae'
    ex_dir = os.path.join(output_dir, 'transfer_vgae', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = graphtrip_weights_dir
        config_updates['save_weights'] = False
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Transfer VGAE experiment already exists in {ex_dir}.")

if __name__ == "__main__":
    """
    How to run:
    python transfer.py -c experiments/configs/transfer_vgae.json -o outputs/validation/ -s 0 -v -dbg -ci 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/transfer_vgae.json', 
                        help='Path to the config file with transfer VGAE model config')
    parser.add_argument('-w', '--weights_base_dir', type=str, default='outputs/graphtrip/weights/', 
                        help='Path to the base directory with graphTRIP VGAE weights')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/validation/', 
                        help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
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
    main(args.config_file, args.weights_base_dir, args.output_dir, args.verbose, args.debug, args.seed, args.config_id)
