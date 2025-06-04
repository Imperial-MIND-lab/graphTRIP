"""
This scripts trains the T-learners for X-graphTRIP.

Dependencies:
- experiments/configs/tlearner_escitalopram.json
- experiments/configs/tlearner_psilocybin.json

Outputs:
- outputs/t_learners/tlearner_escitalopram/
- outputs/t_learners/tlearner_psilocybin/

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
    exname = 'train_tlearner'
    if debug:
        config['num_epochs'] = 2

    # Train T-learner ----------------------------------------------------------
    if not os.path.exists(output_dir):
        config_updates = copy.deepcopy(config)
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['output_dir'] = output_dir
        run(exname, observer, config_updates)
    else:
        print(f"T-learner experiment already exists in {output_dir}.")

if __name__ == "__main__":
    """
    How to run:
    python tlearners.py -c experiments/configs/tlearner_escitalopram.json -o outputs/t_learners/ -s 291 -v -dbg -ci 0
    python tlearners.py -c experiments/configs/tlearner_psilocybin.json -o outputs/t_learners/ -s 291 -v -dbg -ci 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/tlearner_escitalopram.json', 
                        help='Path to the config file with T-learner config')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/t_learners/', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=291, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-ci', '--config_id', type=int, default=None, help='Config ID')
    args = parser.parse_args()
        
    # Add the name of the config file to the output directory
    args.output_dir = os.path.join(args.output_dir, os.path.basename(args.config_file).split('.')[0])

    # Add config subdirectory into output directory, if config_id is provided
    if args.config_id is not None:
        args.output_dir = os.path.join(args.output_dir, f'config_{args.config_id}')
    else:
        args.config_id = 0

    # Run the main function
    main(args.config_file, args.output_dir, args.verbose, args.debug, args.seed, args.config_id)
