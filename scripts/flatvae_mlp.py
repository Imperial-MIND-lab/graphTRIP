"""
This script trains the non-graph baseline: a flat VAE on the vectorised FC matrix and node
features, jointly with the same MLP prediction head as graphTRIP.

Dependencies:
- experiments/configs/flatvae_mlp.json

Outputs:
- outputs/flatvae_mlp/seed_{seed}/

Author: Hanna M. Tolle
Date: 2026-04-02
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
sys.path.append("../")

import os
import argparse
from utils.files import add_project_root
from utils.configs import load_configs_from_json, fetch_job_config
from experiments.run_experiment import run


def main(config_file, output_dir, verbose, debug, seed):
    # Add project root to paths
    config_file = add_project_root(config_file)
    output_dir = add_project_root(output_dir)

    # Make sure the config file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")

    # Load the config
    config = load_configs_from_json(config_file)
    config = fetch_job_config(config, 0)

    # Train the baseline -------------------------------------------------------
    ex_dir = os.path.join(output_dir, f'seed_{seed}')
    if os.path.exists(ex_dir):
        print(f"FlatVAE baseline already exists in {ex_dir}.")
        return

    config['jobid'] = 0
    config['run_name'] = 'flatvae_mlp'
    config['output_dir'] = ex_dir
    config['seed'] = seed
    if verbose:
        config['verbose'] = True
    if debug:
        config['num_epochs'] = 2
    run('train_jointly', 'FileStorageObserver', config)


if __name__ == "__main__":
    """
    How to run:
    python -m scripts.flatvae_mlp -s 0 -v -dbg
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str,
                        default='experiments/configs/flatvae_mlp.json',
                        help='Path to the config file with the FlatVAE baseline config')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/flatvae_mlp/',
                        help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    main(args.config_file, args.output_dir, args.verbose, args.debug, args.seed)
