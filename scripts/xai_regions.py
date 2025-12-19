"""
This scripts runs the regional attribution experiment.

Dependencies:
- outputs/graphtrip/weights/

Outputs:
- outputs/graphtrip/regional_attributions/

Author: Hanna M. Tolle
Date: 2025-05-31
License: BSD 3-Clause
"""
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import sys
sys.path.append("../")

import os
import argparse
from utils.files import add_project_root
from experiments.run_experiment import run


def main(weights_base_dir, output_dir, verbose, debug, seed, mlp_weights_dir=None, medusa=False):
    # Add project root to paths
    weights_base_dir = add_project_root(weights_base_dir)
    output_dir = add_project_root(output_dir)

    if mlp_weights_dir is None:
        mlp_weights_dir = weights_base_dir
    else:
        mlp_weights_dir = add_project_root(mlp_weights_dir)
        if not os.path.exists(mlp_weights_dir):
            raise FileNotFoundError(f"{mlp_weights_dir} not found")

    # Make sure the weights base directory exists
    if not os.path.exists(weights_base_dir):
        raise FileNotFoundError(f"{weights_base_dir} not found")
    
    # Experiment settings
    observer = 'FileStorageObserver'
    weights_dir = os.path.join(weights_base_dir, f'seed_{seed}')
    mlp_weights_dir = os.path.join(mlp_weights_dir, f'seed_{seed}')
    exname = 'regional_attributions'
    ex_dir = os.path.join(output_dir, f'seed_{seed}')
    
    # Common configs
    config_updates = {}
    config_updates['output_dir'] = ex_dir
    config_updates['vgae_weights_dir'] = weights_dir
    config_updates['mlp_weights_dir'] = mlp_weights_dir
    config_updates['seed'] = seed
    config_updates['verbose'] = verbose
    config_updates['num_z_samples'] = 100 if not debug else 2
    config_updates['sigma'] = 2.0
    config_updates['medusa'] = medusa
    
    # Run experiment
    run(exname, observer, config_updates)


if __name__ == "__main__":
    """
    How to run:
    python xai_regions.py -w outputs/graphtrip/weights/ -o outputs/graphtrip/regional_attributions/ -s 0 -v -dbg
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_base_dir', type=str, default='outputs/graphtrip/weights/', 
                        help='Path to the base directory with VGAE weights')
    parser.add_argument('-o', '--output_dir', type=str, default=None, 
                        help='Path to the output directory. If None, uses weights_base_dir/../regional_attributions/')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--mlp_weights_dir', type=str, default=None, 
                        help='Path to the directory with MLP weights. If None, use the same as the VGAE weights.')
    parser.add_argument('--medusa', action='store_true', help='Run Medusa mode')
    args = parser.parse_args()

    # If output directory is not provided, use weights_base_dir/seed_<seed>/regional_attributions/
    if args.output_dir is None:
        args.output_dir = os.path.join(args.weights_base_dir, '../regional_attributions')

    # Run the main function
    main(args.weights_base_dir, args.output_dir, args.verbose, args.debug, args.seed, args.mlp_weights_dir, args.medusa)
