"""
This scripts runs the interpretability experiments.

Dependencies:
- outputs/graphtrip/weights/

Outputs:
- outputs/graphtrip/grail/

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
from utils.configs import load_configs_from_json
from experiments.run_experiment import run


def main(weights_base_dir, output_dir, verbose, debug, seed, job_id=None, mlp_weights_dir=None):
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

    # Load the config from mlp_weights_dir
    config_file = os.path.join(mlp_weights_dir, 'config.json')
    config = load_configs_from_json(config_file)

    # GRAIL -------------------------------------------------------------------
    exname = 'grail'
    ex_dir = os.path.join(output_dir, f'seed_{seed}')
    # Common configs
    config_updates = {}
    config_updates['this_k'] = job_id
    config_updates['output_dir'] = ex_dir
    config_updates['vgae_weights_dir'] = weights_dir
    config_updates['mlp_weights_dir'] = mlp_weights_dir
    config_updates['seed'] = seed
    config_updates['verbose'] = verbose
    config_updates['num_z_samples'] = 100 if not debug else 2
    config_updates['sigma'] = 2.0
    config_updates['all_rsn_conns'] = False

    # Model-specific configs
    # CATE models don't have MLP config and need SklearnLinearModelWrapper
    if config['exname'] == 'train_scate':
        config_updates['mlp_model'] = {
            'model_type': 'SklearnLinearModelWrapper'}
        config_updates['weight_filenames'] = {
            'vgae': [f'k{k}_vgae_weights.pth' for k in range(config['dataset']['num_folds'])],
            'mlp': [f'k{k}_linear_model.pth' for k in range(config['dataset']['num_folds'])],
            'test_fold_indices': ['test_fold_indices.csv']}

    # CFRNet t-learners should do medusa grail
    elif config['exname'] == 'train_cfrnet':
        config_updates['medusa'] = True

    # Run the experiment
    run(exname, observer, config_updates)


if __name__ == "__main__":
    """
    How to run:
    python interpretation.py -w outputs/graphtrip/weights/ -o outputs/graphtrip/grail/ -s 0 -v -dbg
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_base_dir', type=str, 
                        default='outputs/graphtrip/weights/', 
                        help='Path to the base directory with graphTRIP model weights')
    parser.add_argument('-o', '--output_dir', type=str, 
                        default='outputs/graphtrip/grail/', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-j', '--job_id', type=int, default=None, help='Job ID')
    parser.add_argument('--mlp_weights_dir', type=str, default=None, 
                        help='Path to the directory with MLP weights. If None, use the same as the VGAE weights.')
    args = parser.parse_args()

    # Run the main function
    main(args.weights_base_dir, args.output_dir, args.verbose, args.debug, args.seed, args.job_id, args.mlp_weights_dir)
