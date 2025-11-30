"""
This scripts performs LOOCV unsupervised training of the VGAE.
The outputs of this script are required for x_graphtrip.py.

Dependencies:
- experiments/configs/graphtrip.json

Outputs:
- outputs/unsupervised_vgae/weights/seed_{seed}/

Author: Hanna M. Tolle
Date: 2025-11-29
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


def main(config_file, output_dir, verbose, debug, seed, jobid, config_id=0):
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
    config['save_weights'] = True
    if debug:
        config['num_epochs'] = 2

    # Job ID determines the fold to train (must be 0 <= jobid < num_folds)
    num_folds_loocv = 42 # 42 patients in the dataset
    if jobid is None:
        this_k = None # train all folds sequentially
    else:
        if jobid < 0 or jobid >= num_folds_loocv:
            raise ValueError(f"Job ID must be 0 <= jobid < {num_folds_loocv}")
        this_k = jobid

    # Train VGAE (unsupervised training) ---------------------------------------
    exname = 'train_vgae'
    vgae_weights_dir = os.path.join(output_dir, 'vgae_weights', f'seed_{seed}')

    # Get dataset and VGAE configs from original graphTRIP config
    config_updates = {}
    config_updates['dataset'] = copy.deepcopy(config['dataset'])
    config_updates['vgae_model'] = copy.deepcopy(config['vgae_model'])

    # Remove labels from dataset config
    config_updates['dataset']['target'] = None
    config_updates['dataset']['graph_attrs'] = []
    config_updates['dataset']['context_attrs'] = []

    # Train with LOOCV
    config_updates['this_k'] = this_k
    config_updates['dataset']['num_folds'] = num_folds_loocv
    config_updates['dataset']['batch_size'] = 7
    config_updates['dataset']['val_split'] = 0.

    # Remove pooling layer from VGAE config
    config_updates['vgae_model']['pooling_cfg'] = {
        'model_type': 'DummyPooling'
    }

    # Training configurations
    config_updates['lr'] = config['lr']
    config_updates['num_epochs'] = config['num_epochs']
    config_updates['balance_attrs'] = None # no balancing, unsupervised training

    # Directories etc.
    config_updates['output_dir'] = vgae_weights_dir
    config_updates['seed'] = seed
    config_updates['verbose'] = verbose
    config_updates['save_weights'] = True
    run(exname, observer, config_updates)

if __name__ == "__main__":
    """
    How to run:
    python x_graphtrip_training.py -c experiments/configs/ -o outputs/ -s 291 -v -dbg -j 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/graphtrip.json', 
                        help='Path to the config file with X-graphTRIP model config')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/x_graphtrip/', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=291, help='Random seed')
    parser.add_argument('-j', '--jobid', type=int, default=None, help='Job ID')
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
    main(args.config_file, args.output_dir, args.verbose, args.debug, args.seed, args.jobid, args.config_id)
