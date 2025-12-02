"""
This scripts trains X-graphTRIP models for many train-test splits.
T-learners must be trained first with tlearners.py.

Dependencies:
- experiments/configs/graphtrip.json

Outputs:
- outputs/x_graphtrip/vgae_weights/seed_{seed}/test_fold_indices.csv
- outputs/x_graphtrip/estimate_propensity/seed_{seed}/
- outputs/x_graphtrip/tlearner_escitalopram/seed_{seed}/
- outputs/x_graphtrip/tlearner_psilocybin/seed_{seed}/
- outputs/x_graphtrip/cate_model_escitalopram/seed_{seed}/
- outputs/x_graphtrip/cate_model_psilocybin/seed_{seed}/

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
    config['save_weights'] = True
    if debug:
        config['num_epochs'] = 2
    num_folds_loocv = 42 # 42 patients in the dataset

    # Evaluate unsupervised VGAE -------------------------------------------------
    # Must have been trained with unsupervised_vgae.py first.
    exname = 'train_vgae'
    vgae_weights_dir = os.path.join(output_dir, 'vgae_weights', f'seed_{seed}')
    test_fold_indices_file = os.path.join(vgae_weights_dir, 'test_fold_indices.csv')

    # If test_fold_indices.csv exists, then it has already been evaluated
    if not os.path.exists(test_fold_indices_file):
        # Get dataset and VGAE configs from original graphTRIP config
        config_updates = {}
        config_updates['dataset'] = copy.deepcopy(config['dataset'])
        config_updates['vgae_model'] = copy.deepcopy(config['vgae_model'])

        # Remove labels from dataset config
        config_updates['dataset']['target'] = None
        config_updates['dataset']['graph_attrs'] = []
        config_updates['dataset']['context_attrs'] = []

        # Train with LOOCV
        config_updates['this_k'] = num_folds_loocv  # triggers evaluation run
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
        config_updates['balance_attrs'] = None 

        # Directories etc.
        config_updates['output_dir'] = vgae_weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['save_weights'] = True
        run(exname, observer, config_updates)
    else:
        print(f"VGAE weights already exist in {vgae_weights_dir}.")

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
    main(args.config_file, args.output_dir, args.verbose, args.debug, args.seed, args.config_id)
