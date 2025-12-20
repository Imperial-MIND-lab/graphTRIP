"""
This scripts trains causal graphTRIP models.

Dependencies:
- experiments/configs/medusa_graphtrip.json

Outputs:
- outputs/medusa_graphtrip/weights/seed_{seed}/
- outputs/medusa_graphtrip/permutation_importance/seed_{seed}/
- outputs/medusa_graphtrip/estimate_propensity/seed_{seed}/
- outputs/medusa_graphtrip/estimate_propensity_wo_QIDS/seed_{seed}/

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
    if debug:
        config['num_epochs'] = 2

    # Train Medusa graphTRIP model ---------------------------------------------
    exname = 'train_cfrnet'
    weights_dir = os.path.join(output_dir, 'weights', f'seed_{seed}')
    if not os.path.exists(weights_dir):
        print(f"Training Medusa graphTRIP model in {weights_dir}.")
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = weights_dir
        config_updates['save_weights'] = True
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Medusa graphTRIP model already exists in {weights_dir}.")

    # Permutation importance ---------------------------------------------------
    exname = 'permutation_importance'
    ex_dir = os.path.join(output_dir, 'permutation_importance', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        print(f"Running permutation importance in {ex_dir}.")
        config_updates = {}
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['n_repeats'] = 50 if not debug else 2
        run(exname, observer, config_updates)
    else:
        print(f"Permutation importance experiment already exists in {ex_dir}.")

    # Verify sufficient overlap -----------------------------------------------
    exname = 'estimate_propensity'
    ex_dir = os.path.join(output_dir, 'estimate_propensity', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        print(f"Training Medusa propensity model in {ex_dir}.")
        config_updates = {}
        config_updates['logit_C'] = 1.0          # standard logit regularization
        config_updates['n_pca_components'] = 0   # no PCA dimensionality reduction
        config_updates['reinit_pooling'] = False

        # Set target to Condition_bin01
        config_updates['dataset'] = copy.deepcopy(config['dataset'])
        config_updates['dataset']['target'] = 'Condition_bin01'
        
        # Directories etc.
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['save_weights'] = False
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Medusa propensity model experiment already exists in {ex_dir}.")

    # Verify sufficient overlap without QIDS -------------------------------------
    exname = 'estimate_propensity'
    ex_dir = os.path.join(output_dir, 'estimate_propensity_wo_QIDS', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        print(f"Training Medusa propensity model without QIDS in {ex_dir}.")
        
        config_updates = {}
        config_updates['logit_C'] = 1.0          # standard logit regularization
        config_updates['n_pca_components'] = 0   # no PCA dimensionality reduction
        config_updates['reinit_pooling'] = False

        # Remove QIDS_Before from graph_attrs and set target to Condition_bin01
        config_updates['dataset'] = copy.deepcopy(config['dataset'])
        graph_attrs = config_updates['dataset']['graph_attrs']
        new_graph_attrs = [attr for attr in graph_attrs if attr != 'QIDS_Before']
        config_updates['dataset']['graph_attrs'] = new_graph_attrs
        config_updates['dataset']['target'] = 'Condition_bin01'

        # Directories etc.
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['save_weights'] = False
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Medusa propensity model experiment already exists in {ex_dir}.")


if __name__ == "__main__":
    """
    How to run:
    python medusa.py 
       -c experiments/configs/medusa_graphtrip.json 
       -o outputs/medusa_graphtrip/ 
       -s 0 -v -dbg -ci 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/medusa_graphtrip.json', 
                        help='Path to the config file with Medusa graphTRIP model config')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/medusa_graphtrip/', help='Path to the output directory')
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
