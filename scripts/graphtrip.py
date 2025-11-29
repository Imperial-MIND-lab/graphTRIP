"""
This scripts trains the main graphTRIP model on the primary dataset.

Dependencies:
- experiments/configs/graphtrip.json

Outputs:
- outputs/graphtrip/weights/
- outputs/graphtrip/permutation_importance/
- outputs/graphtrip/transfer_atlas/schaefer200/
- outputs/graphtrip/transfer_atlas/aal/
- outputs/graphtrip/attention_weights/
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
import copy
import argparse
from utils.files import add_project_root
from utils.configs import load_configs_from_json, fetch_job_config
from experiments.run_experiment import run


def main(config_file, output_dir, verbose, debug, seed, config_id=None):

    # Add project root to paths
    config_file = add_project_root(config_file)

    # Make sure the config files exist
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")
    
    # Load the config
    config = load_configs_from_json(config_file)
    if config_id is not None:
        config = fetch_job_config(config, config_id)
    else:
        config = fetch_job_config(config, 0)

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

    # Output directory
    if output_dir is None:
        output_dir = config.get('output_dir', 'outputs/graphtrip/')
    output_dir = add_project_root(output_dir)

    # Add config subdirectory into output directory, if config_id is provided
    if config_id is not None:
        output_dir = os.path.join(output_dir, f'config_{config_id}')
    else:
        config_id = 0
    output_dir = add_project_root(output_dir)
    config['output_dir'] = output_dir

    # Train graphTRIP ----------------------------------------------------------
    exname = 'train_jointly'
    weights_dir = os.path.join(output_dir, 'weights', f'seed_{seed}')

    # Run the experiment if it doesn't exist
    if not os.path.exists(weights_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = weights_dir
        config_updates['save_weights'] = True
        run(exname, observer, config_updates)
    else:
        print(f"graphTRIP experiment already exists in {weights_dir}.")

    # Permutation importance ---------------------------------------------------
    exname = 'permutation_importance'
    ex_dir = os.path.join(output_dir, 'permutation_importance', f'seed_{seed}')
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

    # Transfer to Schaefer200 and AAL -----------------------------------------
    exname = 'test_and_finetune'
    atlases = ['schaefer200', 'aal']

    for atlas in atlases:
        ex_dir = os.path.join(output_dir, 'transfer_atlas', atlas, f'seed_{seed}')
        if not os.path.exists(ex_dir):
            # Get ingredient configs
            config_updates = copy.deepcopy(ingredient_config)
            
            # Change the atlas
            config_updates['dataset']['atlas'] = atlas
            num_nodes = 200 if atlas == 'schaefer200' else 116
            config_updates['dataset']['num_nodes'] = num_nodes
            config_updates['vgae_model']['params']['num_nodes'] = num_nodes

            # Other settings
            config_updates['num_epochs'] = 0 # no finetuning
            config_updates['output_dir'] = ex_dir
            config_updates['weights_dir'] = weights_dir
            config_updates['save_weights'] = False 

            run(exname, observer, config_updates)
        else:
            print(f"Transfer to {atlas} experiment already exists in {ex_dir}.")

    # Attention weights -------------------------------------------------------
    exname = 'attention_weights'
    ex_dir = os.path.join(output_dir, 'attention_weights', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Attention weights experiment already exists in {ex_dir}.")

    # GRAIL -------------------------------------------------------------------
    exname = 'grail'
    ex_dir = os.path.join(output_dir, 'grail', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['num_z_samples'] = 100 if not debug else 2
        config_updates['sigma'] = 2.0
        config_updates['all_rsn_conns'] = False
        run(exname, observer, config_updates)
    else:
        print(f"GRAIL experiment already exists in {ex_dir}.")

    # Train linear regression head on VGAE representations ------------------------
    exname = 'train_linreg_on_z'
    ex_dir = os.path.join(output_dir, 'linreg_on_z', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        
        # Training configurations
        config_updates = {}
        config_updates['ridge_alpha'] = 1.0    # standard setting
        config_updates['n_pca_components'] = 0 # no PCA dimensionality reduction

        # Other settings
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['save_weights'] = False 
        run(exname, observer, config_updates)
    else:
        print(f"Linear regression head experiment already exists in {ex_dir}.")

    # Ablation analysis: train VGAE with a regression head end-to-end ------------------------
    exname = 'train_jointly'
    ex_dir = os.path.join(output_dir, 'vgae_with_regression_head', f'seed_{seed}')
    if not os.path.exists(ex_dir):
        config_updates = copy.deepcopy(config)
        config_updates['mlp_model']['params']['num_layers'] = 1 # no hidden layers
        config_updates['output_dir'] = ex_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        config_updates['save_weights'] = False 
        run(exname, observer, config_updates)
    else:
        print(f"Train VGAE with regression head experiment already exists in {ex_dir}.")

if __name__ == "__main__":
    """
    How to run:
    python graphtrip.py -c experiments/configs/graphtrip.json -o outputs/ -s 291 -v -dbg -ci 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/graphtrip.json', 
                        help='Path to the config file with graphTRIP model config')
    parser.add_argument('-o', '--output_dir', type=str, 
                        default=None, help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-ci', '--config_id', type=int, default=None, help='Config ID')
    args = parser.parse_args()

    # Run the main function
    main(args.config_file, args.output_dir, args.verbose, args.debug, args.seed, args.config_id)
