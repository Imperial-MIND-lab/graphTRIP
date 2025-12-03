"""
This scripts trains the main graphTRIP model on the primary dataset.

Dependencies:
- experiments/configs/cfr_graphtrip.json

Outputs:
- outputs/cfr_graphtrip/weights/
- outputs/cfr_graphtrip/permutation_importance/
- outputs/cfr_graphtrip/transfer_atlas/schaefer200/
- outputs/cfr_graphtrip/transfer_atlas/aal/
- outputs/cfr_graphtrip/attention_weights/
- outputs/cfr_graphtrip/grail/

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

    # Train graphTRIP ----------------------------------------------------------
    exname = 'train_cfrnet'
    weights_dir = os.path.join(output_dir, 'weights', f'seed_{seed}')

    # Run the experiment if it doesn't exist
    if not os.path.exists(weights_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = weights_dir
        config_updates['save_weights'] = True
        run(exname, observer, config_updates)
    else:
        print(f"CFR graphTRIP experiment already exists in {weights_dir}.")

    # Permutation importance ---------------------------------------------------
    # Run with drug_condition = [0, 1] to compare drug-specific importances
    exname = 'permutation_importance'
    drug_conditions = [0, 1]
    for drug_condition in drug_conditions:
        drug_name = 'escitalopram' if drug_condition == 0 else 'psilocybin'
        ex_dir = os.path.join(output_dir, 'permutation_importance', drug_name, f'seed_{seed}')
        if not os.path.exists(ex_dir):
            config_updates = {}
            config_updates['dataset'] = {}
            config_updates['dataset']['drug_condition'] = drug_condition
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

    # GRAIL -------------------------------------------------------------------
    # Run GRAIL with drug_condition = [0, 1] to compare drug-specific patterns
    drug_conditions = [0, 1]
    exname = 'grail'
    for drug_condition in drug_conditions:
        drug_name = 'escitalopram' if drug_condition == 0 else 'psilocybin'
        ex_dir = os.path.join(output_dir, 'grail', drug_name, f'seed_{seed}')
        if not os.path.exists(ex_dir):
            config_updates = {}
            config_updates['dataset'] = {}
            config_updates['dataset']['drug_condition'] = drug_condition
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

if __name__ == "__main__":
    """
    How to run:
    python cfr_trip.py -c experiments/configs/cfr_graphtrip.json -o outputs/ -s 291 -v -dbg -ci 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/cfr_trip.json', 
                        help='Path to the config file with graphTRIP model config')
    parser.add_argument('-o', '--output_dir', type=str, 
                        default='outputs/cfr_graphtrip/', help='Path to the output directory')
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
