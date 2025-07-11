"""
This scripts trains X-graphTRIP models for many train-test splits.
T-learners must be trained first with tlearners.py.

Dependencies:
- experiments/configs/x_graphtrip.json
- experiments/configs/x_graphtrip/tlearner_escitalopram/
- experiments/configs/x_graphtrip/tlearner_psilocybin/

Outputs:
- outputs/x_graphtrip/xlearner/job_{fold_shift}/

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


def main(config_file, output_dir, verbose, debug, seed, jobid=0, config_id=0):
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

    # Train X-graphTRIP ---------------------------------------------------------
    exname = 'train_xlearner'
    weights_dir = os.path.join(output_dir, 'weights', f'job_{jobid}')

    # Run the experiment if it doesn't exist
    if not os.path.exists(weights_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = weights_dir
        config_updates['dataset']['fold_shift'] = jobid
        run(exname, observer, config_updates)
    else:
        print(f"X-graphTRIP experiment already exists in {weights_dir}.")

    # Train drug classifier -----------------------------------------------------
    exname = 'train_rep_classifier'
    ex_dir = os.path.join(output_dir, 'drug_classifier', f'job_{jobid}')

    # Run the experiment if it doesn't exist
    if not os.path.exists(ex_dir):
        config_updates = copy.deepcopy(config)
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['save_weights'] = False # no need to save weights

        # Remove incompatible configs
        del config_updates['alpha']
        del config_updates['t0_pred_file']
        del config_updates['t1_pred_file']

        # Change MLP head and prediction target
        config_updates['mlp_model']['model_type'] = 'LogisticRegressionMLP'
        config_updates['dataset']['target'] = 'Condition_bin01'

        run(exname, observer, config_updates)
    else:
        print(f"Drug classifier experiment already exists in {weights_dir}.")

    # Attention weights ---------------------------------------------------------
    exname = 'attention_weights'
    ex_dir = os.path.join(output_dir, 'attention_weights', f'job_{jobid}')
    if not os.path.exists(ex_dir):
        config_updates = {}
        config_updates['output_dir'] = ex_dir
        config_updates['weights_dir'] = weights_dir
        config_updates['seed'] = seed
        config_updates['verbose'] = verbose
        run(exname, observer, config_updates)
    else:
        print(f"Attention weights experiment already exists in {ex_dir}.")

    # GRAIL --------------------------------------------------------------------
    exname = 'grail'
    ex_dir = os.path.join(output_dir, 'grail', f'job_{jobid}')
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
    

if __name__ == "__main__":
    """
    How to run:
    python x_graphtrip_training.py -c experiments/configs/ -o outputs/ -s 291 -v -dbg -j 0
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, 
                        default='experiments/configs/x_graphtrip.json', 
                        help='Path to the config file with X-graphTRIP model config')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/x_graphtrip/', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=291, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-j', '--jobid', type=int, default=0, help='Job ID')
    parser.add_argument('-ci', '--config_id', type=int, default=None, help='Config ID')
    args = parser.parse_args()

    # Add config subdirectory into output directory, if config_id is provided
    if args.config_id is not None:
        args.output_dir = os.path.join(args.output_dir, f'config_{args.config_id}')
    else:
        args.config_id = 0

    # Run the main function
    main(args.config_file, args.output_dir, args.verbose, args.debug, args.seed, args.jobid, args.config_id)
