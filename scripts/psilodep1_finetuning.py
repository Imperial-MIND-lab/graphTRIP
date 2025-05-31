"""
This script trains models on the Psilodep1 dataset (the validation dataset).
Performs pre-training and finetuning.

Dependencies:
- experiments/configs/graphtrip.json
- experiments/configs/psilodep1_finetuning.json

Outputs:
- outputs/psilodep1/pretrained_weights/
- outputs/psilodep1/finetuned_weights/

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


def main(config_dir, output_dir, verbose, debug, seed):
    # Add project root to paths
    config_dir = add_project_root(config_dir)
    output_dir = add_project_root(output_dir)

    # Make sure the config files exist
    psilodep2_config_file = 'graphtrip.json'
    psilodep1_config_file = 'psilodep1_finetuning.json'
    if not os.path.exists(os.path.join(config_dir, psilodep2_config_file)):
        raise FileNotFoundError(f"{psilodep2_config_file} not found in {config_dir}")
    if not os.path.exists(os.path.join(config_dir, psilodep1_config_file)):
        raise FileNotFoundError(f"{psilodep1_config_file} not found in {config_dir}")
    
    # Load the config
    psilodep2_config = load_configs_from_json(os.path.join(config_dir, psilodep2_config_file))
    psilodep2_config = fetch_job_config(psilodep2_config, 0)
    psilodep1_config = load_configs_from_json(os.path.join(config_dir, psilodep1_config_file))
    psilodep1_config = fetch_job_config(psilodep1_config, 0)
        
    # Experiment settings
    observer = 'FileStorageObserver'
    psilodep1_config['seed'] = seed
    psilodep1_config['verbose'] = verbose
    psilodep2_config['seed'] = seed
    psilodep2_config['verbose'] = verbose
    if debug:
        psilodep2_config['num_epochs'] = 2
        psilodep1_config['num_epochs'] = 2

    # 1. Pretraining ----------------------------------------------------------
    exname = 'train_jointly'
    pretraining_dir = os.path.join(output_dir, 'pretrained_weights')

    # Run the experiment if it doesn't exist
    if not os.path.exists(pretraining_dir):
        # Pretraining config
        config_updates = copy.deepcopy(psilodep2_config)
        config_updates['output_dir'] = pretraining_dir
        config_updates['save_weights'] = True

        # Make sure the model is compatible with psilodep1
        incompatible = ['Condition', 'Stop_SSRI']
        new_attrs = [attr for attr in psilodep2_config['dataset']['graph_attrs'] if attr not in incompatible]
        config_updates['dataset']['graph_attrs'] = new_attrs
        config_updates['dataset']['context_attrs'] = incompatible # add as readout-context only

        run(exname, observer, config_updates)
    else:
        print(f"Pretraining weights already exists in {pretraining_dir}.")

    # 2. Finetuning ------------------------------------------------------------
    exname = 'transfer_and_finetune'
    finetuning_dir = os.path.join(output_dir, 'finetuned_weights')
    if not os.path.exists(finetuning_dir):
        config_updates = copy.deepcopy(psilodep1_config)
        config_updates['output_dir'] = finetuning_dir
        config_updates['weights_dir'] = pretraining_dir
        run(exname, observer, config_updates)

if __name__ == "__main__":
    """
    How to run:
    python psilodep1_finetuning.py -c experiments/configs/ -o outputs/ -s 291 -v -dbg
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_dir', type=str, default='experiments/configs/', help='Path to the config directory')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/psilodep1/', help='Path to the output directory')
    parser.add_argument('-s', '--seed', type=int, default=291, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-dbg', '--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Make sure debugging outputs don't overwrite any existing outputs
    if args.debug:
        if args.output_dir == 'outputs/psilodep1/':
            raise ValueError("output_dir must be specified when using debug mode.")

    # Run the main function
    main(args.config_dir, args.output_dir, args.verbose, args.debug, args.seed)
